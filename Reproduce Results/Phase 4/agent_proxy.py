"""
OpenAI-compatible HTTP proxy in front of the Phase 4 three-agent pipeline.

Exposes `/v1/chat/completions` and `/v1/models` so any chat client that speaks
the OpenAI API (Theia IDE's @theia/ai-openai provider, Open WebUI, LibreChat,
Continue.dev, Cline, ChatBox, …) can talk to the multi-agent system as if it
were a single LLM. Internally each user request fans out into the A1 → A2 → A3
chain against a vLLM server hosting the Phase 4 merged model.

Output format (streamed token-by-token when the client requests streaming):

    **Agent 1 reasoning (Oracle Extractor):**
    <A1 raw output>

    ---

    **Agent 2 reasoning (Marked-State Identifier):**
    <A2 raw output>

    ---

    **Agent 3 reasoning (Probability Distribution):**
    <A3 raw output>

Design notes:
  - System prompts sent by the client are IGNORED. Phase 4 was trained with
    no system prompt; injecting one at inference would shift the input
    distribution off-training and degrade quality.
  - QASM extraction is tolerant of common chat-UI input shapes: bare paste,
    markdown code fences (```qasm / ```openqasm / ```), prose surrounding
    a QASM block. The proxy locates the OPENQASM header and the `gate Oracle`
    definition; if both are present the block from `OPENQASM` onward is
    forwarded to the chain.
  - Inputs that are not valid Grover circuits get a friendly error message
    delivered as a chat response (so the user sees it naturally in their UI),
    NOT an HTTP error code.
  - Mid-chain agent parse failures (e.g. Agent 1 produced an unparseable
    oracle) also surface as a chat response — the partial output the user
    has already received is appended with an explanatory note.

Run (vLLM must already be serving the Phase 4 merged model — `run_agent_proxy.sh`
does both in one go):

    python agent_proxy.py \\
        --vllm_url http://localhost:8000/v1 \\
        --vllm_model Phase4_alpha32 \\
        --proxy_model Phase4_Quantum_Agents \\
        --port 8080
"""

import argparse
import asyncio
import json
import re
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
from openai import AsyncOpenAI

# ─── Reuse Phase 3 prompts + parsers (Phase 4 prompts are minimal-identity) ───
_HERE = Path(__file__).resolve().parent
_PHASE3 = _HERE.parent / "Multi Agent System Phase 3"
sys.path.insert(0, str(_HERE))      # Phase 4 prompts first (identity)
sys.path.insert(0, str(_PHASE3))    # then Phase 3 (parsers, etc.)

from prompts import (  # noqa: E402  — Phase 4 minimal identity prompts
    build_agent1_prompt,
    build_agent2_prompt,
    build_agent3_prompt,
)
from parsers import (  # noqa: E402  — Phase 3 parsers
    parse_agent1_output,
    parse_agent2_output,
)


# ─── Sampling parameters (match the Phase 4 eval exactly) ─────────────────────
SAMPLING = dict(
    temperature=0.0,
    max_tokens=8192,
    stop=["<|eot_id|>", "<|end_of_text|>"],
    extra_body={"stop_token_ids": [128001, 128009]},
)


# Globals populated by CLI args at startup (single-process server, this is fine).
class Config:
    vllm_url:    str = "http://localhost:8000/v1"
    vllm_model:  str = "Phase4_alpha32"
    proxy_model: str = "Phase4_Quantum_Agents"


# ─── QASM extraction — tolerant of common chat-UI input shapes ────────────────

# Match the first markdown code fence (```qasm / ```openqasm / plain ```).
# DOTALL so the body can span newlines; non-greedy so we stop at the first ```.
_CODE_FENCE_RE = re.compile(
    r"```(?:qasm|openqasm|qsharp|q)?\s*\n?(.*?)\n?```",
    re.DOTALL | re.IGNORECASE,
)
# Single backticks: `OPENQASM ...`
_BACKTICK_RE = re.compile(r"`([^`]+OPENQASM[^`]+)`", re.IGNORECASE | re.DOTALL)


def extract_qasm(text: str) -> Optional[str]:
    """Try to recover usable QASM from arbitrary user input.

    Returns the trimmed QASM block on success, None if no valid Grover circuit
    can be found. Validation requires both an `OPENQASM` header and a
    `gate Oracle` definition.

    Tolerated shapes:
      - Bare paste of the QASM file contents
      - Single markdown code fence (``` ... ``` with optional language hint)
      - QASM block with surrounding prose ("Here is my circuit:\\n```qasm...```")
      - Inline single-backtick block (rare but seen)
    """
    if not text:
        return None

    # 1) If a code fence is present, use its contents.
    fence = _CODE_FENCE_RE.search(text)
    if fence:
        text = fence.group(1)
    else:
        # 2) Try single-backtick inline block as a fallback.
        bt = _BACKTICK_RE.search(text)
        if bt:
            text = bt.group(1)

    # 3) Locate the OPENQASM header and trim everything before it.
    upper = text.upper()
    idx = upper.find("OPENQASM")
    if idx < 0:
        return None
    text = text[idx:].strip()

    # 4) Validate the structural requirement: a Grover circuit needs a
    # `gate Oracle` definition. Without it, the chain cannot proceed.
    if not re.search(r"gate\s+Oracle\b", text):
        return None

    return text


# ─── OpenAI streaming-chunk helpers ───────────────────────────────────────────

def _now() -> int:
    return int(time.time())


def _role_chunk(response_id: str, model: str) -> dict:
    """Initial SSE chunk that sets role=assistant (per OpenAI spec)."""
    return {
        "id":      response_id,
        "object":  "chat.completion.chunk",
        "created": _now(),
        "model":   model,
        "choices": [{
            "index":         0,
            "delta":         {"role": "assistant"},
            "finish_reason": None,
        }],
    }


def _content_chunk(text: str, response_id: str, model: str) -> dict:
    return {
        "id":      response_id,
        "object":  "chat.completion.chunk",
        "created": _now(),
        "model":   model,
        "choices": [{
            "index":         0,
            "delta":         {"content": text},
            "finish_reason": None,
        }],
    }


def _end_chunk(response_id: str, model: str, finish_reason: str = "stop") -> dict:
    return {
        "id":      response_id,
        "object":  "chat.completion.chunk",
        "created": _now(),
        "model":   model,
        "choices": [{
            "index":         0,
            "delta":         {},
            "finish_reason": finish_reason,
        }],
    }


def _sse(chunk: dict) -> str:
    return f"data: {json.dumps(chunk)}\n\n"


# ─── Agent-chain streaming ────────────────────────────────────────────────────

# Per-agent header text, emitted between agents so the reader sees the chain
# structure interleaved with the agent outputs.
A1_HEADER = "**Agent 1 reasoning (Oracle Extractor):**\n\n"
A2_HEADER = "\n\n---\n\n**Agent 2 reasoning (Marked-State Identifier):**\n\n"
A3_HEADER = "\n\n---\n\n**Agent 3 reasoning (Probability Distribution):**\n\n"
ERR_PREFIX = "\n\n---\n\n**Pipeline error:** "


async def _stream_one_agent(
    client: AsyncOpenAI, user_prompt: str,
) -> AsyncIterator[Tuple[Optional[str], bool, str]]:
    """Stream one agent call against vLLM. Yields (delta, is_final, full_text).

    Tuple semantics:
      - (delta, False, "")      — a content-delta chunk; forward to the client
      - (None,  True,  full)    — final yield with the complete assembled text
    """
    full = ""
    stream = await client.chat.completions.create(
        model=Config.vllm_model,
        messages=[{"role": "user", "content": user_prompt}],
        stream=True,
        **SAMPLING,
    )
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            full += delta
            yield delta, False, ""
    yield None, True, full


async def _stream_chain(qasm: str, response_id: str, model: str) -> AsyncIterator[str]:
    """Run the full A1 → A2 → A3 chain, yielding SSE-formatted chunks suitable
    for direct write to the response. Includes per-agent header text and
    graceful error reporting on mid-chain failures."""
    client = AsyncOpenAI(base_url=Config.vllm_url, api_key="EMPTY",
                        timeout=600.0, max_retries=2)
    try:
        # Role chunk
        yield _sse(_role_chunk(response_id, model))

        # ── Agent 1 ──
        yield _sse(_content_chunk(A1_HEADER, response_id, model))
        a1_full = ""
        async for delta, is_final, full in _stream_one_agent(client, build_agent1_prompt(qasm)):
            if is_final:
                a1_full = full
            else:
                yield _sse(_content_chunk(delta, response_id, model))

        a1 = parse_agent1_output(a1_full)
        if a1 is None:
            yield _sse(_content_chunk(
                ERR_PREFIX + "Agent 1 did not produce a parseable oracle gate "
                "definition. The chain cannot proceed past oracle extraction. "
                "This usually indicates the input circuit is far enough out of "
                "the training distribution that the model lost its output "
                "format; try a smaller circuit (n ≤ 12).",
                response_id, model))
            yield _sse(_end_chunk(response_id, model))
            yield "data: [DONE]\n\n"
            return

        # ── Agent 2 ──
        yield _sse(_content_chunk(A2_HEADER, response_id, model))
        a2_full = ""
        async for delta, is_final, full in _stream_one_agent(client, build_agent2_prompt(a1.text)):
            if is_final:
                a2_full = full
            else:
                yield _sse(_content_chunk(delta, response_id, model))

        a2 = parse_agent2_output(a2_full)
        if a2 is None:
            yield _sse(_content_chunk(
                ERR_PREFIX + "Agent 2 did not produce a parseable marked-state "
                "list (could not locate the `=== Final Marked States ===` "
                "marker or the marked-state bitstrings have inconsistent "
                "length). The probability-distribution stage is skipped.",
                response_id, model))
            yield _sse(_end_chunk(response_id, model))
            yield "data: [DONE]\n\n"
            return

        # ── Agent 3 ──
        yield _sse(_content_chunk(A3_HEADER, response_id, model))
        async for delta, is_final, _full in _stream_one_agent(client, build_agent3_prompt(a2.text)):
            if not is_final and delta:
                yield _sse(_content_chunk(delta, response_id, model))

        # Done
        yield _sse(_end_chunk(response_id, model))
        yield "data: [DONE]\n\n"

    except Exception as e:
        # Surface any transport / vLLM error as a chat message rather than
        # truncating the response with no explanation.
        yield _sse(_content_chunk(
            ERR_PREFIX + f"Unexpected error during inference: "
            f"{type(e).__name__}: {e}",
            response_id, model))
        yield _sse(_end_chunk(response_id, model, finish_reason="stop"))
        yield "data: [DONE]\n\n"
    finally:
        await client.close()


# ─── Non-streaming path ───────────────────────────────────────────────────────

async def _run_chain_blocking(qasm: str) -> Tuple[str, str]:
    """Run the A1 → A2 → A3 chain without streaming. Returns
    (assembled_response_text, finish_reason). Used when client sends
    `stream=false`."""
    client = AsyncOpenAI(base_url=Config.vllm_url, api_key="EMPTY",
                        timeout=600.0, max_retries=2)
    try:
        parts: List[str] = []

        # A1
        parts.append(A1_HEADER)
        r1 = await client.chat.completions.create(
            model=Config.vllm_model,
            messages=[{"role": "user", "content": build_agent1_prompt(qasm)}],
            **SAMPLING,
        )
        a1_full = r1.choices[0].message.content or ""
        parts.append(a1_full)

        a1 = parse_agent1_output(a1_full)
        if a1 is None:
            parts.append(ERR_PREFIX + "Agent 1 did not produce a parseable oracle.")
            return "".join(parts), "stop"

        # A2
        parts.append(A2_HEADER)
        r2 = await client.chat.completions.create(
            model=Config.vllm_model,
            messages=[{"role": "user", "content": build_agent2_prompt(a1.text)}],
            **SAMPLING,
        )
        a2_full = r2.choices[0].message.content or ""
        parts.append(a2_full)

        a2 = parse_agent2_output(a2_full)
        if a2 is None:
            parts.append(ERR_PREFIX + "Agent 2 did not produce a parseable marked-state list.")
            return "".join(parts), "stop"

        # A3
        parts.append(A3_HEADER)
        r3 = await client.chat.completions.create(
            model=Config.vllm_model,
            messages=[{"role": "user", "content": build_agent3_prompt(a2.text)}],
            **SAMPLING,
        )
        parts.append(r3.choices[0].message.content or "")

        return "".join(parts), "stop"
    finally:
        await client.close()


# ─── Error-as-chat helpers ────────────────────────────────────────────────────

def _error_chat_completion(message: str, stream: bool, model: str) -> "FastAPIResponse":
    """Deliver a validation / pre-flight error as if it were a normal
    chat completion. Better UX than HTTP 4xx for chat UIs."""
    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    if stream:
        async def gen():
            yield _sse(_role_chunk(response_id, model))
            yield _sse(_content_chunk(message, response_id, model))
            yield _sse(_end_chunk(response_id, model))
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    return JSONResponse({
        "id":      response_id,
        "object":  "chat.completion",
        "created": _now(),
        "model":   model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": message},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })


# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(title="Phase 4 Quantum Agents — OpenAI-compatible proxy")


@app.get("/v1/models")
async def list_models() -> dict:
    """Required by Theia (and most clients) for model discovery."""
    return {
        "object": "list",
        "data": [{
            "id":       Config.proxy_model,
            "object":   "model",
            "created":  _now(),
            "owned_by": "phase-4",
        }],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    payload = await request.json()
    messages = payload.get("messages", []) or []
    stream   = bool(payload.get("stream", False))

    # Extract the LAST user message; ignore everything else (including any
    # system prompt the client may have prepended). Phase 4 was trained with
    # no system prompt, so this prevents distribution shift.
    user_text = ""
    for msg in messages:
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            user_text = msg["content"]

    if not user_text.strip():
        return _error_chat_completion(
            "No user message was found in the request. Please send the Grover "
            "circuit you'd like analysed (paste the full QASM 3.0 program — "
            "optionally inside a ```qasm code fence — and I'll run the "
            "three-agent pipeline on it).",
            stream, Config.proxy_model,
        )

    qasm = extract_qasm(user_text)
    if qasm is None:
        return _error_chat_completion(
            "I couldn't find a valid Grover circuit in OpenQASM 3.0 format in "
            "your message. To analyse a circuit, please paste the full QASM "
            "program — it must include the `OPENQASM` header and a "
            "`gate Oracle` definition. You can either paste it directly or "
            "wrap it in a markdown code fence (```qasm ... ```). Any prose "
            "around the code block is fine — I'll extract the QASM "
            "automatically.",
            stream, Config.proxy_model,
        )

    response_id = f"chatcmpl-{uuid.uuid4().hex}"

    if stream:
        return StreamingResponse(
            _stream_chain(qasm, response_id, Config.proxy_model),
            media_type="text/event-stream",
        )

    # Non-streaming path
    assembled, finish_reason = await _run_chain_blocking(qasm)
    return JSONResponse({
        "id":      response_id,
        "object":  "chat.completion",
        "created": _now(),
        "model":   Config.proxy_model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": assembled},
            "finish_reason": finish_reason,
        }],
        "usage": {
            # We don't have exact upstream token counts cheap-to-hand;
            # report zeros (clients use these for cost display only).
            "prompt_tokens":     0,
            "completion_tokens": 0,
            "total_tokens":      0,
        },
    })


# Convenience: serve the OpenAI-style health endpoint Theia probes on startup.
@app.get("/health")
@app.get("/v1/health")
async def health() -> dict:
    return {"status": "ok", "model": Config.proxy_model}


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--vllm_url",
                    default="http://localhost:8000/v1",
                    help="Base URL of the underlying vLLM server (default: http://localhost:8000/v1)")
    ap.add_argument("--vllm_model",
                    default="Phase4_alpha32",
                    help="The `served-model-name` the underlying vLLM is using")
    ap.add_argument("--proxy_model",
                    default="Phase4_Quantum_Agents",
                    help="The model name this proxy advertises to chat clients")
    ap.add_argument("--host", default="0.0.0.0",
                    help="Host to bind on (default: 0.0.0.0 — accessible from other machines)")
    ap.add_argument("--port", type=int, default=8080,
                    help="Port to listen on (default: 8080)")
    args = ap.parse_args()

    Config.vllm_url    = args.vllm_url
    Config.vllm_model  = args.vllm_model
    Config.proxy_model = args.proxy_model

    print(f"[agent_proxy] vLLM backend  : {Config.vllm_url}")
    print(f"[agent_proxy] backend model : {Config.vllm_model}")
    print(f"[agent_proxy] advertised as : {Config.proxy_model}")
    print(f"[agent_proxy] listening on  : http://{args.host}:{args.port}")
    print(f"[agent_proxy] point clients at: http://<this-host>:{args.port}/v1")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
