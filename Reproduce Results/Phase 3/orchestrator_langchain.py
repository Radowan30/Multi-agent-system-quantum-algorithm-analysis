"""
LangChain orchestrator for the Phase 3 three-agent pipeline.

Talks to a vLLM HTTP server (started by `vllm_server.VLLMServer`) through
`langchain_openai.ChatOpenAI`, which itself uses `openai.AsyncOpenAI` under
the hood — natively async, with concurrent HTTP requests handled by the
vLLM server's `AsyncLLMEngine` + continuous batching.

Compared to the vanilla orchestrator, the only difference is the
chain-composition layer: each agent is wrapped as a LangChain `Runnable`
and the three are composed with `|`. The underlying network transport,
batching, and concurrency model are identical.

This replaces the earlier `langchain_community.llms.VLLM` (in-process sync)
implementation, which deadlocked under concurrent load because the sync
wrapper serialised on vLLM's internal lock when called from multiple
`asyncio.to_thread` threads at once.

Public API:
    async def run_batch(
        base_url: str,
        served_model_name: str,
        circuits: List[Dict],     # [{"circuit_id": str, "qasm": str}, ...]
        sampling_params,
        max_concurrency: int,
        logger=None,
    ) -> List[ChainResult]
"""

import asyncio
import time
from typing import Any, Dict, List

from chain import ChainResult
from prompts import build_agent1_prompt, build_agent2_prompt, build_agent3_prompt
from parsers import parse_agent1_output, parse_agent2_output, parse_agent3_output


class _ParseFailure(Exception):
    """Raised inside an agent Runnable when its output fails to parse."""
    def __init__(self, stage: str, output_text: str):
        self.stage = stage
        self.output_text = output_text


def _build_pipeline(llm):
    """Compose the three agents as a single LangChain Runnable.

    Pipeline input shape : {"qasm": str, "result": ChainResult}
    Pipeline output shape: same dict, with `a1`/`a2`/`a3` parsed payloads
        added as keys (used by the success path) AND with `result`'s per-agent
        output / time fields populated incrementally as each agent completes.

    The ChainResult lives in the state so that each agent can write its raw
    output and timing into it *immediately* after the LLM call returns —
    BEFORE the parse step that may raise. This guarantees that on a mid-chain
    parse failure, the caller still has the partial timings of every agent
    that ran successfully, plus the raw output of the failed agent itself.
    """
    from langchain_core.messages import HumanMessage
    from langchain_core.runnables import RunnableLambda

    async def _astep_agent(stage: str, build_prompt_fn, parse_fn,
                           state: Dict[str, Any]) -> Dict[str, Any]:
        if stage == "a1":
            user_prompt = build_prompt_fn(state["qasm"])
        elif stage == "a2":
            user_prompt = build_prompt_fn(state["a1"].text)
        else:  # a3
            user_prompt = build_prompt_fn(state["a2"].text)

        t = time.perf_counter()
        ai_msg = await llm.ainvoke([HumanMessage(content=user_prompt)])
        elapsed = time.perf_counter() - t

        raw_out = ai_msg.content if isinstance(ai_msg.content, str) \
                  else "".join(part for part in ai_msg.content)

        # Write raw output + timing into the shared ChainResult IMMEDIATELY,
        # before the parse step. If parse_fn raises _ParseFailure below, the
        # caller's except handler still sees this raw output + elapsed time.
        result = state["result"]
        setattr(result, f"{stage}_output", raw_out)
        setattr(result, f"{stage}_time_s", elapsed)

        parsed = parse_fn(raw_out)
        if parsed is None:
            raise _ParseFailure(stage=stage.upper(), output_text=raw_out)

        # Also write parsed structured payload immediately on success.
        if stage == "a1":
            result.extracted_oracle = parsed.oracle_block
        elif stage == "a2":
            result.marked_states = parsed.marked_states
        elif stage == "a3":
            result.probability_dict = parsed.probability_dict

        return {**state, stage: parsed}

    a1_step = RunnableLambda(
        lambda s: _astep_agent("a1", build_agent1_prompt, parse_agent1_output, s),
        afunc=lambda s: _astep_agent("a1", build_agent1_prompt, parse_agent1_output, s),
    )
    a2_step = RunnableLambda(
        lambda s: _astep_agent("a2", build_agent2_prompt, parse_agent2_output, s),
        afunc=lambda s: _astep_agent("a2", build_agent2_prompt, parse_agent2_output, s),
    )
    a3_step = RunnableLambda(
        lambda s: _astep_agent("a3", build_agent3_prompt, parse_agent3_output, s),
        afunc=lambda s: _astep_agent("a3", build_agent3_prompt, parse_agent3_output, s),
    )
    return a1_step | a2_step | a3_step


async def _run_one_chain(pipeline, circuit_id: str, qasm: str, logger) -> ChainResult:
    """Invoke the LangChain pipeline on one circuit, populate a ChainResult.

    The pipeline writes its per-agent raw output, parsed payload and timing
    into the shared ChainResult immediately as each agent completes (see
    `_build_pipeline`), so this function only needs to:
      - construct the ChainResult and hand it to the pipeline via state
      - on `_ParseFailure`, mark failure_stage (the result is already
        populated for the agents that completed successfully)
      - on success, mark `success = True`
      - either way, stamp the total wall-clock and log
    """
    result = ChainResult(success=False)
    t_start = time.perf_counter()

    try:
        await pipeline.ainvoke({"qasm": qasm, "result": result})
    except _ParseFailure as e:
        # Partial agent data (raw outputs + timings of successful prior agents,
        # plus the failed agent's raw output) is already on `result`.
        result.failure_stage = e.stage
        result.total_time_s = time.perf_counter() - t_start
        if logger: logger.log_chain(circuit_id, qasm, result)
        return result
    except Exception as e:
        # Any other failure — vLLM 400 (prompt + max_tokens exceeds
        # max_model_len), network blip, server error, timeout, etc. Record
        # the failure on the agent that was running and continue. Without
        # this catch, a single oversized circuit would abort asyncio.gather
        # and tank the entire run.
        #
        # Identify which agent was running based on what's already populated.
        # The agent that's currently running is the next one with no a*_output
        # set yet. (Per-agent outputs are written inside _astep_agent BEFORE
        # the exception can fire, so a populated a*_output means that agent
        # completed; an empty one means it never ran or was the one that failed.)
        if result.a1_output is None:
            stage = "A1"
            result.a1_output = f"[orchestrator error: {type(e).__name__}: {e}]"
        elif result.a2_output is None:
            stage = "A2"
            result.a2_output = f"[orchestrator error: {type(e).__name__}: {e}]"
        else:
            stage = "A3"
            result.a3_output = f"[orchestrator error: {type(e).__name__}: {e}]"
        result.failure_stage = stage
        result.total_time_s = time.perf_counter() - t_start
        if logger: logger.log_chain(circuit_id, qasm, result)
        return result

    result.success = True
    result.total_time_s = time.perf_counter() - t_start
    if logger: logger.log_chain(circuit_id, qasm, result)
    return result


async def run_batch(
    base_url: str,
    served_model_name: str,
    circuits: List[Dict],
    sampling_params,
    max_concurrency: int = 32,
    logger=None,
) -> List[ChainResult]:
    """Run the agent chain on all circuits concurrently via LangChain
    Runnable composition + asyncio.gather over the per-circuit pipeline.
    """
    from langchain_openai import ChatOpenAI

    stop_strings = list(sampling_params.stop or [])
    stop_token_ids = list(sampling_params.stop_token_ids or [])

    # ChatOpenAI talks to the vLLM server's OpenAI-compatible /v1/chat/completions
    # endpoint. vLLM-specific extensions like `stop_token_ids` are passed via
    # `extra_body` (the OpenAI client tunnels these as extra JSON fields).
    llm = ChatOpenAI(
        model=served_model_name,
        api_key="EMPTY",
        base_url=base_url,
        temperature=sampling_params.temperature,
        max_tokens=sampling_params.max_tokens,
        stop=stop_strings,
        extra_body={"stop_token_ids": stop_token_ids} if stop_token_ids else None,
        timeout=600.0,
        max_retries=2,
    )

    pipeline = _build_pipeline(llm)
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _bounded(ckt: Dict) -> ChainResult:
        async with semaphore:
            return await _run_one_chain(pipeline, ckt["circuit_id"], ckt["qasm"], logger)

    tasks = [_bounded(c) for c in circuits]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    return results
