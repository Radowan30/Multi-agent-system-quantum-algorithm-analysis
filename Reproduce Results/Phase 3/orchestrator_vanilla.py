"""
Vanilla orchestrator for the Phase 3 three-agent pipeline — pure async,
no framework.

Talks to a vLLM HTTP server (started by `vllm_server.VLLMServer`) using
`openai.AsyncOpenAI`. The vLLM server runs `AsyncLLMEngine` natively and
batches incoming concurrent requests via its continuous-batching scheduler,
so the only thing this orchestrator has to do is fire many concurrent
async chains via `asyncio.gather` + a semaphore for back-pressure.

vLLM optimizations:
  - prefix caching (set on the server, applies automatically)
  - continuous batching (server-side)
  - paged attention (server-side)

Public API:
    async def run_batch(
        base_url: str,
        served_model_name: str,
        circuits: List[Dict],     # [{"circuit_id": str, "qasm": str}, ...]
        sampling_params,           # vllm.SamplingParams (only the fields we need are read)
        max_concurrency: int,
        logger=None,
    ) -> List[ChainResult]
"""

import asyncio
import time
from typing import Dict, List

from chain import ChainResult
from prompts import build_agent1_prompt, build_agent2_prompt, build_agent3_prompt
from parsers import parse_agent1_output, parse_agent2_output, parse_agent3_output


async def _call_llm(client, model_name: str, user_prompt: str,
                    stop_strings: List[str], stop_token_ids: List[int],
                    max_tokens: int, temperature: float) -> str:
    """Send a single user prompt to the vLLM server, return assistant text.

    vLLM-specific kwargs (`stop_token_ids`) go through `extra_body` because
    the OpenAI API spec doesn't natively define them.
    """
    resp = await client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop_strings,
        extra_body={"stop_token_ids": stop_token_ids} if stop_token_ids else {},
    )
    return resp.choices[0].message.content or ""


async def _run_one_chain(
    client, model_name: str,
    stop_strings: List[str], stop_token_ids: List[int],
    max_tokens: int, temperature: float,
    qasm: str, circuit_id: str, logger,
) -> ChainResult:
    """Run A1 → A2 → A3 sequentially for a single circuit (async)."""
    result = ChainResult(success=False)
    t_start = time.perf_counter()

    # Agent 1
    t = time.perf_counter()
    try:
        a1_output = await _call_llm(
            client, model_name, build_agent1_prompt(qasm),
            stop_strings, stop_token_ids, max_tokens, temperature,
        )
    except Exception as e:
        result.failure_stage = "A1"
        result.a1_output = f"[orchestrator error: {type(e).__name__}: {e}]"
        result.a1_time_s = time.perf_counter() - t
        result.total_time_s = time.perf_counter() - t_start
        if logger: logger.log_chain(circuit_id, qasm, result)
        return result
    result.a1_output = a1_output
    result.a1_time_s = time.perf_counter() - t

    a1_parsed = parse_agent1_output(a1_output)
    if a1_parsed is None:
        result.failure_stage = "A1"
        result.total_time_s = time.perf_counter() - t_start
        if logger: logger.log_chain(circuit_id, qasm, result)
        return result
    result.extracted_oracle = a1_parsed.oracle_block

    # Agent 2
    t = time.perf_counter()
    try:
        a2_output = await _call_llm(
            client, model_name, build_agent2_prompt(a1_parsed.text),
            stop_strings, stop_token_ids, max_tokens, temperature,
        )
    except Exception as e:
        result.failure_stage = "A2"
        result.a2_output = f"[orchestrator error: {type(e).__name__}: {e}]"
        result.a2_time_s = time.perf_counter() - t
        result.total_time_s = time.perf_counter() - t_start
        if logger: logger.log_chain(circuit_id, qasm, result)
        return result
    result.a2_output = a2_output
    result.a2_time_s = time.perf_counter() - t

    a2_parsed = parse_agent2_output(a2_output)
    if a2_parsed is None:
        result.failure_stage = "A2"
        result.total_time_s = time.perf_counter() - t_start
        if logger: logger.log_chain(circuit_id, qasm, result)
        return result
    result.marked_states = a2_parsed.marked_states

    # Agent 3
    t = time.perf_counter()
    try:
        a3_output = await _call_llm(
            client, model_name, build_agent3_prompt(a2_parsed.text),
            stop_strings, stop_token_ids, max_tokens, temperature,
        )
    except Exception as e:
        result.failure_stage = "A3"
        result.a3_output = f"[orchestrator error: {type(e).__name__}: {e}]"
        result.a3_time_s = time.perf_counter() - t
        result.total_time_s = time.perf_counter() - t_start
        if logger: logger.log_chain(circuit_id, qasm, result)
        return result
    result.a3_output = a3_output
    result.a3_time_s = time.perf_counter() - t

    a3_parsed = parse_agent3_output(a3_output)
    if a3_parsed is None:
        result.failure_stage = "A3"
        result.total_time_s = time.perf_counter() - t_start
        if logger: logger.log_chain(circuit_id, qasm, result)
        return result
    result.probability_dict = a3_parsed.probability_dict

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
    """Run the agent chain on all circuits concurrently via async HTTP."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url=base_url,
        api_key="EMPTY",
        timeout=600.0,          # long generations at n=19 can take minutes
        max_retries=2,
    )

    stop_strings = list(sampling_params.stop or [])
    stop_token_ids = list(sampling_params.stop_token_ids or [])
    max_tokens = sampling_params.max_tokens
    temperature = sampling_params.temperature

    semaphore = asyncio.Semaphore(max_concurrency)

    async def _bounded(ckt: Dict) -> ChainResult:
        async with semaphore:
            return await _run_one_chain(
                client, served_model_name,
                stop_strings, stop_token_ids, max_tokens, temperature,
                qasm=ckt["qasm"], circuit_id=ckt["circuit_id"], logger=logger,
            )

    tasks = [_bounded(c) for c in circuits]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    await client.close()
    return results
