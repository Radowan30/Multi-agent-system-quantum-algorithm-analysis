"""
Evaluation runner for the Phase 4 monolithic ablation model.

The monolithic model receives the bare QASM circuit once and generates, in a
single call, the concatenation of the three Phase 4 agent outputs. This
runner splits that single output back into the three agent sections, parses
each section with the Phase 3/4 parsers, and writes the SAME files as the
Phase 4 multi-agent runner ("Phase 4/run_eval.py"):

    chain_results.jsonl       per-circuit ChainResult records
    per_circuit_metrics.jsonl per-circuit SA, CF, oracle_correct
    aggregated_metrics.json   per-n aggregates
    cot_agents.md             human-readable section traces

so that "Phase 4/paper_eval_postprocess.py" and the plot
scripts run on the monolithic results unchanged. Per-circuit metrics are
computed by importing `compute_circuit_metrics` and `aggregate_per_n` from
the Phase 4 runner itself, so the metric code is identical.

Settings kept identical to the Phase 4 runner: vLLM server and dtype,
temperature 0, the same stop strings and stop token ids, the same per-call
cap of 8192 new tokens (the longest correct monolithic output at n = 19 is
~5,000 tokens by extrapolation, so the cap never truncates a correct
answer), the same circuit sets and concurrencies (set by the calling script).

Section split. The three agent targets are concatenated with no separator,
and each section starts with a fixed header:
    Agent 1 section: from the start of the output
    Agent 2 section: from "The Oracle entity is extracted below:"
    Agent 3 section: from "=== Probability Reasoning ==="
Splitting before parsing is required: without it, the last marked-state
bitstring of the Agent 2 section runs into "=== Probability Reasoning ==="
on the same line, and the Agent 2 parser would drop it.

Scoring semantics. The chain stops at the first agent whose output cannot be
parsed, because the next agent has no input. A single generation has no
such dependency: its final probability dictionary exists (or not)
regardless of the earlier sections. So each section is parsed
independently: oracle_correct comes from the Agent 1 section, marked states
from the Agent 2 section, and SA / CF from the final probability
dictionary (the Phase 1/2 convention for monolithic outputs).
`failure_stage` records the first section that failed to parse, and
`success` is True only if all three sections parsed.

Timing. `total_time_s` is the wall-clock time of the single call and is
what the RET computation reads. The per-agent time fields are 0.0 because a
single generation cannot be split into per-agent durations.
"""

import argparse
import asyncio
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

_HERE = Path(__file__).resolve().parent
_PHASE4 = _HERE.parent                      # Reproduce Results/Phase 4
_PHASE3 = _PHASE4.parent / "Phase 3"        # Reproduce Results/Phase 3

# Reuse the Phase 4 runner's metric functions verbatim. Loading it also puts
# evaluation_pipeline/ and the Phase 3 folder on sys.path.
_spec = importlib.util.spec_from_file_location("phase4_run_eval", _PHASE4 / "run_eval.py")
_phase4_run_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_phase4_run_eval)
compute_circuit_metrics = _phase4_run_eval.compute_circuit_metrics
aggregate_per_n = _phase4_run_eval.aggregate_per_n

if str(_PHASE3) not in sys.path:
    sys.path.append(str(_PHASE3))
from chain import ChainResult  # noqa: E402
from cot_logger import CotLogger  # noqa: E402
from parsers import parse_agent1_output, parse_agent2_output, parse_agent3_output  # noqa: E402
from vllm_server import VLLMServer  # noqa: E402

A2_SECTION_START = "The Oracle entity is extracted below:"
A3_SECTION_START = "=== Probability Reasoning ==="

# Identical to the Phase 4 runner's SamplingParams.
STOP_STRINGS = ["<|eot_id|>", "<|end_of_text|>", "=== END ==="]
STOP_TOKEN_IDS = [128001, 128009]


def split_sections(text: str):
    """Return (a1_text, a2_text, a3_text); a missing section is None."""
    i2 = text.find(A2_SECTION_START)
    if i2 < 0:
        return text, None, None
    i3 = text.find(A3_SECTION_START, i2)
    if i3 < 0:
        return text[:i2], text[i2:], None
    return text[:i2], text[i2:i3], text[i3:]


def build_result(output: str, elapsed_s: float) -> ChainResult:
    """Parse one monolithic output into a ChainResult."""
    a1_text, a2_text, a3_text = split_sections(output)
    result = ChainResult(success=False)
    result.a1_output = a1_text
    result.a2_output = a2_text
    result.a3_output = a3_text
    result.total_time_s = elapsed_s

    a1 = parse_agent1_output(a1_text)
    a2 = parse_agent2_output(a2_text) if a2_text is not None else None
    a3 = parse_agent3_output(a3_text if a3_text is not None else output)

    if a1 is not None:
        result.extracted_oracle = a1.oracle_block
    if a2 is not None:
        result.marked_states = a2.marked_states
    if a3 is not None:
        result.probability_dict = a3.probability_dict

    for stage, parsed in (("A1", a1), ("A2", a2), ("A3", a3)):
        if parsed is None:
            result.failure_stage = stage
            break
    result.success = result.failure_stage is None
    return result


async def run_batch(base_url: str, model_name: str, circuits: List[Dict],
                    max_tokens: int, max_concurrency: int, logger) -> List[Dict]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY",
                         # One call does the work of three chain calls, so allow
                         # three times the chain's 600 s per call. Retries match
                         # the chain's ChatOpenAI client (max_retries=2).
                         timeout=1800.0, max_retries=2)
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _one(ckt: Dict) -> Dict:
        async with semaphore:
            prompt = ckt["qasm"].rstrip()   # identical to Phase 4's Agent 1 input
            t0 = time.perf_counter()
            try:
                resp = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=max_tokens,
                    stop=STOP_STRINGS,
                    extra_body={"stop_token_ids": STOP_TOKEN_IDS},
                )
                output = resp.choices[0].message.content or ""
                finish = resp.choices[0].finish_reason
                n_out = resp.usage.completion_tokens if resp.usage else None
            except Exception as e:  # recorded as a failed circuit, as in the chain
                output = f"[runner error: {type(e).__name__}: {e}]"
                finish, n_out = "error", None
            elapsed = time.perf_counter() - t0
            result = build_result(output, elapsed)
            if logger:
                logger.log_chain(ckt["circuit_id"], ckt["qasm"], result)
            return {"result": result, "raw_output": output,
                    "finish_reason": finish, "completion_tokens": n_out}

    records = await asyncio.gather(*[_one(c) for c in circuits])
    await client.close()
    return records


async def _amain():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--circuits_jsonl", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_model_len", type=int, default=150000)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--max_concurrency", type=int, default=8)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or Path(args.model_path).name

    circuits = []
    with open(args.circuits_jsonl) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                rec["qasm"] = Path(rec["qasm_path"]).read_text()
                circuits.append(rec)
    print(f"[{label}|monolithic] Loaded {len(circuits)} circuits", flush=True)

    logger = CotLogger(str(out_dir / "cot_agents.md"))
    served_model_name = Path(args.model_path).name
    async with VLLMServer(model_path=args.model_path,
                          served_model_name=served_model_name,
                          max_model_len=args.max_model_len,
                          gpu_memory_utilization=args.gpu_memory_utilization) as server:
        print(f"[{label}|monolithic] Running {len(circuits)} circuits at "
              f"concurrency={args.max_concurrency}", flush=True)
        t_start = time.perf_counter()
        records = await run_batch(server.base_url, served_model_name, circuits,
                                  args.max_tokens, args.max_concurrency, logger)
        eval_time = time.perf_counter() - t_start

    per_circuit_metrics = []
    with open(out_dir / "chain_results.jsonl", "w") as f:
        for ckt, rec in zip(circuits, records):
            result = rec["result"]
            f.write(json.dumps({
                "circuit_id": ckt["circuit_id"],
                "n": ckt["n"],
                "k": ckt["k"],
                "marked_states": ckt["marked_states"],
                "chain_result": result.to_dict(),
                "monolithic_output": rec["raw_output"],
                "finish_reason": rec["finish_reason"],
                "completion_tokens": rec["completion_tokens"],
            }) + "\n")
            metrics = compute_circuit_metrics(
                n=ckt["n"],
                true_marked_states=ckt["marked_states"],
                true_oracle_qasm=ckt["qasm"],
                chain_result=result,
            )
            metrics.update({"circuit_id": ckt["circuit_id"], "n": ckt["n"], "k": ckt["k"]})
            per_circuit_metrics.append(metrics)

    with open(out_dir / "per_circuit_metrics.jsonl", "w") as f:
        for rec in per_circuit_metrics:
            f.write(json.dumps(rec) + "\n")

    aggregated = aggregate_per_n(per_circuit_metrics)
    with open(out_dir / "aggregated_metrics.json", "w") as f:
        json.dump({
            "label": label,
            "orchestrator": "monolithic",
            "model_path": args.model_path,
            "n_circuits": len(circuits),
            "total_eval_time_s": eval_time,
            "per_n": {str(n): v for n, v in aggregated.items()},
        }, f, indent=2)

    print(f"\n[{label}|monolithic] === Per-n summary ===")
    print(f"{'n':>3} {'circ':>5} {'succ':>4} {'sa':>6} {'cf_raw':>6} {'ora':>5}")
    for n, agg in aggregated.items():
        print(f"{n:>3} {agg['n_circuits']:>5} {agg['n_chain_success']:>4} "
              f"{agg['sa'][0]:>6.3f} {agg['cf_raw'][0]:>6.3f} {agg['oracle_accuracy']:>5.3f}")
    print(f"\n[{label}|monolithic] Done in {eval_time/60:.1f} min. Results -> {out_dir}")


if __name__ == "__main__":
    asyncio.run(_amain())
