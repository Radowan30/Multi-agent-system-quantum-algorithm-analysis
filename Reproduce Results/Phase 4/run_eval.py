"""
Phase 4 agent-pipeline evaluation runner.

Copy of Phase 3's run_eval.py, dropped into the Phase 4 folder so that the
`from prompts import ...` inside Phase 3's orchestrator resolves to Phase
4's minimal identity builders (which sit next to this file) instead of
Phase 3's prompt-engineered templates. Everything else — the vLLM server
lifecycle, orchestrator choice, ChainResult metric computation,
cot_agents.md logging — is reused verbatim from Phase 3's modules
(chain.py, cot_logger.py, parsers.py, orchestrator_vanilla/langchain.py,
vllm_server.py) by appending the Phase 3 directory to sys.path AFTER
this directory.

Why a copy and not a wrapper: a wrapper hit a circular import (the
wrapper file is itself named `run_eval.py` and `sys.path.insert(0,
Phase4)` made `from run_eval import main` resolve back to the wrapper).
A copy sidesteps the issue entirely and keeps Phase 4 self-contained.

Loads a circuit set, starts a vLLM OpenAI-compatible HTTP server, runs the
three-agent chain via the chosen orchestrator (vanilla or langchain), computes
SA / CF / Oracle Extraction Accuracy, and writes results + raw outputs +
cot_agents.md to the output directory.

Both orchestrators talk to the same vLLM HTTP server (started here, torn down
on exit). The server runs `AsyncLLMEngine` natively so concurrent requests
from `asyncio.gather` get continuous-batched by vLLM. Same vLLM
optimizations (prefix caching, continuous batching, paged attention) apply
to both. The chain output shape (ChainResult) is identical, so all downstream
metric / logging code is orchestrator-agnostic.

Usage:
    python run_eval.py \\
        --model_path /path/to/model \\
        --circuits_jsonl circuits.jsonl \\
        --output_dir ./out \\
        --orchestrator vanilla \\
        [--max_model_len 122880] \\
        [--gpu_memory_utilization 0.90] \\
        [--max_tokens 8192] \\
        [--max_concurrency 64] \\
        [--label "untrained_llama31"]

Circuit JSONL format (one circuit per line):
    {"circuit_id": "grover_n3_k1_m010", "n": 3, "k": 1,
     "marked_states": ["010"], "qasm_path": "/abs/path/to/file.qasm"}

Outputs written to <output_dir>:
    chain_results.jsonl       — per-circuit ChainResult records
    per_circuit_metrics.jsonl — per-circuit SA, CF, oracle_correct
    aggregated_metrics.json   — per-n aggregates (means + stds)
    cot_agents.md             — human-readable agent traces
"""

import argparse
import asyncio
import importlib
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# Project-level imports for the existing metric implementations
_PROJECT_DIR = _HERE.parent
sys.path.insert(0, str(_PROJECT_DIR / "evaluation"))
sys.path.insert(0, str(_PROJECT_DIR / "GroverGPT-plus"))

# Phase 3 dir appended (NOT prepended) so shared infrastructure modules
# (chain, cot_logger, parsers, orchestrator_vanilla, orchestrator_langchain,
# vllm_server) are importable, while Phase 4's local `prompts.py` still wins
# for the orchestrator's `from prompts import ...` (because _HERE is ahead).
sys.path.append(str(_PROJECT_DIR / "Multi Agent System Phase 3"))

from metrics import (  # noqa: E402
    grover_ground_truth,
    search_accuracy,
    classical_fidelity,
)
from oracle_extraction_accuracy import (  # noqa: E402
    extract_oracle_body,
    normalize_oracle_block,
)

from chain import ChainResult  # noqa: E402
from cot_logger import CotLogger  # noqa: E402


# ─── Per-circuit metric computation ────────────────────────────────────────

def compute_circuit_metrics(
    n: int,
    true_marked_states: List[str],
    true_oracle_qasm: str,
    chain_result: ChainResult,
) -> Dict:
    """Compute SA, CF (raw + renorm), oracle accuracy for one circuit.

    Failure-counts-as-zero semantics (intentional):
      - If any agent failed to parse, SA = CF_raw = CF_renorm = 0.
        These metrics measure final-answer quality; the model got nothing
        meaningful out, so it scores zero (not "missing data"). This matches
        Phase 1 / Phase 2 conventions where a parse-failed monolithic CoT
        also scores 0.
      - Oracle accuracy is 1 if Agent 1 produced an oracle that matches
        ground truth (under QASM normalisation), even if downstream agents
        then failed. If Agent 1 didn't produce an oracle at all (A1 parse
        failure), or produced one that doesn't match, oracle_correct = 0.
        This isolates Agent 1's contribution from downstream failures.
    """
    metrics = {
        "sa": 0.0,
        "cf_raw": 0.0,
        "cf_renorm": 0.0,
        "oracle_correct": 0,
        "success": chain_result.success,
        "failure_stage": chain_result.failure_stage,
    }

    # Oracle extraction accuracy — derivable from Agent 1's output alone,
    # even if A2/A3 failed downstream.
    if chain_result.extracted_oracle is not None:
        true_body = extract_oracle_body(true_oracle_qasm)
        pred_body = extract_oracle_body(chain_result.extracted_oracle)
        if true_body is not None and pred_body is not None:
            if normalize_oracle_block(true_body) == normalize_oracle_block(pred_body):
                metrics["oracle_correct"] = 1
        # else (true_body or pred_body is None) → oracle_correct stays 0.

    # SA / CF — only meaningful if A3 produced a parseable dict. Otherwise
    # they stay 0 (the model failed to produce a usable distribution).
    if chain_result.probability_dict is not None:
        gt = grover_ground_truth(n, true_marked_states)
        metrics["sa"] = search_accuracy(chain_result.probability_dict, true_marked_states)
        metrics["cf_raw"] = classical_fidelity(chain_result.probability_dict, gt, mode="raw")
        metrics["cf_renorm"] = classical_fidelity(chain_result.probability_dict, gt, mode="renormalize")

    return metrics


def aggregate_per_n(per_circuit: List[Dict]) -> Dict[int, Dict]:
    """Group per-circuit metrics by n, compute means + stds."""
    by_n = defaultdict(list)
    for rec in per_circuit:
        by_n[rec["n"]].append(rec)

    out = {}
    for n in sorted(by_n):
        rows = by_n[n]
        n_circuits = len(rows)
        n_chain_success = sum(1 for r in rows if r["success"])
        n_a1_fail = sum(1 for r in rows if r["failure_stage"] == "A1")
        n_a2_fail = sum(1 for r in rows if r["failure_stage"] == "A2")
        n_a3_fail = sum(1 for r in rows if r["failure_stage"] == "A3")

        # All metrics are now 0-on-failure (not None), so aggregates always
        # use the full per-n sample. Failures correctly drag the means down.
        def _series(key):
            vals = [r[key] for r in rows]
            if not vals:
                return [None, None]
            if len(vals) < 2:
                return [vals[0], 0.0]
            return [statistics.mean(vals), statistics.stdev(vals)]

        oracle_vals = [r["oracle_correct"] for r in rows]
        oracle_acc = sum(oracle_vals) / len(oracle_vals)

        out[n] = {
            "n": n,
            "n_circuits": n_circuits,
            "n_chain_success": n_chain_success,
            "n_a1_failures": n_a1_fail,
            "n_a2_failures": n_a2_fail,
            "n_a3_failures": n_a3_fail,
            "sa": _series("sa"),
            "cf_raw": _series("cf_raw"),
            "cf_renorm": _series("cf_renorm"),
            "oracle_accuracy": oracle_acc,
        }
    return out


# ─── Main runner ───────────────────────────────────────────────────────────

async def _amain():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--circuits_jsonl", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument(
        "--orchestrator", choices=["vanilla", "langchain"], required=True,
        help="Which orchestrator implementation to use",
    )
    ap.add_argument("--max_model_len", type=int, default=122880)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument(
        "--max_concurrency", type=int, default=64,
        help="Max in-flight async tasks (circuits being chained at once)",
    )
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or Path(args.model_path).name

    # Load circuits + their QASM contents
    circuits = []
    with open(args.circuits_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                rec["qasm"] = Path(rec["qasm_path"]).read_text()
                circuits.append(rec)
    print(f"[{label}|{args.orchestrator}] Loaded {len(circuits)} circuits")

    # Build sampling params (orchestrator-independent)
    from vllm import SamplingParams
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        # Belt-and-suspenders stopping for LLaMA 3 family (vLLM issue #4180):
        # - stop_token_ids fires on the exact token IDs (more reliable than
        #   string matching, which can miss tokens that span boundaries).
        #   128001 = <|end_of_text|>, 128009 = <|eot_id|>.
        # - stop strings catch the same EOS markers if emitted as plain text
        #   AND the custom "=== END ===" terminator that the agent prompts
        #   teach the model to emit after its answer.
        stop=["<|eot_id|>", "<|end_of_text|>", "=== END ==="],
        stop_token_ids=[128001, 128009],
    )

    # Logger — both orchestrators write to the same cot_agents.md format
    logger = CotLogger(str(out_dir / "cot_agents.md"))

    # Dynamic import of the chosen orchestrator. Both expose `run_batch` with
    # the same signature: (base_url, served_model_name, circuits, sampling_params,
    # max_concurrency, logger).
    mod_name = f"orchestrator_{args.orchestrator}"
    orchestrator = importlib.import_module(mod_name)

    # Start vLLM HTTP server (AsyncLLMEngine natively async), run the
    # orchestrator against it, tear down server on exit.
    from vllm_server import VLLMServer
    served_model_name = Path(args.model_path).name
    async with VLLMServer(
        model_path=args.model_path,
        served_model_name=served_model_name,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    ) as server:
        print(f"[{label}|{args.orchestrator}] Starting agent chain on "
              f"{len(circuits)} circuits at concurrency={args.max_concurrency}",
              flush=True)
        t_start = time.perf_counter()
        results = await orchestrator.run_batch(
            base_url=server.base_url,
            served_model_name=served_model_name,
            circuits=[{"circuit_id": c["circuit_id"], "qasm": c["qasm"]} for c in circuits],
            sampling_params=sampling_params,
            max_concurrency=args.max_concurrency,
            logger=logger,
        )
        eval_time = time.perf_counter() - t_start

    # Persist chain results + compute metrics
    chain_results_path = out_dir / "chain_results.jsonl"
    per_circuit_metrics: List[Dict] = []
    with open(chain_results_path, "w") as f:
        for ckt, result in zip(circuits, results):
            f.write(json.dumps({
                "circuit_id": ckt["circuit_id"],
                "n": ckt["n"],
                "k": ckt["k"],
                "marked_states": ckt["marked_states"],
                "chain_result": result.to_dict(),
            }) + "\n")

            metrics = compute_circuit_metrics(
                n=ckt["n"],
                true_marked_states=ckt["marked_states"],
                true_oracle_qasm=ckt["qasm"],
                chain_result=result,
            )
            metrics.update({"circuit_id": ckt["circuit_id"],
                            "n": ckt["n"], "k": ckt["k"]})
            per_circuit_metrics.append(metrics)

    with open(out_dir / "per_circuit_metrics.jsonl", "w") as f:
        for rec in per_circuit_metrics:
            f.write(json.dumps(rec) + "\n")

    aggregated = aggregate_per_n(per_circuit_metrics)
    with open(out_dir / "aggregated_metrics.json", "w") as f:
        json.dump({
            "label": label,
            "orchestrator": args.orchestrator,
            "model_path": args.model_path,
            "n_circuits": len(circuits),
            "total_eval_time_s": eval_time,
            "per_n": {str(n): v for n, v in aggregated.items()},
        }, f, indent=2)

    # Compact per-n summary to stdout
    print(f"\n[{label}|{args.orchestrator}] === Per-n summary ===")
    print(f"{'n':>3} {'k':>1} {'circ':>5} {'succ':>4} {'sa':>6} {'cf_raw':>6} {'ora':>5}")
    for n, agg in aggregated.items():
        sa = "—" if agg["sa"][0] is None else f"{agg['sa'][0]:.3f}"
        cf = "—" if agg["cf_raw"][0] is None else f"{agg['cf_raw'][0]:.3f}"
        ora = "—" if agg["oracle_accuracy"] is None else f"{agg['oracle_accuracy']:.3f}"
        print(f"{n:>3} {' ':>1} {agg['n_circuits']:>5} {agg['n_chain_success']:>4} "
              f"{sa:>6} {cf:>6} {ora:>5}")

    print(f"\n[{label}|{args.orchestrator}] Done in {eval_time/60:.1f} min. "
          f"Results -> {out_dir}")


def main():
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
