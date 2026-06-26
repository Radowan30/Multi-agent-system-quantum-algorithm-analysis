"""
Shared evaluation pipeline for Phase 1 and Phase 2.

Both phases share the same:
  - Model output format: probability dict inside === Simulation Results ===
  - Metrics: SA, CF, CR, SRR (full mode only), Relative Execution Time
  - Parser: parse_output.py

Phase 3 uses a different output format (Agent 2 produces marked states followed
by a <MARKED_STATES> tag; probabilities come from a deterministic function call,
not the model) and requires a separate pipeline — eval_phase_3.py.

How to use:
  Import run_phase() into a phase-specific run_eval.py and pass the phase config.
  See evaluation/phase-1/run_eval.py for an example.
"""

import glob
import json
import os
import random
import statistics
import sys
import time
from typing import Dict, List, Optional, Tuple

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EVAL_DIR)

from metrics import (
    grover_ground_truth,
    search_accuracy,
    classical_fidelity,
    compression_ratio_per_circuit,
    sequence_reduction_ratio_per_circuit,
    aggregate_cr,
    aggregate_srr,
    aggregate_stats,
    relative_execution_time,
)
from parse_output import parse_model_output


def run_phase(
    manifest_path: str,
    model_path: str,
    base_tokenizer_path: str,
    quantum_tokenizer_path: str,
    results_dir: str,
    gpu_memory_utilization: float = 0.85,
    max_tokens: int = 8192,
    max_model_len: int = 8192,
    n_ret_circuits: int = 3,
    ret_seed: int = 42,
    ret_circuit_dir: Optional[str] = None,
) -> Dict[int, Dict]:
    """
    Run inference and compute all metrics for circuits in a manifest.

    Parameters
    ----------
    manifest_path          : JSON manifest from generate_eval_circuits.py
    model_path             : merged vLLM-compatible model directory
    base_tokenizer_path    : base LLaMA tokenizer path (for CR/SRR L_base)
    quantum_tokenizer_path : quantum-native tokenizer path (for CR/SRR L_quantum)
    results_dir            : directory to write results JSON (e.g. phase-1/results/)
    max_model_len          : context window — 8192 for Phase 1, 128000 for Phase 2
    n_ret_circuits         : circuits sampled per n for RET timing (single-instance)
    ret_seed               : RNG seed for reproducible RET circuit sampling
    ret_circuit_dir        : data_MMS dir (default: <project>/GroverGPT-plus/data_MMS)

    Returns
    -------
    results : {n: {sa, cf, cr, srr, ret, mean_time_s, n_circuits, n_parse_failures}}
    """
    manifest = _load_manifest(manifest_path)
    mode = manifest["mode"]

    print("[Eval] Loading tokenizers ...")
    base_tok     = AutoTokenizer.from_pretrained(base_tokenizer_path)
    quantum_tok  = (
        AutoTokenizer.from_pretrained(quantum_tokenizer_path)
        if quantum_tokenizer_path != base_tokenizer_path
        else base_tok
    )
    inference_tok = AutoTokenizer.from_pretrained(model_path)

    print("[Eval] Loading model into vLLM ...")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        stop=["<|eot_id|>", "<|end_of_text|>"],
    )
    print("[Eval] Ready.\n")

    sorted_by_n = sorted(manifest["by_n"].items(), key=lambda x: int(x[0]))

    # ── Pass 1 — RET timing (full-circuit mode only) ───────────────────────
    # Single-instance (one circuit at a time), n_ret_circuits sampled per n from
    # the natural data_MMS distribution (mixed k). Run in a dedicated pass — the
    # large SA/CF batch below would otherwise perturb the timing.
    #
    # RET is a full-circuit metric (n=2-9): the data_MMS circuits sampled here
    # are full circuits, and large-n full circuits exceed the context window.
    # Oracle mode therefore skips RET entirely.
    ret_timing: Dict[int, Tuple[float, float]] = {}
    if mode == "full":
        print("[Eval] Warmup ...")
        _warmup(llm, sampling_params, inference_tok, sorted_by_n)

        for n_str, _ in sorted_by_n:
            n = int(n_str)
            ret_circuits = _sample_ret_circuits(ret_circuit_dir, n, n_ret_circuits, ret_seed)
            if not ret_circuits:
                print(f"[n={n}] RET — no circuits found, skipping")
                ret_timing[n] = (0.0, 0.0)
                continue
            times = []
            for qasm, _fname in ret_circuits:
                prompt = inference_tok.apply_chat_template(
                    [{"role": "user", "content": qasm}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                t0 = time.perf_counter()
                llm.generate([prompt], sampling_params)
                times.append(time.perf_counter() - t0)
            mean_t = statistics.mean(times)
            std_t  = statistics.stdev(times) if len(times) > 1 else 0.0
            ret_timing[n] = (mean_t, std_t)
            print(f"[n={n}] RET — {len(times)} circuits single-instance: "
                  f"mean={mean_t:.2f}s std={std_t:.2f}s")
    else:
        print("[Eval] Oracle mode — skipping RET (RET is a full-circuit metric).")

    # ── Pass 2 — batch inference for SA/CF/CR/SRR ──────────────────────────
    # Raw model outputs are also saved to <results_dir>/outputs_*.jsonl so that
    # post-hoc metrics (e.g. oracle-extraction accuracy) can be computed offline
    # without re-running inference. One JSON record per circuit, schema:
    #   {n, k, source_file, marked_states, output_text}
    os.makedirs(results_dir, exist_ok=True)
    source = manifest.get("circuit_source", "strict")
    outputs_path = os.path.join(
        results_dir,
        f"outputs_{manifest['mode']}_{manifest['n_min']}_{manifest['n_max']}_{source}.jsonl",
    )
    outputs_f = open(outputs_path, "w")

    results: Dict[int, Dict] = {}
    for n_str, circuits in sorted_by_n:
        n = int(n_str)
        print(f"[n={n}] {len(circuits)} circuits — running inference ...")

        prompts = [
            inference_tok.apply_chat_template(
                [{"role": "user", "content": c["qasm"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for c in circuits
        ]
        outputs = llm.generate(prompts, sampling_params)

        sa_vals, cf_raw_vals, cf_renorm_vals, cr_vals, srr_vals = [], [], [], [], []
        n_failures = 0

        for circuit, output in zip(circuits, outputs):
            marked = circuit["marked_states"]
            output_text = output.outputs[0].text

            outputs_f.write(json.dumps({
                "n":             int(circuit["n"]),
                "k":             int(circuit["k"]),
                "source_file":   circuit["source_file"],
                "marked_states": marked,
                "output_text":   output_text,
            }) + "\n")

            # CR/SRR depend only on the QASM, not the model output — compute
            # them for every circuit, including ones that fail to parse.
            if mode == "full":
                qasm        = circuit["qasm"]
                base_len    = len(base_tok.encode(qasm))
                quantum_len = len(quantum_tok.encode(qasm))
                if base_len > 0 and quantum_len > 0:
                    cr_vals.append(compression_ratio_per_circuit(base_len, quantum_len))
                    srr_vals.append(sequence_reduction_ratio_per_circuit(base_len, quantum_len))

            pred_probs, _ = parse_model_output(output_text)

            if pred_probs is None:
                # No usable probability distribution (no dict, or unparsable
                # values). The model failed to deliver an output — score it as
                # a miss for both metrics rather than excluding the circuit,
                # which would inflate the reported averages.
                n_failures += 1
                sa_vals.append(0.0)
                cf_raw_vals.append(0.0)
                cf_renorm_vals.append(0.0)
                continue

            gt_probs = grover_ground_truth(n, marked)
            sa_vals.append(search_accuracy(pred_probs, marked))
            # Compute CF both ways every run — inference is the cost; the two
            # CF passes are free. "cf" = raw (paper-faithful), "cf_renorm" =
            # over-sum-renormalised. See metrics.classical_fidelity.
            cf_raw_vals.append(classical_fidelity(pred_probs, gt_probs, mode="raw"))
            cf_renorm_vals.append(
                classical_fidelity(pred_probs, gt_probs, mode="renormalize")
            )

        mean_time_s, std_time_s = ret_timing.get(n, (0.0, 0.0))
        per_n: Dict = {
            "n":                n,
            "mode":             mode,
            "n_circuits":       len(circuits),
            "n_parse_failures": n_failures,
            "mean_time_s":      mean_time_s,
            "time_std_s":       std_time_s,
            "sa":               list(aggregate_stats(sa_vals))         if sa_vals        else [None, None],
            "cf":               list(aggregate_stats(cf_raw_vals))     if cf_raw_vals    else [None, None],
            "cf_renorm":        list(aggregate_stats(cf_renorm_vals))  if cf_renorm_vals else [None, None],
        }
        if mode == "full":
            per_n["cr"]  = list(aggregate_cr(cr_vals))   if cr_vals  else [None, None]
            per_n["srr"] = list(aggregate_srr(srr_vals)) if srr_vals else [None, None]

        results[n] = per_n
        _print_n_summary(per_n, mode)

    # Relative Execution Time: S(n) = T(n) / T(2).
    # Error bar on S(n) is the std of the n's timing runs divided by the
    # (fixed) T(2) normaliser — i.e. mean ± std propagated through the ratio.
    times_by_n = {n: r["mean_time_s"] for n, r in results.items()}
    if 2 in times_by_n and times_by_n[2] > 0:
        ret = relative_execution_time(times_by_n)
        t2 = times_by_n[2]
        for n in results:
            results[n]["ret"]     = ret[n]
            results[n]["ret_std"] = results[n]["time_std_s"] / t2

    outputs_f.close()
    print(f"Outputs saved to: {outputs_path}")

    _save_results(results, manifest, results_dir)
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_manifest(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _sample_ret_circuits(
    ret_circuit_dir: Optional[str], n: int, n_circuits: int, seed: int
) -> List[Tuple[str, str]]:
    """
    Sample n_circuits QASM files for n from data_MMS/grover_n{n}/, drawn from
    the full circuit population (mixed k — single- and multi-target) with a
    fixed seed for reproducibility. Returns a list of (qasm_text, filename).

    RET times each sampled circuit once in single-instance mode; the spread
    across them is the reported per-n standard deviation. Sampling several
    circuits (rather than one fixed circuit) keeps the measurement from being
    hostage to a single outlier — e.g. the n=8 all-zeros circuit whose output
    degenerates to a one-state distribution.
    """
    if ret_circuit_dir is None:
        ret_circuit_dir = os.path.join(
            os.path.dirname(_EVAL_DIR), "GroverGPT-plus", "data_MMS"
        )
    pool = sorted(glob.glob(os.path.join(ret_circuit_dir, f"grover_n{n}", "*.qasm")))
    if not pool:
        return []
    chosen = random.Random(seed).sample(pool, min(n_circuits, len(pool)))
    out: List[Tuple[str, str]] = []
    for path in chosen:
        with open(path) as f:
            out.append((f.read(), os.path.basename(path)))
    return out


def _warmup(llm, sampling_params, inference_tok, sorted_by_n) -> None:
    """One discarded single-instance generate, so the first timed RET run
    measures steady-state inference rather than cold-start overhead."""
    for _n_str, circuits in sorted_by_n:
        if circuits:
            prompt = inference_tok.apply_chat_template(
                [{"role": "user", "content": circuits[0]["qasm"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            llm.generate([prompt], sampling_params)
            return


def _save_results(results: Dict[int, Dict], manifest: dict, results_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    source = manifest.get("circuit_source", "strict")
    fname = f"results_{manifest['mode']}_{manifest['n_min']}_{manifest['n_max']}_{source}.json"
    out_path = os.path.join(results_dir, fname)
    with open(out_path, "w") as f:
        json.dump({str(n): r for n, r in results.items()}, f, indent=2)
    print(f"\nResults saved to: {out_path}")


def _print_n_summary(r: dict, mode: str) -> None:
    n = r["n"]
    sa = r["sa"]
    cf = r["cf"]
    cf_r = r.get("cf_renorm")
    sa_str = f"{sa[0]:.3f}±{sa[1]:.3f}" if sa[0] is not None else "N/A"
    cf_str = f"{cf[0]:.3f}±{cf[1]:.3f}" if cf[0] is not None else "N/A"
    cfr_str = f"{cf_r[0]:.3f}±{cf_r[1]:.3f}" if cf_r and cf_r[0] is not None else "N/A"
    msg = (
        f"[n={n}] circuits={r['n_circuits']} failures={r['n_parse_failures']} "
        f"SA={sa_str} CF(raw)={cf_str} CF(renorm)={cfr_str}"
    )
    if mode == "full" and r.get("cr") and r["cr"][0] is not None:
        msg += f" CR={r['cr'][0]:.2f} SRR={r['srr'][0]:.3f}"
    msg += f" time={r['mean_time_s']:.2f}±{r.get('time_std_s', 0.0):.2f}s"
    print(msg)
