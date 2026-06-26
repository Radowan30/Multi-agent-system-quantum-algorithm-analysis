"""
Phase 4 paper-eval post-processor.

Reads:
  - <bulk_dir>/chain_results.jsonl       (per-circuit ChainResult — bulk pass at concurrency=8)
  - <bulk_dir>/per_circuit_metrics.jsonl (per-circuit SA/CF/oracle — bulk pass)
  - <ret_dir>/chain_results.jsonl        (per-circuit ChainResult — RET pass at concurrency=1)
  - The paper circuit manifest           (for ground-truth oracle bodies)

Writes (all into <out_dir>):

  results_full_2_19_paper.json
      Phase 1/2-compatible per-n aggregate with SA, CF (raw + renorm), RET.
      Schema matches `eval_phase_1_2._save_results` so `plot_results.py` works
      unchanged.

  oracle_extraction_full_2_19_paper.json
      Phase 1/2-compatible Agent 1 accuracy (= oracle extraction accuracy in
      Phase 4). UNCONDITIONAL — reported for every n.

  marked_state_accuracy_full_2_19_paper.json
      Agent 2 accuracy with the STRICT CONDITIONAL rule: per-n point reported
      ONLY if A1 was correct on every circuit at that n. n's with any A1
      failure are omitted from `per_n`. Same schema as oracle_extraction_*.json
      so `plot_oracle_accuracy.py` works unchanged.

  agent3_accuracy_full_2_19_paper.json
      Agent 3 accuracy (= mean SA) with the STRICT CONDITIONAL rule: per-n
      point reported ONLY if A2 was correct on every circuit at that n.

  outputs_full_2_19_paper.jsonl
      Per-circuit raw model output in Phase 1/2 schema. `output_text` is built
      by wrapping Agent 1's `extracted_oracle` in the legacy header
      ("The Oracle entity is extracted below:") so any legacy text-matching
      tools that scan this file still work.

  cot_traces.md
      Per-circuit chain trace organised by (n, k) cell — H2 per (n, k),
      every circuit listed under its cell. TOC at top.

Per-agent rules (set in stone, see `project_phase4_per_agent_accuracy`
memory and Research Plan §6.5):
  - A1 score: binary, oracle_correct from per_circuit_metrics
  - A2 score: binary, set(A2_marked_states) == set(ground_truth_marked_states)
  - A3 score: continuous, the circuit's SA value
  - A1 aggregation: unconditional, every n
  - A2 aggregation: only if every circuit at n has A1=1
  - A3 aggregation: only if every circuit at n has A2=1
"""

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
sys.path.insert(0, str(_PROJECT / "evaluation"))
sys.path.insert(0, str(_PROJECT / "GroverGPT-plus"))

from oracle_extraction_accuracy import extract_oracle_body, normalize_oracle_block  # noqa: E402

# Legacy header that Phase 1/2's parsers expect to locate the oracle in
# the raw text output. We wrap Phase 4 A1 outputs in this so tooling that
# scans outputs_*.jsonl for the legacy header still works.
_LEGACY_ORACLE_HEADER = "The Oracle entity is extracted below:"


# ─── Score computations ───────────────────────────────────────────────────────

def _a1_correct(chain_result: dict, true_oracle_qasm: str) -> int:
    """Binary A1 score: extracted oracle matches ground truth (QASM-normalised)."""
    a1_oracle = chain_result.get("extracted_oracle")
    if a1_oracle is None:
        return 0
    true_body = extract_oracle_body(true_oracle_qasm)
    if true_body is None:
        return 0
    pred_body = extract_oracle_body(a1_oracle)
    if pred_body is None:
        return 0
    return 1 if normalize_oracle_block(true_body) == normalize_oracle_block(pred_body) else 0


def _a2_correct(chain_result: dict, true_marked: List[str]) -> int:
    """Binary A2 score: marked-state set equality with ground truth."""
    a2_marks = chain_result.get("marked_states") or []
    return 1 if set(a2_marks) == set(true_marked) else 0


# ─── Aggregation ──────────────────────────────────────────────────────────────

def _aggregate_chain(rows: List[dict]) -> Dict[int, dict]:
    """Per-n SA / CF (raw + renorm) / parse-failure aggregation, matching
    Phase 1/2's results_*.json schema. Bulk-pass timing values are ignored
    here; RET is merged in separately."""
    by_n: Dict[int, List[dict]] = defaultdict(list)
    for r in rows:
        by_n[r["n"]].append(r)
    out: Dict[int, dict] = {}
    for n in sorted(by_n):
        circs = by_n[n]
        sas = [c["metrics"]["sa"] for c in circs]
        cfs_raw = [c["metrics"]["cf_raw"] for c in circs]
        cfs_renorm = [c["metrics"]["cf_renorm"] for c in circs]
        n_parse_failures = sum(
            1 for c in circs if c["chain_result"]["failure_stage"] is not None
        )
        out[n] = {
            "n": n,
            "mode": "full",
            "n_circuits": len(circs),
            "n_parse_failures": n_parse_failures,
            "sa":        [statistics.mean(sas),       statistics.stdev(sas)       if len(sas)       > 1 else 0.0],
            "cf":        [statistics.mean(cfs_raw),   statistics.stdev(cfs_raw)   if len(cfs_raw)   > 1 else 0.0],
            "cf_renorm": [statistics.mean(cfs_renorm), statistics.stdev(cfs_renorm) if len(cfs_renorm) > 1 else 0.0],
        }
    return out


def _aggregate_oracle_accuracy(rows: List[dict]) -> dict:
    """Unconditional A1 (oracle) accuracy per n, in Phase 1/2 schema."""
    by_n: Dict[int, List[int]] = defaultdict(list)
    for r in rows:
        by_n[r["n"]].append(r["a1_score"])
    per_n = {}
    for n in sorted(by_n):
        scores = by_n[n]
        per_n[str(n)] = {
            "accuracy": sum(scores) / len(scores),
            "correct":  sum(scores),
            "total":    len(scores),
        }
    return per_n


def _aggregate_strict_conditional(
    rows: List[dict],
    gate_field: str,
    score_field: str,
    is_continuous: bool,
) -> dict:
    """Strict-conditional per-n aggregation.

    `gate_field`  — per-row field that must be 1 on EVERY circuit at n=N
                    for the n=N point to be reported (e.g. 'a1_score' to
                    gate A2, 'a2_score' to gate A3).
    `score_field` — per-row field to aggregate (e.g. 'a2_score', 'sa').
    `is_continuous` — False for binary scores (writes 'correct'/'total');
                      True for continuous (writes 'mean' instead).

    n's where any circuit has gate=0 are OMITTED entirely from per_n.
    """
    by_n: Dict[int, List[dict]] = defaultdict(list)
    for r in rows:
        by_n[r["n"]].append(r)
    per_n = {}
    for n in sorted(by_n):
        circs = by_n[n]
        gate_ok = all(c[gate_field] == 1 for c in circs)
        if not gate_ok:
            continue
        scores = [c[score_field] for c in circs]
        if is_continuous:
            per_n[str(n)] = {
                "accuracy": statistics.mean(scores),
                "stdev":    statistics.stdev(scores) if len(scores) > 1 else 0.0,
                "n_circuits": len(circs),
            }
        else:
            per_n[str(n)] = {
                "accuracy": sum(scores) / len(scores),
                "correct":  sum(scores),
                "total":    len(scores),
            }
    return per_n


# ─── RET merging ──────────────────────────────────────────────────────────────

def _ret_timings_from_chain_results(ret_rows: List[dict]) -> Dict[int, Tuple[float, float]]:
    """Group RET-pass chain_results.jsonl by n; return (mean_time, std_time) per n."""
    by_n: Dict[int, List[float]] = defaultdict(list)
    for r in ret_rows:
        by_n[r["n"]].append(r["chain_result"]["total_time_s"])
    out: Dict[int, Tuple[float, float]] = {}
    for n in sorted(by_n):
        times = by_n[n]
        mean = statistics.mean(times)
        stdv = statistics.stdev(times) if len(times) > 1 else 0.0
        out[n] = (mean, stdv)
    return out


def _merge_ret_into_results(results: Dict[int, dict],
                            ret_timings: Dict[int, Tuple[float, float]]) -> None:
    """Add mean_time_s/time_std_s/ret/ret_std to each n entry. RET formula:
    S(n) = T(n)/T(2); ret_std = time_std_s(n) / T(2)."""
    t2 = ret_timings.get(2, (None, None))[0]
    if not t2 or t2 <= 0:
        print("[postprocess] WARN: no RET timing for n=2 — RET fields omitted")
        # Still attach raw timings (so they appear in the JSON even without RET ratios)
        for n, (mean_t, std_t) in ret_timings.items():
            if n in results:
                results[n]["mean_time_s"] = mean_t
                results[n]["time_std_s"]  = std_t
        return
    for n, (mean_t, std_t) in ret_timings.items():
        if n not in results:
            continue
        results[n]["mean_time_s"] = mean_t
        results[n]["time_std_s"]  = std_t
        results[n]["ret"]         = mean_t / t2
        results[n]["ret_std"]     = std_t / t2


# ─── outputs_*.jsonl writer (Phase 1/2 schema) ────────────────────────────────

def _write_legacy_outputs(rows: List[dict], path: Path) -> None:
    """Write per-circuit raw outputs in Phase 1/2 schema, with A1's extracted
    oracle wrapped under the legacy `The Oracle entity is extracted below:`
    header so any tool that text-matches on that header still works."""
    with open(path, "w") as f:
        for r in rows:
            cr = r["chain_result"]
            a1_text = cr.get("a1_output") or ""
            oracle = cr.get("extracted_oracle")
            # Synthesise a legacy-shaped output_text: prepend the legacy
            # header + oracle body so legacy text scanners see what they expect.
            if oracle is not None:
                legacy_block = f"\n\n{_LEGACY_ORACLE_HEADER}\n{oracle}\n"
                output_text = a1_text + legacy_block
            else:
                output_text = a1_text
            rec = {
                "n":             r["n"],
                "k":             r["k"],
                "source_file":   r["source_file"],
                "marked_states": r["marked_states"],
                "output_text":   output_text,
            }
            f.write(json.dumps(rec) + "\n")


# ─── cot_traces.md writer (by (n,k) cell) ─────────────────────────────────────

def _write_cot_traces(rows: List[dict], path: Path) -> None:
    """Write a per-(n,k) organised CoT trace markdown. TOC at top, then one
    H2 section per cell containing every circuit's chain in that cell."""
    by_cell: Dict[Tuple[int, int], List[dict]] = defaultdict(list)
    for r in rows:
        by_cell[(r["n"], r["k"])].append(r)
    cells = sorted(by_cell)

    lines: List[str] = []
    lines.append("# Phase 4 — Per-Circuit Agent Chain Traces (paper eval)\n")
    lines.append("Per-circuit chains organised by (n, k) cell. Use the TOC to jump.\n\n")
    lines.append("## Table of Contents\n")
    for n, k in cells:
        anchor = f"n-{n}-k-{k}-{len(by_cell[(n, k)])}-circuits"
        lines.append(f"- [n = {n}, k = {k} — {len(by_cell[(n, k)])} circuits](#{anchor})\n")
    lines.append("\n---\n\n")

    for n, k in cells:
        cell_rows = by_cell[(n, k)]
        lines.append(f"## n = {n}, k = {k} — {len(cell_rows)} circuits {{#n-{n}-k-{k}-{len(cell_rows)}-circuits}}\n\n")
        for r in cell_rows:
            cr = r["chain_result"]
            cid = r["circuit_id"]
            lines.append(f"### `{cid}`\n\n")
            lines.append(
                f"**success:** {cr['success']}    "
                f"**failure_stage:** {cr['failure_stage']}    "
                f"**total_time_s:** {cr['total_time_s']:.3f}    "
                f"**marked_states (truth):** `{r['marked_states']}`\n\n"
            )

            # Agent 1
            lines.append("**Agent 1 (Oracle Extractor)** — ")
            lines.append(f"_time: {cr.get('a1_time_s', 0.0):.3f}s_\n\n")
            lines.append("```\n")
            lines.append((cr.get("a1_output") or "(no output)").rstrip() + "\n")
            lines.append("```\n\n")
            if cr.get("extracted_oracle"):
                lines.append("Parsed oracle:\n")
                lines.append("```\n" + cr["extracted_oracle"].rstrip() + "\n```\n\n")

            # Agent 2 (only if it ran)
            if cr.get("a2_output") is not None:
                lines.append("**Agent 2 (Marked-State Identifier)** — ")
                lines.append(f"_time: {cr.get('a2_time_s', 0.0):.3f}s_\n\n")
                lines.append("```\n")
                lines.append(cr["a2_output"].rstrip() + "\n")
                lines.append("```\n\n")
                if cr.get("marked_states"):
                    lines.append("Parsed marked states:\n")
                    lines.append("```\n" + "\n".join(cr["marked_states"]) + "\n```\n\n")

            # Agent 3 (only if it ran)
            if cr.get("a3_output") is not None:
                lines.append("**Agent 3 (Probability Distribution)** — ")
                lines.append(f"_time: {cr.get('a3_time_s', 0.0):.3f}s_\n\n")
                lines.append("```\n")
                lines.append(cr["a3_output"].rstrip() + "\n")
                lines.append("```\n\n")
                if cr.get("probability_dict"):
                    lines.append("Parsed probability dict:\n")
                    lines.append("```\n")
                    for state, prob in cr["probability_dict"].items():
                        lines.append(f"  '{state}': {prob}\n")
                    lines.append("```\n\n")

            lines.append("---\n\n")

    path.write_text("".join(lines))


# ─── Main driver ──────────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _resolve_qasm_path(cid: str, qasm_dirs: List[Path]) -> Optional[Path]:
    """Find <cid>.qasm in the first qasm_dir that contains it."""
    for d in qasm_dirs:
        p = d / f"{cid}.qasm"
        if p.exists():
            return p
    return None


def _join_chain_and_metrics(chain_rows: List[dict],
                            metric_rows: List[dict],
                            qasm_dirs: List[Path]) -> List[dict]:
    """Merge chain_results + per_circuit_metrics by circuit_id; attach
    per-circuit A1/A2 binary scores and source_file. `qasm_dirs` is searched
    in order — used for layouts where qasm/ is at the bulk_dir's parent
    (e.g. Phase 3 preliminary, where 4 models share one qasm/ dir)."""
    metrics_by_id = {m["circuit_id"]: m for m in metric_rows}
    merged: List[dict] = []
    n_missing_qasm = 0
    for chain in chain_rows:
        cid = chain["circuit_id"]
        m = metrics_by_id.get(cid)
        if m is None:
            continue
        qasm_path = _resolve_qasm_path(cid, qasm_dirs)
        if qasm_path is None:
            n_missing_qasm += 1
            true_qasm = ""
        else:
            true_qasm = qasm_path.read_text()
        true_marked = chain["marked_states"]
        a1 = _a1_correct(chain["chain_result"], true_qasm)
        a2 = _a2_correct(chain["chain_result"], true_marked)
        merged.append({
            "circuit_id":    cid,
            "n":             chain["n"],
            "k":             chain["k"],
            "marked_states": true_marked,
            "source_file":   f"data_MMS/grover_n{chain['n']}/{cid}.qasm",
            "qasm_path":     str(qasm_path) if qasm_path else "",
            "chain_result":  chain["chain_result"],
            "metrics":       m,
            "a1_score":      a1,
            "a2_score":      a2,
            "sa":            m["sa"],
        })
    if n_missing_qasm:
        print(f"[postprocess] WARN: {n_missing_qasm} circuits had no QASM file "
              f"found in any of {[str(d) for d in qasm_dirs]} — A1 score "
              f"forced to 0 for those.")
    return merged


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bulk_dir", required=True,
                    help="Phase 4 run_eval output dir for the bulk pass")
    ap.add_argument("--ret_dir", default=None,
                    help="Phase 4 run_eval output dir for the RET (concurrency=1) pass. "
                         "Omit (or pass --skip_ret) to skip RET merging — useful when "
                         "re-aggregating stratified/preliminary data where the per-circuit "
                         "timings were collected under concurrent batching and aren't "
                         "comparable to Phase 1/2 single-instance RET methodology.")
    ap.add_argument("--skip_ret", action="store_true",
                    help="Skip RET merging entirely. The results JSON will omit "
                         "ret/ret_std/mean_time_s/time_std_s fields, and plot_results.py "
                         "will auto-skip the RET plot.")
    ap.add_argument("--out_dir", required=True,
                    help="Where to write the Phase 1/2-compatible files + cot_traces.md")
    ap.add_argument("--n_min", type=int, default=2)
    ap.add_argument("--n_max", type=int, default=19)
    args = ap.parse_args()

    bulk_dir = Path(args.bulk_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bulk_chain = _load_jsonl(bulk_dir / "chain_results.jsonl")
    bulk_metrics = _load_jsonl(bulk_dir / "per_circuit_metrics.jsonl")

    skip_ret = args.skip_ret or args.ret_dir is None
    if skip_ret:
        ret_chain = []
        print(f"Loaded {len(bulk_chain)} bulk chains, "
              f"{len(bulk_metrics)} bulk metrics, RET merging SKIPPED.")
    else:
        ret_dir = Path(args.ret_dir)
        ret_chain = _load_jsonl(ret_dir / "chain_results.jsonl")
        print(f"Loaded {len(bulk_chain)} bulk chains, "
              f"{len(bulk_metrics)} bulk metrics, "
              f"{len(ret_chain)} RET chains.")

    # Search bulk_dir/qasm first (Phase 4 paper-eval layout), then
    # bulk_dir.parent/qasm (Phase 3 preliminary layout, where the 4 model dirs
    # share one qasm/ at the parent).
    qasm_dirs = [bulk_dir / "qasm", bulk_dir.parent / "qasm"]
    rows = _join_chain_and_metrics(bulk_chain, bulk_metrics, qasm_dirs)
    print(f"Joined → {len(rows)} per-circuit rows for aggregation.")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    results = _aggregate_chain(rows)
    if not skip_ret:
        ret_timings = _ret_timings_from_chain_results(ret_chain)
        _merge_ret_into_results(results, ret_timings)

    oracle_acc = _aggregate_oracle_accuracy(rows)
    marked_acc = _aggregate_strict_conditional(
        rows, gate_field="a1_score", score_field="a2_score", is_continuous=False,
    )
    a3_acc = _aggregate_strict_conditional(
        rows, gate_field="a2_score", score_field="sa", is_continuous=True,
    )

    n_min, n_max = args.n_min, args.n_max
    src = "paper"

    # ── Write Phase 1/2-compatible files ──────────────────────────────────────
    # results_full_2_19_paper.json
    results_path = out_dir / f"results_full_{n_min}_{n_max}_{src}.json"
    with open(results_path, "w") as f:
        json.dump({str(n): r for n, r in results.items()}, f, indent=2)
    print(f"Wrote: {results_path}")

    # oracle_extraction_full_2_19_paper.json (A1 unconditional)
    oracle_path = out_dir / f"oracle_extraction_full_{n_min}_{n_max}_{src}.json"
    with open(oracle_path, "w") as f:
        json.dump({
            "mode": "full",
            "circuit_source": src,
            "n_min": n_min,
            "n_max": n_max,
            "rule": "unconditional — A1 score averaged across all circuits at each n",
            "per_n": oracle_acc,
        }, f, indent=2)
    print(f"Wrote: {oracle_path}")

    # marked_state_accuracy_full_2_19_paper.json (A2 strict-conditional on A1)
    marked_path = out_dir / f"marked_state_accuracy_full_{n_min}_{n_max}_{src}.json"
    with open(marked_path, "w") as f:
        json.dump({
            "mode": "full",
            "circuit_source": src,
            "n_min": n_min,
            "n_max": n_max,
            "rule": "strict conditional — point reported only if A1 correct on every circuit at n",
            "per_n": marked_acc,
        }, f, indent=2)
    print(f"Wrote: {marked_path}")

    # agent3_accuracy_full_2_19_paper.json (A3 strict-conditional on A2)
    a3_path = out_dir / f"agent3_accuracy_full_{n_min}_{n_max}_{src}.json"
    with open(a3_path, "w") as f:
        json.dump({
            "mode": "full",
            "circuit_source": src,
            "n_min": n_min,
            "n_max": n_max,
            "rule": "strict conditional — point reported only if A2 correct on every circuit at n; "
                    "metric is mean SA",
            "per_n": a3_acc,
        }, f, indent=2)
    print(f"Wrote: {a3_path}")

    # outputs_*.jsonl (legacy-compatible)
    outputs_path = out_dir / f"outputs_full_{n_min}_{n_max}_{src}.jsonl"
    _write_legacy_outputs(rows, outputs_path)
    print(f"Wrote: {outputs_path}")

    # cot_traces.md (organised by (n, k))
    cot_path = out_dir / "cot_traces.md"
    _write_cot_traces(rows, cot_path)
    print(f"Wrote: {cot_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== Per-n summary ===")
    print(f"{'n':>3} {'circ':>5} {'fail':>4} {'SA':>6} {'CF':>6} {'CFr':>6} "
          f"{'A1':>5} {'A2*':>5} {'A3*':>5} {'RET':>6}")
    for n in sorted(results):
        r = results[n]
        sa = r["sa"][0]
        cf = r["cf"][0]
        cfr = r["cf_renorm"][0]
        ora = oracle_acc.get(str(n), {}).get("accuracy")
        a2 = marked_acc.get(str(n), {}).get("accuracy")
        a3 = a3_acc.get(str(n), {}).get("accuracy")
        ret_v = r.get("ret")
        def fmt(x, w=6, prec=3):
            return f"{x:>{w}.{prec}f}" if x is not None else f"{'—':>{w}}"
        print(f"{n:>3} {r['n_circuits']:>5} {r['n_parse_failures']:>4} "
              f"{fmt(sa)} {fmt(cf)} {fmt(cfr)} {fmt(ora,5)} {fmt(a2,5)} {fmt(a3,5)} {fmt(ret_v)}")


if __name__ == "__main__":
    main()
