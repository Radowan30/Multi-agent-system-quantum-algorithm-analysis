"""
Aggregate and plot the Phase 3 results for the four LLM variations.

For each variation folder under --results_root (written by run_eval.py), this
script:
  1. Re-aggregates chain_results.jsonl + per_circuit_metrics.jsonl into the
     Phase 1/2-style JSONs with Phase 4's paper_eval_postprocess.py
     (SA, CF raw + renorm, Agent 1 unconditional, Agent 2 strict-conditional
     on Agent 1, Agent 3 strict-conditional on Agent 2).
  2. Writes 5 plots per variation:
       - sa_cf_full_paper.png        (raw CF)
       - sa_cf_full_paper_renorm.png (renormalised CF)
       - oracle_accuracy.png         (Agent 1)
       - marked_state_accuracy.png   (Agent 2, strict conditional)
       - agent3_accuracy.png         (Agent 3, strict conditional)

No RET plot: Phase 3 runs the chains at concurrency 32, so per-circuit times
are not single-instance timings.

Usage (venv-eval):
    python make_phase3_plots.py [--results_root $WORK/results/phase-3]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPRO = Path(__file__).resolve().parent.parent          # Reproduce Results/
PY_EVAL = Path(sys.executable)                          # run this script with venv-eval
POSTPROCESS = REPRO / "Phase 4" / "paper_eval_postprocess.py"
EVAL_PIPELINE = REPRO / "evaluation_pipeline"
PLOT_ORACLE = EVAL_PIPELINE / "plot_oracle_accuracy.py"

MODELS = [
    ("untrained_llama31",                "Untrained LLaMA 3.1 8B"),
    ("finetuned_llama31_alpha32",        "Fine-tuned LLaMA 3.1 8B (Phase 2 alpha32)"),
    ("untrained_gradient262k",           "Untrained Gradient-262k"),
    ("finetuned_gradient262k_alpha32",   "Fine-tuned Gradient-262k (Phase 2 alpha32)"),
]

# The Phase 3 set has one circuit per (n, k) cell over n = 2..19
N_MIN, N_MAX = 2, 19


def run(cmd: list, cwd: Path = None):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None, check=True)


def make_plots_for_model(model_dir: Path, label: str):
    print(f"\n{'=' * 72}\n  {label}\n  {model_dir}\n{'=' * 72}")

    # ── 1. Postprocess ─────────────────────────────────────────────────────
    print("\n[1/6] Aggregating per-agent + chain metrics...")
    run([
        PY_EVAL, POSTPROCESS,
        "--bulk_dir", model_dir,
        "--skip_ret",
        "--out_dir", model_dir,
        "--n_min", N_MIN, "--n_max", N_MAX,
    ])

    # ── 2. SA + CF (raw) ───────────────────────────────────────────────────
    # plot_results.py expects --phase=2 (n range 2..19, full mode). It pulls
    # results from <results_dir>/results_full_2_19_paper.json. With no RET
    # data in the JSON (skip_ret), plot_ret auto-skips. CR/SRR fields are
    # also absent → plot_cr_srr would produce empty plots, so we bypass main()
    # and call plot_sa_cf directly via a tiny inline shim.
    print("\n[2/6] Generating SA + CF (raw) plot...")
    _run_plot_sa_cf(model_dir, cf_key="cf")

    # ── 3. SA + CF (renorm) ─────────────────────────────────────────────────
    print("\n[3/6] Generating SA + CF (renorm) plot...")
    _run_plot_sa_cf(model_dir, cf_key="cf_renorm")

    # ── 4. Oracle Accuracy (A1) ─────────────────────────────────────────────
    print("\n[4/6] Generating Oracle Accuracy (A1) plot...")
    run([
        PY_EVAL, PLOT_ORACLE,
        "--input", model_dir / f"oracle_extraction_full_{N_MIN}_{N_MAX}_paper.json",
                   f"Agent 1 — Oracle Extraction",
        "--output", model_dir / "oracle_accuracy.png",
        "--title", f"{label} — Agent 1 (Oracle Extraction) Accuracy",
    ])

    # ── 5. Marked-State Accuracy (A2, strict-conditional on A1) ─────────────
    print("\n[5/6] Generating Marked-State Accuracy (A2) plot...")
    run([
        PY_EVAL, PLOT_ORACLE,
        "--input", model_dir / f"marked_state_accuracy_full_{N_MIN}_{N_MAX}_paper.json",
                   f"Agent 2 — Marked-State Identification",
        "--output", model_dir / "marked_state_accuracy.png",
        "--title", f"{label} — Agent 2 Marked-State Identification "
                   f"(strict conditional on A1)",
    ])

    # ── 6. Agent 3 Accuracy (strict-conditional on A2) ─────────────────────
    print("\n[6/6] Generating Agent 3 Accuracy plot...")
    run([
        PY_EVAL, PLOT_ORACLE,
        "--input", model_dir / f"agent3_accuracy_full_{N_MIN}_{N_MAX}_paper.json",
                   f"Agent 3 — Probability Distribution (mean SA)",
        "--output", model_dir / "agent3_accuracy.png",
        "--title", f"{label} — Agent 3 Probability Distribution Accuracy "
                   f"(strict conditional on A2)",
    ])


def _run_plot_sa_cf(model_dir: Path, cf_key: str):
    """Call plot_results.plot_sa_cf directly (bypassing its main() which also
    tries CR/SRR + RET that we don't have). cf_key: 'cf' or 'cf_renorm'."""
    sys.path.insert(0, str(EVAL_PIPELINE))
    import importlib
    pr = importlib.import_module("plot_results")
    results = pr.load_results(
        phase=2, mode="full", n_min=N_MIN, n_max=N_MAX,
        circuit_source="paper", results_dir=str(model_dir),
    )
    pr.plot_sa_cf(results, mode="full", phase=2,
                  output_dir=str(model_dir), circuit_source="paper",
                  cf_key=cf_key)
    sys.path.pop(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_root", default=None,
                    help="Folder holding the four variation folders (default: $WORK/results/phase-3)")
    args = ap.parse_args()
    if args.results_root is None:
        if not os.environ.get("WORK"):
            ap.error("pass --results_root, or set WORK")
        args.results_root = os.path.join(os.environ["WORK"], "results", "phase-3")
    phase3_dir = Path(args.results_root)

    for slug, label in MODELS:
        model_dir = phase3_dir / slug
        if not (model_dir / "chain_results.jsonl").exists():
            print(f"[SKIP] {slug}: no chain_results.jsonl")
            continue
        make_plots_for_model(model_dir, label)

    print(f"\n{'=' * 72}\n  All Phase 3 plots done.\n{'=' * 72}")


if __name__ == "__main__":
    main()
