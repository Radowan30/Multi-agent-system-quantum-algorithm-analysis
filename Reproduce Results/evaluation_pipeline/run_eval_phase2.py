"""
Phase 2 evaluation — Fine-tuned LLaMA 3.1 8B Instruct (128K context, same
quantum-native 139-token extended tokeniser as Phase 1).

Training data is unchanged from Phase 1: Grover_FullCircuit_2_7_MMS +
Grover_Oracle_2_10_MMS. Only the evaluation range is extended — LLaMA 3.1's
128K context window now allows full-circuit inputs up to n=19 (vs n=9 in
Phase 1), while oracle-only stays at n=2..20 for direct comparability.

Two-step workflow (mirrors Phase 1):

  Step 1 — Generate circuits (run once, only needs Qiskit):
    python evaluation/generate_eval_circuits.py --mode full   --n_min 2 --n_max 19 --circuit_source paper
    python evaluation/generate_eval_circuits.py --mode oracle --n_min 2 --n_max 20 --circuit_source paper

  Step 2 — Run evaluation:
    python evaluation/phase-2/run_eval.py --mode full   --circuit_source paper --model_path <merged>
    python evaluation/phase-2/run_eval.py --mode oracle --circuit_source paper --model_path <merged>

Results saved to: evaluation/phase-2/results/ (override with --results_dir).
"""

import argparse
import os
import sys

_PHASE2_DIR  = os.path.dirname(os.path.abspath(__file__))
_EVAL_DIR    = os.path.dirname(_PHASE2_DIR)
_PROJECT_DIR = os.path.dirname(_EVAL_DIR)
sys.path.insert(0, _EVAL_DIR)

from eval_phase_1_2 import run_phase

# ─── Phase 2 model configuration ──────────────────────────────────────────────
MODEL_PATH             = os.path.join(_PROJECT_DIR, "saves/Llama-3.1-8B-Instruct/merged/Llama31_alpha16_cl4000")
QUANTUM_TOKENIZER_PATH = MODEL_PATH

# Base (unextended) tokeniser used only for the CR/SRR baseline.
BASE_TOKENIZER_PATH    = os.path.join(
    _PROJECT_DIR, "models/Llama-3.1-8B-Instruct/original_tokenizer_backup"
)

RESULTS_DIR    = os.path.join(_PHASE2_DIR, "results")
MANIFESTS_DIR  = os.path.join(_EVAL_DIR, "circuit_manifests")
MAX_MODEL_LEN  = 120000        # margin under LLaMA 3.1's 131 072-token window

# Phase 2 evaluation ranges
N_RANGES = {
    "full":   (2, 19),
    "oracle": (2, 20),
}


def main():
    parser = argparse.ArgumentParser(description="Phase 2 LLaMA 3.1 8B evaluation")
    parser.add_argument(
        "--mode", choices=["full", "oracle"], required=True,
        help="'full' = complete Grover circuit (n=2-19); 'oracle' = oracle-only (n=2-20)",
    )
    parser.add_argument(
        "--circuit_source", choices=["strict", "paper"], default="paper",
        help="'paper' (default): data_MMS circuits for all n",
    )
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Override manifest path (default: auto-derived from mode and circuit_source)",
    )
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Override the merged model directory.",
    )
    parser.add_argument(
        "--results_dir", type=str, default=None,
        help="Override the results output directory.",
    )
    parser.add_argument(
        "--max_model_len", type=int, default=MAX_MODEL_LEN,
        help="Override max_model_len (default 120000 — LLaMA 3.1 supports up to 131072).",
    )
    args = parser.parse_args()

    model_path  = args.model_path or MODEL_PATH
    results_dir = args.results_dir or RESULTS_DIR
    n_min, n_max = N_RANGES[args.mode]
    manifest_path = args.manifest or os.path.join(
        MANIFESTS_DIR, f"circuits_{args.mode}_{n_min}_{n_max}_{args.circuit_source}.json"
    )

    if not os.path.exists(manifest_path):
        print(f"[Error] Manifest not found: {manifest_path}")
        print(f"  Run first:")
        print(f"    python evaluation/generate_eval_circuits.py "
              f"--mode {args.mode} --n_min {n_min} --n_max {n_max} "
              f"--circuit_source {args.circuit_source}")
        sys.exit(1)

    run_phase(
        manifest_path=manifest_path,
        model_path=model_path,
        base_tokenizer_path=BASE_TOKENIZER_PATH,
        quantum_tokenizer_path=model_path,
        results_dir=results_dir,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_tokens=8192,
        max_model_len=args.max_model_len,
    )


if __name__ == "__main__":
    main()
