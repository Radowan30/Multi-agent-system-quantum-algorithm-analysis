"""
Phase 1 evaluation — GroverGPT+ (LLaMA 3 8B Instruct, quantum-native tokeniser).

Two-step workflow:

  Step 1 — Generate circuits (run once, only needs Qiskit):
    python evaluation/generate_eval_circuits.py --mode full   --n_min 2 --n_max 9  --circuit_source strict
    python evaluation/generate_eval_circuits.py --mode full   --n_min 2 --n_max 9  --circuit_source paper
    python evaluation/generate_eval_circuits.py --mode oracle --n_min 2 --n_max 20 --circuit_source strict
    python evaluation/generate_eval_circuits.py --mode oracle --n_min 2 --n_max 20 --circuit_source paper

  Step 2 — Run evaluation (must match the circuit_source used in Step 1):
    python evaluation/phase-1/run_eval.py --mode full   --circuit_source strict
    python evaluation/phase-1/run_eval.py --mode full   --circuit_source paper
    python evaluation/phase-1/run_eval.py --mode oracle --circuit_source strict
    python evaluation/phase-1/run_eval.py --mode oracle --circuit_source paper

Results are saved to: evaluation/phase-1/results/
"""

import argparse
import os
import sys

_PHASE1_DIR  = os.path.dirname(os.path.abspath(__file__))
_EVAL_DIR    = os.path.dirname(_PHASE1_DIR)
_PROJECT_DIR = os.path.dirname(_EVAL_DIR)
sys.path.insert(0, _EVAL_DIR)

from eval_phase_1_2 import run_phase

# ─── Phase 1 model configuration ──────────────────────────────────────────────
MODEL_PATH             = os.path.join(_PROJECT_DIR, "saves/Meta-Llama-3-8B-Instruct/merged/GroverGPT+")
QUANTUM_TOKENIZER_PATH = MODEL_PATH   # GroverGPT+ tokeniser IS the quantum-native tokeniser

# The base model tokenizer directory was modified in-place during training setup
# (quantum tokens were added via extend_tokenizer_data_MMS.py).  Use the backup
# of the original unmodified tokenizer as the true base for CR/SRR computation.
BASE_TOKENIZER_PATH    = os.path.join(_PROJECT_DIR, "models/Meta-Llama-3-8B-Instruct/original_tokenizer_backup")

RESULTS_DIR    = os.path.join(_PHASE1_DIR, "results")
MANIFESTS_DIR  = os.path.join(_EVAL_DIR, "circuit_manifests")
MAX_MODEL_LEN  = 8192   # LLaMA 3 8B context window

# Phase 1 evaluation ranges (Research_Plan Section 3.3)
N_RANGES = {
    "full":   (2, 9),
    "oracle": (2, 20),
}


def main():
    parser = argparse.ArgumentParser(description="Phase 1 GroverGPT+ evaluation")
    parser.add_argument(
        "--mode", choices=["full", "oracle"], required=True,
        help="'full' = complete Grover circuit (n=2-9); 'oracle' = oracle-only (n=2-20)",
    )
    parser.add_argument(
        "--circuit_source", choices=["strict", "paper"], default="strict",
        help="'strict': fresh unseen circuits for in-training n (default); "
             "'paper': data_MMS circuits for all n (matches published evaluation)",
    )
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Override manifest path (default: auto-derived from mode and circuit_source)",
    )
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Override the merged model directory (default: merged/GroverGPT+). "
             "The model's own tokenizer is used as the quantum-native tokenizer.",
    )
    parser.add_argument(
        "--results_dir", type=str, default=None,
        help="Override the results output directory (default: phase-1/results/)",
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
        max_model_len=MAX_MODEL_LEN,
    )


if __name__ == "__main__":
    main()
