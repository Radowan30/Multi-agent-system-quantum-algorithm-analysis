"""
Phase 1 evaluation — GroverGPT+ (LLaMA 3 8B Instruct, quantum-native tokeniser).

Two-step workflow (from Reproduce Results/evaluation_pipeline, with WORK set):

  Step 1 — Generate the circuit manifests (run once, only needs Qiskit):
    python generate_eval_circuits.py --mode full   --n_min 2 --n_max 9  --circuit_source paper
    python generate_eval_circuits.py --mode oracle --n_min 2 --n_max 20 --circuit_source paper

  Step 2 — Run the evaluation:
    python run_eval_phase1.py --mode full   --circuit_source paper
    python run_eval_phase1.py --mode oracle --circuit_source paper

Defaults: model $WORK/saves/Meta-Llama-3-8B-Instruct/merged/GroverGPT+_alpha32,
results $WORK/results/phase-1, manifests $WORK/eval_circuits/manifests.
"""

import argparse
import os
import sys

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EVAL_DIR)

from eval_phase_1_2 import run_phase
from workdir import manifests_dir, models_dir, results_dir, saves_dir

MAX_MODEL_LEN = 8192   # LLaMA 3 8B context window

# Phase 1 evaluation ranges
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
        "--circuit_source", choices=["strict", "paper"], default="paper",
        help="'paper' (default, used for the published results): data_MMS circuits "
             "for all n; 'strict': fresh unseen circuits for in-training n",
    )
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Override manifest path (default: auto-derived from mode and circuit_source)",
    )
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Merged model directory (default: $WORK/saves/Meta-Llama-3-8B-Instruct/"
             "merged/GroverGPT+_alpha32). Its tokenizer is the quantum-native tokenizer.",
    )
    parser.add_argument(
        "--base_tokenizer", type=str, default=None,
        help="Unextended tokenizer used as the CR/SRR baseline "
             "(default: the downloaded base model, $WORK/models/Meta-Llama-3-8B-Instruct)",
    )
    parser.add_argument(
        "--results_dir", type=str, default=None,
        help="Results output directory (default: $WORK/results/phase-1)",
    )
    args = parser.parse_args()

    model_path = args.model_path or os.path.join(
        saves_dir(), "Meta-Llama-3-8B-Instruct", "merged", "GroverGPT+_alpha32")
    base_tokenizer = args.base_tokenizer or os.path.join(models_dir(), "Meta-Llama-3-8B-Instruct")
    out_dir = args.results_dir or os.path.join(results_dir(), "phase-1")
    n_min, n_max = N_RANGES[args.mode]
    manifest_path = args.manifest or os.path.join(
        manifests_dir(), f"circuits_{args.mode}_{n_min}_{n_max}_{args.circuit_source}.json"
    )

    if not os.path.exists(manifest_path):
        print(f"[Error] Manifest not found: {manifest_path}")
        print(f"  Run first:")
        print(f"    python generate_eval_circuits.py "
              f"--mode {args.mode} --n_min {n_min} --n_max {n_max} "
              f"--circuit_source {args.circuit_source}")
        sys.exit(1)

    run_phase(
        manifest_path=manifest_path,
        model_path=model_path,
        base_tokenizer_path=base_tokenizer,
        quantum_tokenizer_path=model_path,
        results_dir=out_dir,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_tokens=8192,
        max_model_len=MAX_MODEL_LEN,
    )


if __name__ == "__main__":
    main()
