"""
Phase 2 evaluation — fine-tuned larger-context models (Llama-3-8B-Instruct-262k
and LLaMA 3.1 8B Instruct, same quantum-native 139-token tokeniser as Phase 1).

Training data is unchanged from Phase 1: Grover_FullCircuit_2_7_MMS +
Grover_Oracle_2_10_MMS. Only the evaluation range is extended — the larger
context window allows full-circuit inputs up to n=19 (vs n=9 in Phase 1),
while oracle-only stays at n=2..20 for direct comparability.

Two-step workflow (from Reproduce Results/evaluation_pipeline, with WORK set):

  Step 1 — Generate the circuit manifests (run once, only needs Qiskit):
    python generate_eval_circuits.py --mode full   --n_min 2 --n_max 19 --circuit_source paper
    python generate_eval_circuits.py --mode oracle --n_min 2 --n_max 20 --circuit_source paper

  Step 2 — Run the evaluation:
    python run_eval_phase2.py --mode full   --model_path <merged> --results_dir <dir>
    python run_eval_phase2.py --mode oracle --model_path <merged> --results_dir <dir> --max_model_len 8192
"""

import argparse
import os
import sys

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EVAL_DIR)

from eval_phase_1_2 import run_phase
from workdir import manifests_dir, models_dir

MAX_MODEL_LEN = 120000        # margin under LLaMA 3.1's 131 072-token window

# Phase 2 evaluation ranges
N_RANGES = {
    "full":   (2, 19),
    "oracle": (2, 20),
}


def main():
    parser = argparse.ArgumentParser(description="Phase 2 larger-context model evaluation")
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
        "--model_path", type=str, required=True,
        help="Merged model directory. Its tokenizer is the quantum-native tokenizer.",
    )
    parser.add_argument(
        "--base_tokenizer", type=str, default=None,
        help="Unextended tokenizer used as the CR/SRR baseline (default: the downloaded "
             "base model $WORK/models/Llama-3.1-8B-Instruct; every LLaMA 3 family base "
             "has the same 128,256-token vocabulary, so CR/SRR do not depend on this choice)",
    )
    parser.add_argument(
        "--results_dir", type=str, required=True,
        help="Results output directory.",
    )
    parser.add_argument(
        "--max_model_len", type=int, default=MAX_MODEL_LEN,
        help="Override max_model_len (default 120000 — LLaMA 3.1 supports up to 131072).",
    )
    args = parser.parse_args()

    base_tokenizer = args.base_tokenizer or os.path.join(models_dir(), "Llama-3.1-8B-Instruct")
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
        model_path=args.model_path,
        base_tokenizer_path=base_tokenizer,
        quantum_tokenizer_path=args.model_path,
        results_dir=args.results_dir,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_tokens=8192,
        max_model_len=args.max_model_len,
    )


if __name__ == "__main__":
    main()
