"""
Generate and save all evaluation circuits to disk before running any phase evaluation.

This script uses venv-eval (Qiskit). Run it once per mode to pre-generate QASM
files and a manifest JSON that the evaluation pipeline reads without needing Qiskit.

Usage:
  # Phase 1 — strict (fresh unseen circuits for in-training n)
  python evaluation/generate_eval_circuits.py --mode full   --n_min 2 --n_max 9  --circuit_source strict
  python evaluation/generate_eval_circuits.py --mode oracle --n_min 2 --n_max 20 --circuit_source strict

  # Phase 1 — paper (data_MMS circuits for all n, matching published evaluation)
  python evaluation/generate_eval_circuits.py --mode full   --n_min 2 --n_max 9  --circuit_source paper
  python evaluation/generate_eval_circuits.py --mode oracle --n_min 2 --n_max 20 --circuit_source paper

  # Phase 2 (full circuit only, extended range)
  python evaluation/generate_eval_circuits.py --mode full --n_min 2 --n_max 19 --circuit_source strict

Output:
  evaluation/circuit_manifests/circuits_{mode}_{n_min}_{n_max}_{circuit_source}.json
  evaluation/data_MMS_eval/{circuit_source}/grover_n{n}/*.qasm
    (strict mode only; paper mode reads directly from data_MMS)
"""

import argparse
import json
import os
import random
import sys

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EVAL_DIR)

from circuit_utils import get_circuits, target_circuit_count

_MANIFESTS_DIR = os.path.join(_EVAL_DIR, "circuit_manifests")


def build_manifest(mode: str, n_min: int, n_max: int, seed: int,
                   circuit_source: str) -> dict:
    rng = random.Random(seed)
    by_n = {}
    for n in range(n_min, n_max + 1):
        print(f"  [n={n}] loading/generating {mode} circuits ({circuit_source}) ...")
        pool = get_circuits(n, mode, seed=seed, circuit_source=circuit_source)

        # For in-training-range n with circuit_source='strict', get_circuits
        # returns all unseen circuits (up to _MAX_UNSEEN_PER_K per k) — which
        # may exceed the evaluation target.  Sample down to target_circuit_count(n).
        # OOD circuits and 'paper' source are already bounded to target_circuit_count(n).
        target = target_circuit_count(n)
        circuits = rng.sample(pool, min(target, len(pool)))

        by_n[str(n)] = [
            {
                "n":             c["n"],
                "k":             c["k"],
                "marked_states": c["marked_states"],
                "qasm":          c["qasm"],
                "source_file":   c["source_file"],
            }
            for c in circuits
        ]
        print(f"  [n={n}] pool={len(pool)} → manifest={len(circuits)} circuits")
    return {
        "mode":           mode,
        "n_min":          n_min,
        "n_max":          n_max,
        "seed":           seed,
        "circuit_source": circuit_source,
        "by_n":           by_n,
    }


def main():
    parser = argparse.ArgumentParser(description="Pre-generate evaluation circuits and manifest")
    parser.add_argument("--mode",           choices=["full", "oracle"], required=True)
    parser.add_argument("--n_min",          type=int, required=True)
    parser.add_argument("--n_max",          type=int, required=True)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--circuit_source", choices=["strict", "paper"], default="strict",
                        help="'strict': fresh unseen circuits for in-training n (default); "
                             "'paper': data_MMS circuits for all n (matches published evaluation)")
    parser.add_argument("--output",         type=str, default=None,
                        help="Output manifest path (default: auto-named in evaluation/circuit_manifests/)")
    args = parser.parse_args()

    os.makedirs(_MANIFESTS_DIR, exist_ok=True)
    out = args.output or os.path.join(
        _MANIFESTS_DIR,
        f"circuits_{args.mode}_{args.n_min}_{args.n_max}_{args.circuit_source}.json",
    )

    print(f"Building {args.mode} manifest ({args.circuit_source}) for n={args.n_min}..{args.n_max} ...")
    manifest = build_manifest(args.mode, args.n_min, args.n_max, args.seed, args.circuit_source)

    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)

    total = sum(len(v) for v in manifest["by_n"].values())
    print(f"\nDone — {total} circuits written to manifest: {out}")


if __name__ == "__main__":
    main()
