"""
Build the Phase 3 evaluation set: one circuit per (n, k) stratum for
n = 2..19 and k = 1..3, drawn from the paper manifest.

The manifest has 51 non-empty strata (54 minus the three degenerate Grover
cells it intentionally leaves out: n=2 k=2, n=2 k=3, n=3 k=3), so the set has
51 circuits: one at n=2, two at n=3 and three at every n >= 4. A small set
keeps the cost of evaluating four LLM variations manageable.

Selection is deterministic: strata are visited in sorted (n, k) order and one
random.Random(seed) generator picks one circuit per stratum from the
manifest's list. With seed 42 this reproduces the exact 51 circuits behind
the published Phase 3 results.

Source : $WORK/eval_circuits/manifests/circuits_full_2_19_paper.json
         (made by evaluation_pipeline/generate_eval_circuits.py)
Output : <out_dir>/circuits.jsonl   (+ <out_dir>/qasm/*.qasm)
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation_pipeline"))
from workdir import manifests_dir  # noqa: E402

N_MIN, N_MAX = 2, 19


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    manifest_path = Path(manifests_dir()) / "circuits_full_2_19_paper.json"
    if not manifest_path.exists():
        sys.exit(f"Manifest not found: {manifest_path}\nRun evaluation_pipeline/generate_eval_circuits.py "
                 "--mode full --n_min 2 --n_max 19 --circuit_source paper first.")
    manifest = json.loads(manifest_path.read_text())

    # Group the manifest's circuits by (n, k), keeping the manifest order.
    by_cell = defaultdict(list)
    for n_str, circs in manifest["by_n"].items():
        n = int(n_str)
        if N_MIN <= n <= N_MAX:
            for c in circs:
                by_cell[(n, c["k"])].append(c)

    rng = random.Random(args.seed)
    picked = [rng.choice(by_cell[cell]) for cell in sorted(by_cell)]

    out_dir = Path(args.out_dir).resolve()
    qasm_dir = out_dir / "qasm"
    qasm_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "circuits.jsonl"
    with open(jsonl_path, "w") as f:
        for c in picked:
            cid = Path(c["source_file"]).stem
            qp = qasm_dir / f"{cid}.qasm"
            qp.write_text(c["qasm"])
            f.write(json.dumps({
                "circuit_id": cid,
                "n": c["n"],
                "k": c["k"],
                "marked_states": c["marked_states"],
                "qasm_path": str(qp),
            }) + "\n")

    per_n = defaultdict(int)
    for c in picked:
        per_n[c["n"]] += 1
    print(f"Phase 3 set: {len(picked)} circuits, one per (n, k) stratum (seed={args.seed})")
    print("  " + "  ".join(f"n={n}:{per_n[n]}" for n in sorted(per_n)))
    print(f"Wrote {jsonl_path}")


if __name__ == "__main__":
    main()
