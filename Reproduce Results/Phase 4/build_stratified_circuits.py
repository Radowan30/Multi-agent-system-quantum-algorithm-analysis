"""
Build a stratified 3-circuits-per-(n,k)-cell JSONL for the Phase 4 full
evaluation.

For n=2..19 across k=1..3 the paper manifest defines 51 strata (the
expected 54 minus the 3 missing degenerate cells: n=2 k=2, n=2 k=3,
n=3 k=3 — see [[project_data_mms_holes]]). 3 random circuits per
stratum → 153 circuits total.

Within-stratum variance from 3 samples lets per-cell SA/CF be reported
with a meaningful std, which the single-sample Phase 3 preliminary
couldn't do.

Sampling is deterministic — `random.Random(seed).sample(...)` over the
manifest's per-n circuit list. Seed 42 unless `--seed` overrides.

Source: evaluation/circuit_manifests/circuits_full_2_19_paper.json
Output: <out_dir>/circuits.jsonl  (+ <out_dir>/qasm/*.qasm)
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT / "evaluation" / "circuit_manifests" / "circuits_full_2_19_paper.json"

N_MIN, N_MAX = 2, 19
PER_STRATUM = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    qasm_dir = out_dir / "qasm"
    qasm_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(MANIFEST.read_text())
    rng = random.Random(args.seed)

    # Group manifest circuits by (n, k).
    by_cell = defaultdict(list)
    for n_str, circs in manifest["by_n"].items():
        n = int(n_str)
        if not (N_MIN <= n <= N_MAX):
            continue
        for c in circs:
            by_cell[(n, c["k"])].append(c)

    rows = []
    skipped_cells = []
    for (n, k), circs in sorted(by_cell.items()):
        if len(circs) < PER_STRATUM:
            skipped_cells.append((n, k, len(circs)))
            continue
        # Sort for determinism, then sample. (Manifest order is already
        # deterministic but explicit sort guards against any upstream churn.)
        circs_sorted = sorted(circs, key=lambda c: c["source_file"])
        picked = rng.sample(circs_sorted, PER_STRATUM)
        for c in picked:
            rows.append((n, k, c))

    if skipped_cells:
        print("Strata with too few circuits (skipped):")
        for n, k, cnt in skipped_cells:
            print(f"  n={n} k={k}: only {cnt} available, need {PER_STRATUM}")

    jsonl_path = out_dir / "circuits.jsonl"
    with open(jsonl_path, "w") as f:
        for n, k, c in rows:
            cid = Path(c["source_file"]).stem
            qp = qasm_dir / f"{cid}.qasm"
            qp.write_text(c["qasm"])
            rec = {
                "circuit_id": cid,
                "n": c["n"],
                "k": c["k"],
                "marked_states": c["marked_states"],
                "qasm_path": str(qp),
            }
            f.write(json.dumps(rec) + "\n")

    # Brief summary per n
    per_n = defaultdict(int)
    for n, k, _ in rows:
        per_n[n] += 1
    print(f"\nStratified sample: {len(rows)} circuits across "
          f"{len({(n, k) for n, k, _ in rows})} strata "
          f"(seed={args.seed})")
    for n in sorted(per_n):
        print(f"  n={n}: {per_n[n]} circuits")
    print(f"\nWrote {jsonl_path}")


if __name__ == "__main__":
    main()
