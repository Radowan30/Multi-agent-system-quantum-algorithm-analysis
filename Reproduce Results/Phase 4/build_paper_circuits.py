"""
Build a JSONL of ALL circuits in the paper manifest (~4024 circuits across
n=2..19, k=1..3) for the Phase 4 FULL paper-comparison evaluation.

Direct counterpart to Phase 1 / Phase 2's full evaluation (which run on the
same manifest via `eval_phase_1_2.py`). The resulting JSONL is consumed by
`run_eval.py` to produce per-circuit chain results.

Per-cell counts (paper-target, set by `circuit_utils.target_circuit_count`):
  n=2: 4   n=3: 36   n=4..6: 100   n=7: 128   n=8: 256   n=9..19: 300 each
(no degenerate Grover cells: n=2 k=2/k=3, n=3 k=3 — see project_data_mms_holes)

Source : evaluation/circuit_manifests/circuits_full_2_19_paper.json
Output : <out_dir>/circuits.jsonl   (+ <out_dir>/qasm/*.qasm)
"""

import argparse
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT / "evaluation" / "circuit_manifests" / "circuits_full_2_19_paper.json"

N_MIN, N_MAX = 2, 19


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    qasm_dir = out_dir / "qasm"
    qasm_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(MANIFEST.read_text())
    total = 0
    per_n = {}

    with open(out_dir / "circuits.jsonl", "w") as f:
        for n_str, circs in sorted(manifest["by_n"].items(), key=lambda x: int(x[0])):
            n = int(n_str)
            if not (N_MIN <= n <= N_MAX):
                continue
            per_n[n] = len(circs)
            # Deterministic ordering (manifest's source_file is unique-per-circuit)
            for c in sorted(circs, key=lambda x: x["source_file"]):
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
                total += 1

    print(f"Wrote {total} circuits across n={N_MIN}..{N_MAX}")
    for n in sorted(per_n):
        print(f"  n={n:>2}: {per_n[n]:>4} circuits")
    print(f"\n-> {out_dir / 'circuits.jsonl'}")


if __name__ == "__main__":
    main()
