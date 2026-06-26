"""
Build a small JSONL of 3 mixed-k circuits per n (n=2..19) for the Phase 4
RET (Relative Execution Time) single-instance timing pass.

Methodology mirrors Phase 1 / Phase 2 (`eval_phase_1_2._sample_ret_circuits`):
3 circuits per n sampled from `data_MMS/grover_n{n}/` with a fixed seed,
mixed-k (drawn from the natural data_MMS k distribution, no per-k stratification).

The resulting JSONL is consumed by `run_eval.py` with `--max_concurrency 1`
so each chain is timed in isolation, free of continuous-batching interference.

Source : GroverGPT-plus/data_MMS/grover_n{n}/*.qasm
Output : <out_dir>/circuits.jsonl   (+ <out_dir>/qasm/*.qasm)
"""

import argparse
import json
import random
import re
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DATA_MMS = PROJECT / "GroverGPT-plus" / "data_MMS"

N_MIN, N_MAX = 2, 19
N_CIRCUITS_PER_N = 3
SEED = 42

# QASM filename pattern: grover_n<n>_k<k>_m<mask1>[_<mask2>...]
_CID_RE = re.compile(r"^grover_n(\d+)_k(\d+)_m(.+)$")


def _parse_cid(stem: str):
    """Return (n, k, marked_states_list) from filename stem like grover_n5_k2_m00000_00100."""
    m = _CID_RE.match(stem)
    if not m:
        return None
    n = int(m.group(1))
    k = int(m.group(2))
    marks_part = m.group(3)
    marked_states = marks_part.split("_")
    if len(marked_states) != k or not all(len(s) == n and set(s) <= {"0", "1"} for s in marked_states):
        return None
    return n, k, marked_states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n_per", type=int, default=N_CIRCUITS_PER_N)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    qasm_dir = out_dir / "qasm"
    qasm_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    total = 0
    per_n = {}

    with open(out_dir / "circuits.jsonl", "w") as f:
        for n in range(N_MIN, N_MAX + 1):
            pool = sorted((DATA_MMS / f"grover_n{n}").glob("*.qasm"))
            if not pool:
                print(f"  n={n}: NO POOL — skipping (data_MMS/grover_n{n}/ missing)")
                per_n[n] = 0
                continue
            picked = rng.sample(pool, min(args.n_per, len(pool)))
            for qasm_path in picked:
                parsed = _parse_cid(qasm_path.stem)
                if parsed is None:
                    print(f"  WARN: unparseable filename {qasm_path.stem} — skipping")
                    continue
                n_p, k_p, marked_states = parsed
                cid = qasm_path.stem
                qp = qasm_dir / f"{cid}.qasm"
                qp.write_text(qasm_path.read_text())
                rec = {
                    "circuit_id": cid,
                    "n": n_p,
                    "k": k_p,
                    "marked_states": marked_states,
                    "qasm_path": str(qp),
                }
                f.write(json.dumps(rec) + "\n")
                total += 1
            per_n[n] = len(picked)

    print(f"Wrote {total} RET circuits across n={N_MIN}..{N_MAX} (seed={args.seed})")
    for n in sorted(per_n):
        print(f"  n={n:>2}: {per_n[n]} circuits")
    print(f"\n-> {out_dir / 'circuits.jsonl'}")


if __name__ == "__main__":
    main()
