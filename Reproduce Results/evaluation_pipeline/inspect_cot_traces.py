"""Inspect a GroverGPT+ model's CoT reasoning traces, one (n, t) cell at a time.

Reusable across phases/models. For each (n, t) cell with data in data_MMS, this
script picks one random circuit (deterministic via --seed; n values listed in
--include_all_at_n are exhaustive instead of random-pick-one). It runs
single-instance inference (so per-circuit execution time is meaningful),
captures the raw model output, reconstructs the analytical ground-truth CoT via
dataset_generate_MMS.py, computes SA + CF, and writes a side-by-side markdown
comparison document.

Output: <results_dir>/cot_traces.md — model output (left) vs ground truth CoT
(right), with SA, CF, and per-circuit single-instance timing.

Usage:
  venv-eval/bin/python evaluation/inspect_cot_traces.py \\
      --model_path saves/.../merged/GroverGPT+_alpha16_cl4000 \\
      --results_dir evaluation/phase-1/results_alpha16_cl4000

  # Custom n-range (e.g. for a Phase 2 model trained on full circuits up to n=19):
  venv-eval/bin/python evaluation/inspect_cot_traces.py \\
      --model_path ... --results_dir ... --full_n_range 2-19

  # Include all circuits (not just one per (n,t) cell) at multiple small n:
  venv-eval/bin/python evaluation/inspect_cot_traces.py \\
      --model_path ... --results_dir ... --include_all_at_n 2,3
"""

import argparse
import datetime
import glob
import html
import os
import random
import sys
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_EVAL_DIR)
sys.path.insert(0, _EVAL_DIR)
sys.path.insert(0, os.path.join(_PROJECT_DIR, "GroverGPT-plus"))

from metrics import classical_fidelity, grover_ground_truth, search_accuracy  # noqa: E402
from parse_output import parse_model_output                                     # noqa: E402
from dataset_generate_MMS import (                                              # noqa: E402
    analyze_marked_state, extract_oracle_gate, extract_oracle_structure,
    generate_reasoning,
)

DEFAULT_DATA_DIR = os.path.join(_PROJECT_DIR, "GroverGPT-plus", "data_MMS")


# ---------------------------------------------------------------------------
# Circuit sampling
# ---------------------------------------------------------------------------

def _parse_n_range(s):
    """'2-9' -> range(2, 10). '5' -> range(5, 6)."""
    if "-" in s:
        a, b = s.split("-")
        return range(int(a), int(b) + 1)
    n = int(s)
    return range(n, n + 1)


def _parse_int_set(s):
    """'2,3' -> {2, 3}. '' -> set(). '2' -> {2}."""
    return {int(x) for x in s.split(",") if x.strip()}


def sample_circuits(data_dir, mode, n_range, seed, include_all_at_n):
    """For each (n, t) cell with data, pick one random circuit — except for n
    values in `include_all_at_n`, where every circuit in the cell is taken
    (useful for the smallest n with only a handful of total circuits).

    Returns: list of (n, t, qasm_for_model, marked_states_list, filename).
    `qasm_for_model` is the full QASM for mode='full' or just the Oracle gate
    definition for mode='oracle'.
    """
    rng = random.Random(seed)
    out = []
    for n in n_range:
        for t in (1, 2, 3):
            files = sorted(glob.glob(f"{data_dir}/grover_n{n}/*_k{t}_*.qasm"))
            if not files:
                continue
            chosen = files if n in include_all_at_n else [rng.choice(files)]
            for f in chosen:
                with open(f) as fp:
                    content = fp.read()
                base = os.path.basename(f).replace(".qasm", "")
                marked = base.split("_m", 1)[1].split("_")
                qasm_in = extract_oracle_gate(content) if mode == "oracle" else content
                if qasm_in is None:
                    continue
                out.append((n, t, qasm_in, marked, os.path.basename(f)))
    return out


def reference_full_path(data_dir, fname):
    """Path to the full data_MMS QASM for a given basename — needed for ground
    truth CoT reconstruction (the oracle-only model input lacks the bits we
    need to re-derive marked states the same way)."""
    n_str = fname.split("_")[1]   # 'n3'
    return os.path.join(data_dir, f"grover_{n_str}", fname)


# ---------------------------------------------------------------------------
# Ground-truth CoT (re-derived via dataset_generate_MMS.py — already audited
# byte-faithful against the published Grover_FullCircuit/Oracle JSON labels)
# ---------------------------------------------------------------------------

def ground_truth_cot(qasm_full_content):
    res = extract_oracle_structure(qasm_full_content)
    if not res or not res[0]:
        return "<could not extract oracle from QASM>"
    params, ops, oh, ob = res
    marked, blocks, x_ops = analyze_marked_state(params, ops)
    return generate_reasoning(params, ops, marked, blocks, x_ops, oh, ob)


# ---------------------------------------------------------------------------
# Per-circuit single-instance inference + scoring
# ---------------------------------------------------------------------------

def cot_capture(llm, sp, tok, circuits, data_dir):
    """Single-instance generate per circuit; capture output + time + SA + both CFs.

    Both CF modes are computed per circuit so the document can show them side
    by side without the user having to choose at run time:
      - CF (raw): paper-faithful, unbounded — can exceed 1.0 when the model
        over-sums its predicted distribution.
      - CF (renorm): the model's distribution is rescaled to sum to 1 first,
        bounding CF in [0, 1].
    """
    rows = []
    for n, t, qasm_in, marked, fname in circuits:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": qasm_in}],
            tokenize=False, add_generation_prompt=True,
        )
        t0 = time.perf_counter()
        gen = llm.generate([prompt], sp)[0].outputs[0].text
        elapsed = time.perf_counter() - t0
        pred, _ = parse_model_output(gen)
        if pred is None:
            sa = cf_raw = cf_renorm = 0.0
        else:
            gt = grover_ground_truth(n, marked)
            sa = search_accuracy(pred, marked)
            cf_raw    = classical_fidelity(pred, gt, mode="raw")
            cf_renorm = classical_fidelity(pred, gt, mode="renormalize")
        with open(reference_full_path(data_dir, fname)) as fp:
            gt_cot = ground_truth_cot(fp.read())
        rows.append(dict(
            n=n, t=t, fname=fname, marked=marked,
            model_out=gen, gt_cot=gt_cot,
            sa=sa, cf_raw=cf_raw, cf_renorm=cf_renorm, time=elapsed,
        ))
        print(f"  n={n} t={t}: SA={sa:.2f} CF(raw)={cf_raw:.3f} CF(renorm)={cf_renorm:.3f} "
              f"time={elapsed:.2f}s  ({fname})")
    return rows


# ---------------------------------------------------------------------------
# Document writer
# ---------------------------------------------------------------------------

def write_doc(out_path, model_path, data_dir, seed, include_all_at_n,
              full_rs, oracle_rs):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    include_all_str = (
        ", ".join(str(n) for n in sorted(include_all_at_n))
        if include_all_at_n else "none"
    )
    with open(out_path, "w") as f:
        f.write(f"# CoT Reasoning Trace Comparison — `{os.path.basename(model_path)}`\n\n")
        f.write(f"_Generated: {timestamp}_\n\n")
        f.write("| Field | Value |\n|---|---|\n")
        f.write(f"| Model | `{model_path}` |\n")
        f.write(f"| Data source | `{data_dir}` |\n")
        f.write(f"| Random seed | `{seed}` |\n")
        f.write(f"| All circuits taken (no random pick) at n | `{include_all_str}` |\n\n")
        f.write(
            "For each circuit: **left** = raw model output (as emitted, no editing); "
            "**right** = analytical ground-truth CoT (regenerated via "
            "`dataset_generate_MMS.py` on the same QASM). One random circuit per "
            "(n, t) cell where `data_MMS` has coverage — except n listed above, "
            "where every circuit in the cell is included. Single-instance "
            "generation timing per circuit.\n\n"
            "Each circuit reports **both** CF values: **CF (raw)** is "
            "paper-faithful and unbounded (can exceed 1.0 when the model "
            "over-sums its predicted distribution); **CF (renorm)** rescales "
            "the predicted distribution to sum to 1 first, bounding CF in "
            "[0, 1]. See `evaluation/metrics.py:classical_fidelity` for the "
            "exact definitions.\n\n"
        )
        for section, rs in [
            ("Full Circuit Inputs", full_rs),
            ("Oracle-only Inputs", oracle_rs),
        ]:
            ns_in_section = sorted({r["n"] for r in rs}) if rs else []
            header_range = (
                f"n = {ns_in_section[0]}–{ns_in_section[-1]}"
                if ns_in_section else "no data"
            )
            f.write(f"---\n\n## {section} ({header_range})\n\n")
            if not rs:
                f.write("_(no circuits captured)_\n\n")
                continue
            for r in rs:
                f.write(f"### n={r['n']}, t={r['t']} — `{r['fname']}`\n\n")
                f.write(
                    f"Marked states: `{r['marked']}` &nbsp;|&nbsp; "
                    f"**SA** = `{r['sa']:.3f}` &nbsp;|&nbsp; "
                    f"**CF (raw)** = `{r['cf_raw']:.3f}` &nbsp;|&nbsp; "
                    f"**CF (renorm)** = `{r['cf_renorm']:.3f}` &nbsp;|&nbsp; "
                    f"**time** = `{r['time']:.2f}s`\n\n"
                )
                f.write(
                    '<table>\n'
                    '<tr><th width="50%">Model output</th>'
                    '<th width="50%">Ground truth CoT</th></tr>\n'
                    '<tr>\n'
                    f"<td><pre>{html.escape(r['model_out'])}</pre></td>\n"
                    f"<td><pre>{html.escape(r['gt_cot'])}</pre></td>\n"
                    '</tr>\n</table>\n\n'
                )
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_path", required=True, help="Merged GroverGPT+ model directory")
    ap.add_argument("--results_dir", required=True, help="Where to write cot_traces.md")
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR,
                    help=f"data_MMS directory (default: {DEFAULT_DATA_DIR})")
    ap.add_argument("--full_n_range", default="2-9",
                    help="n range for the full-circuit section (default: 2-9, matches Phase 1)")
    ap.add_argument("--oracle_n_range", default="2-20",
                    help="n range for the oracle-only section (default: 2-20, matches Phase 1)")
    ap.add_argument("--include_all_at_n", default="2",
                    help="Comma-separated n values for which to include EVERY circuit "
                         "in each (n, t) cell instead of picking one randomly "
                         "(useful for very small n where the cell is tiny). Default: 2.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    full_n        = _parse_n_range(args.full_n_range)
    oracle_n      = _parse_n_range(args.oracle_n_range)
    include_all_n = _parse_int_set(args.include_all_at_n)

    full_circs   = sample_circuits(args.data_dir, "full",   full_n,   args.seed, include_all_n)
    oracle_circs = sample_circuits(args.data_dir, "oracle", oracle_n, args.seed, include_all_n)
    print(f"Sampled full={len(full_circs)} oracle={len(oracle_circs)} circuits")

    tok = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(
        model=args.model_path, dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sp = SamplingParams(
        temperature=0.0, max_tokens=args.max_tokens,
        stop=["<|eot_id|>", "<|end_of_text|>"],
    )

    print(f"\n=== CoT capture — full ===")
    full_rs = cot_capture(llm, sp, tok, full_circs, args.data_dir)
    print(f"\n=== CoT capture — oracle ===")
    oracle_rs = cot_capture(llm, sp, tok, oracle_circs, args.data_dir)

    write_doc(
        out_path=os.path.join(args.results_dir, "cot_traces.md"),
        model_path=args.model_path,
        data_dir=args.data_dir,
        seed=args.seed,
        include_all_at_n=include_all_n,
        full_rs=full_rs,
        oracle_rs=oracle_rs,
    )


if __name__ == "__main__":
    main()
