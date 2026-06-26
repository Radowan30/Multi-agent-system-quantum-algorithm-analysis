"""
Phase 2 RET plot, normalized to a configurable reference n.

The Phase 2 LLaMA 3.1 fine-tunes fail catastrophically at n=2 — the model
emits non-stopping output that runs to max_new_tokens. This inflates the
n=2 reference inference time (~88 s in alpha32 results), which in turn
forces all RET = T(n) / T(2) ratios below 1.0 for the n's where the model
actually behaves. The default log-y-axis floor at 10^0 then visually hides
those points.

This script recomputes RET against a non-default reference (default n=3,
the first n where alpha32 produces clean outputs) and produces a plot that
starts at the reference n. RET uncertainty is propagated through the
division.

Usage:
    python evaluation/phase-2/plot_ret_normalized.py \
        --results_dir evaluation/phase-2/results_llama31_alpha32 \
        --circuit_source paper \
        --normalize_n 3 \
        --label "LLaMA 3.1 8B (Phase 2, normalized to n=3)"
"""

import argparse
import json
import math
import os
import sys

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt


_BLUE = "#2C72B0"
_AXES_BG = "#F5F3FF"


def _apply_paper_style(ax):
    ax.set_facecolor(_AXES_BG)
    ax.grid(True, color="#D8D0F0", linewidth=0.6, zorder=0)
    ax.tick_params(labelsize=9)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


def _recompute_ret(results: dict, ref_n: int):
    """Recompute RET = T(n) / T(ref_n) with error propagation.

    Returns parallel lists (ns, rets, errs) for n >= ref_n where mean_time_s
    is present and positive.
    """
    if ref_n not in results:
        raise ValueError(f"reference n={ref_n} not present in results")
    ref = results[ref_n]
    t_ref = ref.get("mean_time_s")
    s_ref = ref.get("time_std_s", 0.0) or 0.0
    if not t_ref or t_ref <= 0:
        raise ValueError(
            f"reference n={ref_n} has invalid mean_time_s={t_ref}"
        )

    ns, rets, errs = [], [], []
    for n in sorted(results):
        if n < ref_n:
            continue
        r = results[n]
        t_n = r.get("mean_time_s")
        s_n = r.get("time_std_s", 0.0) or 0.0
        if not t_n or t_n <= 0:
            continue
        ret = t_n / t_ref
        # sigma_RET = RET * sqrt((s_n/t_n)^2 + (s_ref/t_ref)^2)
        rel_n = (s_n / t_n) ** 2 if t_n else 0.0
        rel_ref = (s_ref / t_ref) ** 2 if t_ref else 0.0
        err = ret * math.sqrt(rel_n + rel_ref)
        ns.append(n)
        rets.append(ret)
        errs.append(err)
    return ns, rets, errs


def plot_ret(results: dict, ref_n: int, label: str, out_path: str) -> None:
    ns, rets, errs = _recompute_ret(results, ref_n)

    if not ns:
        print(f"[skip] No usable timing data for n >= {ref_n}")
        return

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.errorbar(
        ns, rets, yerr=errs,
        fmt="s--", color=_BLUE, markersize=5, linewidth=1.4,
        capsize=3, elinewidth=1.0, capthick=1.0, ecolor=_BLUE,
        label=label,
    )

    # Set log y-axis with a floor at 1 (since RET = 1 at the reference n).
    # Cap the top at the next power-of-10 above the largest RET seen, with a
    # minimum range of one decade.
    max_ret_with_err = max((r + e for r, e in zip(rets, errs)))
    top_decade = max(10.0, 10 ** math.ceil(math.log10(max_ret_with_err)))
    ax.set_yscale("log")
    ax.set_ylim(1, top_decade)

    ax.set_xlabel("Number of Qubits", fontsize=9)
    ax.set_ylabel(
        f"Relative Execution Time\n(normalized to n={ref_n})", fontsize=9
    )
    ax.set_xticks(ns)
    ax.set_xticklabels(ns, fontsize=8)

    # Y ticks at the two decade bounds (matching paper style).
    ax.yaxis.set_major_locator(
        matplotlib.ticker.FixedLocator([1, top_decade])
    )
    ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterMathtext())
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    ax.legend(fontsize=8, loc="upper left")
    _apply_paper_style(ax)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", required=True,
                    help="Directory containing results_full_<n_min>_<n_max>_<src>.json")
    ap.add_argument("--circuit_source", choices=["strict", "paper"], default="paper")
    ap.add_argument("--n_min", type=int, default=2)
    ap.add_argument("--n_max", type=int, default=19)
    ap.add_argument("--normalize_n", type=int, default=3,
                    help="Qubit count to use as RET reference (default: 3)")
    ap.add_argument("--label", default=None,
                    help="Legend label (default auto-generated)")
    ap.add_argument("--output_filename", default=None,
                    help="Output filename (default: ret_normalized_n<N>_<src>.png)")
    args = ap.parse_args()

    fname = f"results_full_{args.n_min}_{args.n_max}_{args.circuit_source}.json"
    path = os.path.join(args.results_dir, fname)
    if not os.path.exists(path):
        print(f"[error] Results file not found: {path}", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        results = {int(k): v for k, v in json.load(f).items()}

    label = args.label or f"LLaMA 3.1 8B (Phase 2), normalized to n={args.normalize_n}"
    out_filename = (
        args.output_filename
        or f"ret_normalized_n{args.normalize_n}_{args.circuit_source}.png"
    )
    out_path = os.path.join(args.results_dir, out_filename)

    plot_ret(results, args.normalize_n, label, out_path)


if __name__ == "__main__":
    main()
