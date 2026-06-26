"""
Plot oracle-extraction accuracy vs n.

Phase-agnostic: takes one or more `oracle_extraction_*.json` files (produced by
`oracle_extraction_accuracy.py`) and plots accuracy curves.

Two layouts:

(1) SINGLE (default): one axes, all curves overlaid. Useful for cross-model
    comparison within a single input mode.

(2) DUAL (--dual): side-by-side subplots — full-circuit input on the left,
    oracle-only input on the right, with a shared y-axis. Use this when
    comparing across input modes, because the two modes have different
    training ranges and a same-n overlay can mislead. Optionally shade the
    in-training-distribution range per mode (--train_full_range,
    --train_oracle_range) so the reader sees the OOD performance comparison
    rather than a direct same-n comparison.

Typical uses:

  # Single, cross-model, one mode
  python evaluation/plot_oracle_accuracy.py \\
      --input results_alpha16/oracle_extraction_full_2_9_paper.json "alpha16" \\
      --input results_alpha32/oracle_extraction_full_2_9_paper.json "alpha32" \\
      --output comparison_full.png

  # Dual, single model, both modes — for the OOD-comparison narrative
  python evaluation/plot_oracle_accuracy.py --dual \\
      --full   results_alpha16/oracle_extraction_full_2_9_paper.json    "alpha16" \\
      --oracle results_alpha16/oracle_extraction_oracle_2_20_paper.json "alpha16" \\
      --train_full_range  2,7   \\
      --train_oracle_range 2,10 \\
      --output dual_alpha16.png

  # Dual, both models on both axes
  python evaluation/plot_oracle_accuracy.py --dual \\
      --full   results_alpha16/oracle_extraction_full_2_9_paper.json    "alpha16" \\
      --full   results_alpha32/oracle_extraction_full_2_9_paper.json    "alpha32" \\
      --oracle results_alpha16/oracle_extraction_oracle_2_20_paper.json "alpha16" \\
      --oracle results_alpha32/oracle_extraction_oracle_2_20_paper.json "alpha32" \\
      --train_full_range  2,7   \\
      --train_oracle_range 2,10 \\
      --output dual_cross_model.png
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


_COLORS  = ["#7B52A6", "#D4820A", "#2C72B0", "#5BA85F", "#B83C3C"]
_MARKERS = ["o", "s", "^", "D", "v"]
_AXES_BG = "#F5F3FF"
_IID_SHADE = "#A8D8A8"  # light green — "in training distribution" zone


def load_curve(path: str):
    """Returns (ns: list[int], accuracies: list[float], mode: str)."""
    with open(path) as f:
        data = json.load(f)
    items = sorted(data["per_n"].items(), key=lambda kv: int(kv[0]))
    ns   = [int(k) for k, _ in items]
    accs = [v["accuracy"] for _, v in items]
    return ns, accs, data.get("mode", "?")


def _draw_curves(ax, curves, color_offset=0):
    all_ns = set()
    for i, (label, path) in enumerate(curves):
        ns, accs, mode = load_curve(path)
        all_ns.update(ns)
        full_label = label or f"{os.path.basename(path)} ({mode})"
        ax.plot(
            ns, accs,
            color=_COLORS[(i + color_offset) % len(_COLORS)],
            marker=_MARKERS[(i + color_offset) % len(_MARKERS)],
            linewidth=2, markersize=7,
            label=full_label,
        )
    return all_ns


def _shade_training(ax, train_range):
    if train_range is None:
        return
    lo, hi = train_range
    ax.axvspan(lo, hi, alpha=0.25, color=_IID_SHADE,
               label=f"In training distribution (n={lo}–{hi})")


def plot_single(curves, output_path, title=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor(_AXES_BG)
    all_ns = _draw_curves(ax, curves)
    ax.set_xlabel("Number of qubits (n)", fontsize=12)
    ax.set_ylabel("Oracle extraction accuracy", fontsize=12)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks(sorted(all_ns))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=11)
    if title:
        ax.set_title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[OracleAccPlot] saved: {output_path}")


def plot_dual(curves_full, curves_oracle, output_path,
              train_full_range=None, train_oracle_range=None, title=None):
    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(14, 5), sharey=True,
        gridspec_kw={"width_ratios": [1, 2.2]},   # right axis spans wider n range
    )

    # Training-range shading drawn first so the legend orders sensibly.
    _shade_training(ax_l, train_full_range)
    _shade_training(ax_r, train_oracle_range)

    ns_l = _draw_curves(ax_l, curves_full)
    ns_r = _draw_curves(ax_r, curves_oracle)

    for ax, name, ns in [
        (ax_l, "Full-circuit input",  ns_l),
        (ax_r, "Oracle-only input",   ns_r),
    ]:
        ax.set_facecolor(_AXES_BG)
        ax.set_xlabel("Number of qubits (n)", fontsize=12)
        ax.set_ylabel("Oracle extraction accuracy", fontsize=12)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xticks(sorted(ns))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=10)
        ax.set_title(name, fontsize=12)
        ax.tick_params(labelleft=True)   # force y-tick labels on both subplots

    if title:
        fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[OracleAccPlot] saved: {output_path}")


def _parse_range_pair(s):
    if s is None:
        return None
    lo_s, hi_s = s.split(",")
    return int(lo_s), int(hi_s)


def main():
    p = argparse.ArgumentParser(description="Plot oracle-extraction accuracy curves")
    p.add_argument("--dual", action="store_true",
                   help="Side-by-side subplots — uses --full and --oracle instead of --input")
    p.add_argument("--input", action="append", nargs="+", default=[], metavar=("PATH", "LABEL"),
                   help="(single-axes mode) Analysis JSON path, optional label. Repeatable.")
    p.add_argument("--full", action="append", nargs="+", default=[], metavar=("PATH", "LABEL"),
                   help="(dual mode) Full-circuit input analysis JSON, optional label. Repeatable.")
    p.add_argument("--oracle", action="append", nargs="+", default=[], metavar=("PATH", "LABEL"),
                   help="(dual mode) Oracle-only input analysis JSON, optional label. Repeatable.")
    p.add_argument("--train_full_range", default=None,
                   help="(dual mode) Training range for full-circuit input, format 'lo,hi'. "
                        "Shades the in-distribution zone. Example: '2,7' for Phase 1 GroverGPT+.")
    p.add_argument("--train_oracle_range", default=None,
                   help="(dual mode) Training range for oracle-only input, format 'lo,hi'. "
                        "Example: '2,10' for Phase 1 GroverGPT+.")
    p.add_argument("--output", required=True, help="Output PNG path")
    p.add_argument("--title", default=None, help="Optional figure title")
    args = p.parse_args()

    def _pack(items):
        return [(spec[1] if len(spec) > 1 else None, spec[0]) for spec in items]

    if args.dual:
        if not args.full or not args.oracle:
            p.error("--dual requires at least one --full and one --oracle")
        plot_dual(
            _pack(args.full), _pack(args.oracle), args.output,
            train_full_range=_parse_range_pair(args.train_full_range),
            train_oracle_range=_parse_range_pair(args.train_oracle_range),
            title=args.title,
        )
    else:
        if not args.input:
            p.error("--input is required when --dual is not set")
        plot_single(_pack(args.input), args.output, title=args.title)


if __name__ == "__main__":
    main()
