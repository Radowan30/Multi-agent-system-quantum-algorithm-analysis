"""
Plot evaluation results for Phases 1 and 2, matching the style of [1].

Usage:
  python evaluation/plot_results.py --phase 1 --mode full
  python evaluation/plot_results.py --phase 1 --mode oracle
  python evaluation/plot_results.py --phase 2 --mode full

Plots saved to evaluation/phase-{N}/results/:
  sa_cf_{mode}.png   — SA (a) and CF (b) side-by-side with error bars   [1] Fig 4/5
  cr_srr.png         — Compression Ratio (a) and SRR (b) side-by-side    [1] Supp Fig 6
  ret.png            — Relative Execution Time, log y-axis               [1] Fig 6
"""

import argparse
import json
import os
import sys

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

PHASE_RESULTS_DIR = {
    1: os.path.join(_EVAL_DIR, "phase-1", "results"),
    2: os.path.join(_EVAL_DIR, "phase-2", "results"),
}

PHASE_MODEL_LABEL = {
    1: "GroverGPT+",
    2: "LLaMA 3.1 8B (Phase 2)",
}

N_RANGES = {
    1: {"full": (2, 9),  "oracle": (2, 20)},
    2: {"full": (2, 19), "oracle": (2, 20)},
}

# Colors matching the paper
_PURPLE = "#7B52A6"
_ORANGE = "#D4820A"
_BLUE   = "#2C72B0"

# Axes background colour (very light lavender, matching paper style)
_AXES_BG = "#F5F3FF"

# ---------------------------------------------------------------------------
# Load / extract helpers
# ---------------------------------------------------------------------------

def load_results(phase: int, mode: str, n_min: int, n_max: int,
                 circuit_source: str = "strict", results_dir: str = None) -> dict:
    fname = f"results_{mode}_{n_min}_{n_max}_{circuit_source}.json"
    path  = os.path.join(results_dir or PHASE_RESULTS_DIR[phase], fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Results not found: {path}")
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def _series(results: dict, key: str):
    """Return (ns, means, stds) for a [mean, std] metric key."""
    ns, means, stds = [], [], []
    for n in sorted(results):
        val = results[n].get(key)
        if val and val[0] is not None:
            ns.append(n)
            means.append(val[0])
            stds.append(val[1])
    return np.array(ns), np.array(means), np.array(stds)


def _fmt_value(v: float) -> str:
    """Compact value label: scientific for very large/small, else 1-2 decimals."""
    if v >= 1e3 or (0 < v < 0.01):
        return f"{v:.2e}"
    return f"{v:.1f}" if v >= 10 else f"{v:.2f}"


# ── Broken-axis helpers ───────────────────────────────────────────────────────

def _cluster_clipped_bands(values, ratio_threshold: float = 1.5):
    """Group clipped values into bands. Two values land in the same band when
    the ratio of their magnitudes is below `ratio_threshold` (1.5 by default).
    Returns a list of (band_min, band_max) tuples sorted by min ascending.

    For the Phase 2 Gradient RET clipped set {25.0, 25.4, 42.5, 54.4, 86.9, 97.8}
    at ratio_threshold=1.5, the clustering gives three bands:
      (25.0, 25.4) — ratio between bottom and top of band = 1.016
      (42.5, 54.4) — ratio = 1.28
      (86.9, 97.8) — ratio = 1.13
    """
    if not values:
        return []
    sorted_vals = sorted(values)
    bands = [[sorted_vals[0]]]
    for v in sorted_vals[1:]:
        if v / bands[-1][-1] < ratio_threshold:
            bands[-1].append(v)
        else:
            bands.append([v])
    return [(min(b), max(b)) for b in bands]


def _draw_break_marks(upper_ax, lower_ax, d: float = 0.012, lw: float = 1.0):
    """Draw the conventional zigzag break-marks between two adjacent axes
    (upper axes sits above lower axes in the figure). Two short diagonal
    line-segments at each panel-edge corner, drawn in axes coordinates."""
    kwargs = dict(color="black", clip_on=False, lw=lw)
    # Bottom edge of upper axes — two diagonals
    upper_ax.plot([-d, +d], [-d, +d], transform=upper_ax.transAxes, **kwargs)
    upper_ax.plot([1 - d, 1 + d], [-d, +d], transform=upper_ax.transAxes, **kwargs)
    # Top edge of lower axes — two diagonals
    lower_ax.plot([-d, +d], [1 - d, 1 + d], transform=lower_ax.transAxes, **kwargs)
    lower_ax.plot([1 - d, 1 + d], [1 - d, 1 + d], transform=lower_ax.transAxes, **kwargs)


def _band_padding(band_min: float, band_max: float) -> float:
    """Vertical padding for a band. Small relative pad for normal ranges; a
    fixed minimum for single-value bands (band_max == band_min)."""
    if band_max == band_min:
        return max(0.05 * band_min, 0.5)
    return (band_max - band_min) * 0.20


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def _apply_paper_style(ax):
    """Apply background colour and grid matching the paper's figure style."""
    ax.set_facecolor(_AXES_BG)
    ax.grid(True, color="#D8D0F0", linewidth=0.6, zorder=0)
    ax.tick_params(labelsize=9)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


def _label_subplot(ax, letter: str):
    """Add 'a)' / 'b)' label in the top-left corner, matching [1]."""
    ax.text(
        0.03, 0.97, f"{letter})",
        transform=ax.transAxes,
        fontsize=11, fontweight="bold",
        va="top", ha="left",
    )


# ---------------------------------------------------------------------------
# SA + CF — side-by-side subplots (matching [1] Fig 4 / Fig 5)
# ---------------------------------------------------------------------------

def plot_sa_cf(results: dict, mode: str, phase: int, output_dir: str,
               circuit_source: str = "strict", cf_key: str = "cf") -> None:
    """Plot SA + CF subplots side-by-side.

    `cf_key` selects which CF series to plot from the per-n results JSON:
      "cf"        — raw CF (paper-faithful, can exceed 1.0).
      "cf_renorm" — over-sum-renormalised CF (bounded [0, 1]).
    When `cf_key == "cf_renorm"`, the output filename gets a "_renorm" suffix
    so the two versions can coexist in the same results directory.

    When either series has values above the y-axis ceiling (1.5), the
    affected column is rendered with a broken y-axis: a thin top panel
    per band of clipped values, separated by conventional zigzag break-marks.
    The main panel keeps the paper-spec y-range [-0.05, 1.5] for visual
    comparability across plots.
    """
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    ns_sa, sa_means, sa_stds = _series(results, "sa")
    ns_cf, cf_means, cf_stds = _series(results, cf_key)
    is_renorm    = cf_key == "cf_renorm"
    cf_label     = "Classical Fidelity (renormalized)" if is_renorm else "Classical Fidelity"
    fname_suffix = "_renorm" if is_renorm else ""

    y_max = 1.5
    sa_bands = _cluster_clipped_bands([v for v in sa_means if v > y_max])
    cf_bands = _cluster_clipped_bands([v for v in cf_means if v > y_max])

    # Slight extra height per side that has bands, to give room for the band
    # panels without compressing the main panel.
    extra = max(len(sa_bands), len(cf_bands)) * 0.25
    fig = plt.figure(figsize=(8.5, 3.8 + extra))
    outer = GridSpec(1, 2, figure=fig, wspace=0.3,
                     left=0.07, right=0.97, top=0.94, bottom=0.14)

    _draw_sa_cf_column(
        fig, outer[0, 0],
        ns=ns_sa, means=sa_means, stds=sa_stds,
        ylabel="SA", legend_label="Searching Accuracy",
        color=_PURPLE, bands=sa_bands, y_max=y_max,
        subplot_letter="a",
    )
    _draw_sa_cf_column(
        fig, outer[0, 1],
        ns=ns_cf, means=cf_means, stds=cf_stds,
        ylabel="CF", legend_label=cf_label,
        color=_ORANGE, bands=cf_bands, y_max=y_max,
        subplot_letter="b",
    )

    out = _save(fig, output_dir, f"sa_cf_{mode}_{circuit_source}{fname_suffix}.png")
    print(f"Saved: {out}")


def _draw_sa_cf_column(fig, subplot_spec, ns, means, stds, ylabel: str,
                       legend_label: str, color: str, bands, y_max: float,
                       subplot_letter: str) -> None:
    """Draw one side of the SA/CF figure. If `bands` is empty, a single axes
    is created in `subplot_spec` with the paper-spec [-0.05, 1.5] y-range.
    If bands exist, a broken-axis layout is built inside `subplot_spec`:
    thin top panels per band of clipped values, conventional zigzag breaks
    between adjacent panels, and the main panel keeping the paper-spec range."""
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    if not bands:
        ax = fig.add_subplot(subplot_spec)
        _plot_metric_errorbar(ax, ns, means, stds, color, legend_label)
        ax.set_xlabel("Number of Qubits", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(-0.05, y_max)
        ax.set_xticks(_even_ticks(ns))
        ax.legend(fontsize=8, loc="upper right")
        _apply_paper_style(ax)
        _label_subplot(ax, subplot_letter)
        return

    # Broken-axis layout: bands stacked above main.
    bands_top_to_bottom = sorted(bands, reverse=True, key=lambda b: b[0])
    n_panels = 1 + len(bands_top_to_bottom)
    height_ratios = [0.45] * len(bands_top_to_bottom) + [4.0]
    inner = GridSpecFromSubplotSpec(
        n_panels, 1, subplot_spec=subplot_spec,
        height_ratios=height_ratios, hspace=0.14,
    )
    # sharex via the main panel so all panels keep the same x-range
    # (otherwise NaN-masking on band panels can auto-shrink their x-extent).
    ax_main = fig.add_subplot(inner[-1])
    band_axes = [fig.add_subplot(inner[i], sharex=ax_main)
                 for i in range(n_panels - 1)]
    axes = band_axes + [ax_main]

    # Per-panel y-ranges (main last in axes order).
    panel_ranges = [(b_min - _band_padding(b_min, b_max),
                     b_max + _band_padding(b_min, b_max))
                    for (b_min, b_max) in bands_top_to_bottom]
    panel_ranges.append((-0.05, y_max))

    # Plot data on every panel, NaN-masked outside the panel's y-range.
    for ax, (y_lo, y_hi) in zip(axes, panel_ranges):
        masked = [v if y_lo <= v <= y_hi else float("nan") for v in means]
        _plot_metric_errorbar(
            ax, ns, masked, stds, color,
            legend_label=(legend_label if ax is ax_main else None),
        )

    # Main panel — paper-spec styling.
    ax_main.set_ylim(-0.05, y_max)
    ax_main.set_xlabel("Number of Qubits", fontsize=9)
    ax_main.set_ylabel(ylabel, fontsize=9)
    ax_main.set_xticks(_even_ticks(ns))
    ax_main.legend(fontsize=8, loc="lower left")
    _apply_paper_style(ax_main)
    ax_main.spines["top"].set_visible(False)

    # Band panels — show data ticks for the actual clipped values.
    for ax, (y_lo, y_hi) in zip(band_axes, panel_ranges[:-1]):
        ax.set_ylim(y_lo, y_hi)
        in_band = sorted(set(v for v in means if y_lo <= v <= y_hi))
        ax.set_yticks(in_band)
        ax.set_yticklabels([_fmt_value(v) for v in in_band], fontsize=8)
        # Hide bottom tick marks AND labels. Do NOT call set_xticklabels([])
        # here — it wipes the shared x-axis labels for the whole sharex group.
        ax.tick_params(axis="x", bottom=False, top=False,
                       labelbottom=False, labeltop=False)
        _apply_paper_style(ax)
        ax.spines["bottom"].set_visible(False)
        if ax is not band_axes[0]:
            ax.spines["top"].set_visible(False)

    # Conventional break marks between adjacent panels.
    for upper, lower in zip(axes, axes[1:]):
        _draw_break_marks(upper, lower)

    # Subplot label on the topmost panel so the letter sits in the top-left
    # corner of the column (matching the single-axes case).
    _label_subplot(band_axes[0], subplot_letter)


def _plot_metric_errorbar(ax, ns, means, stds, color: str, legend_label):
    """The shared SA/CF errorbar style (circle markers, solid line, capped)."""
    ax.errorbar(
        ns, means, yerr=stds,
        fmt="o-", color=color, capsize=4, markersize=4,
        linewidth=1.4, capthick=1.0, elinewidth=1.0,
        label=legend_label,
    )


# ---------------------------------------------------------------------------
# CR + SRR — side-by-side subplots (matching [1] Supp Fig 6)
# ---------------------------------------------------------------------------

def plot_cr_srr(results: dict, phase: int, output_dir: str,
                circuit_source: str = "strict") -> None:
    ns_cr,  cr_means,  _  = _series(results, "cr")
    ns_srr, srr_means, _2 = _series(results, "srr")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.5, 3.8))

    # — (a) Compression Ratio —
    ax_a.plot(ns_cr, cr_means, color=_PURPLE, linewidth=1.8, label="Compression Ratio")
    ax_a.set_xlabel("Number of Qubits", fontsize=9)
    ax_a.set_ylabel("Compression Ratio", fontsize=9)
    ax_a.set_xticks(_even_ticks(ns_cr))
    ax_a.legend(fontsize=8, loc="lower right")
    _apply_paper_style(ax_a)
    _label_subplot(ax_a, "a")

    # — (b) Sequence Reduction Ratio —
    ax_b.plot(ns_srr, srr_means, color=_ORANGE, linewidth=1.8, label="Reduction Ratio")
    ax_b.set_xlabel("Number of Qubits", fontsize=9)
    ax_b.set_ylabel("Sequence Reduction Ratio", fontsize=9)
    ax_b.set_xticks(_even_ticks(ns_srr))
    ax_b.legend(fontsize=8, loc="lower right")
    _apply_paper_style(ax_b)
    _label_subplot(ax_b, "b")

    fig.tight_layout()
    out = _save(fig, output_dir, f"cr_srr_{circuit_source}.png")
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Relative Execution Time — log y-axis (matching [1] Fig 6)
# ---------------------------------------------------------------------------

def plot_ret(results: dict, phase: int, output_dir: str,
             circuit_source: str = "strict") -> None:
    """Relative-Execution-Time plot. Main panel preserves the paper's log y-axis
    spanning 10^0..10^1. RET values that exceed the y_max are auto-clustered
    into bands (factor-1.5 grouping) and each band is drawn as a thin separate
    panel stacked above the main panel, with conventional zigzag break-marks
    between adjacent panels. Each panel only renders data points that fall
    inside its y-range (other points are NaN-masked so the dashed line is
    broken cleanly at every cross-band jump). The total figure height stays
    close to the single-panel version — bands are deliberately thin so the
    main panel keeps roughly its original visual size."""
    ns, rets, errs = [], [], []
    for n in sorted(results):
        ret = results[n].get("ret")
        if ret is not None:
            ns.append(n)
            rets.append(ret)
            errs.append(results[n].get("ret_std", 0.0))

    if not ns:
        print("No RET data in results — skipping ret plot.")
        return

    model_label = PHASE_MODEL_LABEL.get(phase, f"Phase {phase}")
    y_max_main = 10.0
    bands = _cluster_clipped_bands([r for r in rets if r > y_max_main])

    # ── Single-panel fallback when no clipped values ──────────────────────
    if not bands:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        _draw_ret_series(ax, ns, rets, errs, model_label, plot_label=True)
        _style_ret_main(ax, ns, y_max_main)
        fig.tight_layout()
        out = _save(fig, output_dir, f"ret_{circuit_source}.png")
        print(f"Saved: {out}")
        return

    # ── Broken-axis layout: bands (thin) stacked above main (tall) ────────
    bands_top_to_bottom = sorted(bands, reverse=True, key=lambda b: b[0])  # highest first
    n_panels = 1 + len(bands_top_to_bottom)
    # Thin bands + tall main. height_ratio 0.45:4.0 gives bands ~10% the
    # height of main — enough to comfortably show a band's data points
    # without dominating the figure.
    height_ratios = [0.45] * len(bands_top_to_bottom) + [4.0]
    fig = plt.figure(figsize=(6, 4.8))
    # hspace 0.14 gives a comfortable visual gap between the main panel's
    # top edge (10^1 on the log axis) and the lowest band's first y-tick —
    # tighter values made the lowest band feel cramped against main.
    gs = fig.add_gridspec(n_panels, 1, height_ratios=height_ratios, hspace=0.14)
    # sharex via the main panel so every panel keeps the SAME x-range —
    # otherwise matplotlib auto-scales each panel to its own non-NaN data
    # extent, and the main panel (which masks high-RET points to NaN) gets
    # truncated to the in-range x-extent (e.g. 2..16 instead of 2..19).
    ax_main = fig.add_subplot(gs[-1])
    band_axes = [fig.add_subplot(gs[i], sharex=ax_main)
                 for i in range(n_panels - 1)]
    axes = band_axes + [ax_main]

    # Plot the data on every panel, NaN-masked so each panel only shows the
    # points inside its y-range (clean line breaks at cross-band jumps).
    panel_ranges = [(b_min - _band_padding(b_min, b_max),
                     b_max + _band_padding(b_min, b_max))
                    for (b_min, b_max) in bands_top_to_bottom]
    panel_ranges.append((1.0, y_max_main))  # main last (in axes order)

    for ax, (y_lo, y_hi) in zip(axes, panel_ranges):
        rets_masked = [r if y_lo <= r <= y_hi else float("nan") for r in rets]
        ax.errorbar(
            ns, rets_masked, yerr=errs,
            fmt="s--", color=_BLUE, markersize=5, linewidth=1.4,
            capsize=3, elinewidth=1.0, capthick=1.0, ecolor=_BLUE,
            label=model_label if ax is ax_main else None,
        )

    # Style each axes appropriately. Force the x-range on every panel to span
    # the full n range so band panels don't lose their data markers when their
    # data x-extent happens to land inside the main's in-range x-extent.
    _style_ret_main(ax_main, ns, y_max_main)
    ax_main.set_xlim(ns[0] - 0.5, ns[-1] + 0.5)

    for ax, (y_lo, y_hi) in zip(band_axes, panel_ranges[:-1]):
        ax.set_ylim(y_lo, y_hi)
        # Tick at only the band's min and max in-range values. Intermediate
        # values get squeezed together inside the thin band panel and overlap;
        # min/max alone read cleanly.
        in_band = sorted(set(r for r in rets if y_lo <= r <= y_hi))
        if len(in_band) >= 2:
            in_band = [in_band[0], in_band[-1]]
        ax.set_yticks(in_band)
        ax.set_yticklabels([_fmt_value(v) for v in in_band], fontsize=8)
        # Hide bottom tick marks AND labels on band panels. Do NOT call
        # set_xticklabels([]) here — with sharex, that wipes the shared
        # x-axis labels for the whole group (including the main panel).
        ax.tick_params(axis="x", bottom=False, top=False,
                       labelbottom=False, labeltop=False)
        _apply_paper_style(ax)
        # Hide the spines that face the panel break.
        ax.spines["bottom"].set_visible(False)
        if ax is not band_axes[0]:
            ax.spines["top"].set_visible(False)

    # Main axes: hide top spine so the break-mark to its upper neighbour reads cleanly.
    ax_main.spines["top"].set_visible(False)

    # Draw conventional break-marks between every pair of adjacent panels.
    for upper, lower in zip(axes, axes[1:]):
        _draw_break_marks(upper, lower)

    # Legend in the main panel.
    ax_main.legend(fontsize=8, loc="upper left")

    # Shared y-axis label spanning the full vertical extent (set on main; placed
    # via figure text since per-panel ylabels would create duplicates).
    ax_main.set_ylabel("")
    fig.text(0.02, 0.5, "Relative Execution Time\n(normalized to n=2)",
             ha="left", va="center", rotation="vertical", fontsize=9)

    fig.subplots_adjust(top=0.97, bottom=0.11, left=0.13, right=0.96)
    out = _save(fig, output_dir, f"ret_{circuit_source}.png")
    print(f"Saved: {out}")


def _draw_ret_series(ax, ns, rets, errs, model_label, plot_label: bool):
    """Helper: the standard RET errorbar call (dashed line, square markers)."""
    ax.errorbar(
        ns, rets, yerr=errs,
        fmt="s--", color=_BLUE, markersize=5, linewidth=1.4,
        capsize=3, elinewidth=1.0, capthick=1.0, ecolor=_BLUE,
        label=model_label if plot_label else None,
    )


def _style_ret_main(ax, ns, y_max: float):
    """Apply the paper-spec styling to the RET main panel: log scale, 1..y_max,
    even-tick x-axis, decade-only y-ticks, paper style + legend handled outside."""
    ax.set_yscale("log")
    ax.set_ylim(1, y_max)
    ax.set_xlabel("Number of Qubits", fontsize=9)
    ax.set_ylabel("Relative Execution Time\n(normalized to n=2)", fontsize=9)
    ax.set_xticks(ns)
    ax.set_xticklabels(ns, fontsize=8)
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([1, y_max]))
    ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterMathtext())
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    _apply_paper_style(ax)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _even_ticks(ns: np.ndarray):
    """Return even-numbered tick positions (matching paper x-axis style)."""
    if len(ns) == 0:
        return []
    lo, hi = int(ns[0]), int(ns[-1])
    return [n for n in range(lo, hi + 1) if n % 2 == 0 or lo == hi]


def _save(fig, output_dir: str, fname: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot Phase 1/2 evaluation results")
    parser.add_argument("--phase",          type=int, choices=[1, 2], required=True)
    parser.add_argument("--mode",           choices=["full", "oracle"], required=True)
    parser.add_argument("--circuit_source", choices=["strict", "paper"], default="strict")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Directory to read results from and write plots to "
                             "(default: the phase's results/ directory)")
    parser.add_argument(
        "--cf_key", choices=["cf", "cf_renorm", "both"], default="cf",
        help="Which CF series to plot from the per-n results JSON: 'cf' (raw, "
             "paper-faithful, default), 'cf_renorm' (over-sum renormalized), or "
             "'both' (generates two SA+CF figures, with '_renorm' suffix on the "
             "renormalized one).",
    )
    args = parser.parse_args()

    n_range = N_RANGES[args.phase].get(args.mode)
    if n_range is None:
        print(f"[Skip] Phase {args.phase} does not include mode='{args.mode}'")
        return

    n_min, n_max = n_range
    output_dir = args.results_dir or PHASE_RESULTS_DIR[args.phase]

    try:
        results = load_results(args.phase, args.mode, n_min, n_max,
                               args.circuit_source, args.results_dir)
    except FileNotFoundError as e:
        print(f"[Error] {e}")
        sys.exit(1)

    cf_keys = ["cf", "cf_renorm"] if args.cf_key == "both" else [args.cf_key]
    for cfk in cf_keys:
        plot_sa_cf(results, args.mode, args.phase, output_dir, args.circuit_source, cf_key=cfk)

    if args.mode == "full":
        plot_cr_srr(results, args.phase, output_dir, args.circuit_source)
        plot_ret(results, args.phase, output_dir, args.circuit_source)


if __name__ == "__main__":
    main()
