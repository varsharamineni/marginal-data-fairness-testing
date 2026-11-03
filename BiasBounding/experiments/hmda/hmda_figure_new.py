"""
Figure of the HMDA bounds for one setting: for each metric, the intervals of the selected
methods side by side (by default Kallus et al. where they apply, and the grid at its largest
feasible num). Intervals are thin error bars with caps; the true value is a star on the
grid interval (optionally the conditional-independence estimate as an open circle).

Methods (--methods, shown in the given order): Kallus, grid, hit-and-run, vertex, SCIP.
For grid, hit-and-run, vertex and SCIP the largest parameter that was run is used (largest
num, most samples, largest vertex cap, longest SCIP time limit) unless --param fixes it, e.g.
--param grid=10 hit-and-run=10000.

The metrics live on different scales, so they are split by type: differences, ratios, and
mutual information. Each type gets its own panel, side by side in one figure (one row per
group pair); --separate writes one figure per type instead.

Reads <out>/bounds_wide.csv (run hmda_collect.py first) and writes
<out>/figures/bounds_<decision>_<cut_source>_<proxy>_<version>_rep<rep>.png and .pdf
(with --separate: one file per type, suffixed _differences, _ratios, _mi).

Usage
  python hmda_figure.py                                   # classifier, county, split margins
  python hmda_figure.py --decision lender --proxy both --version exact --show-ci
  python hmda_figure.py --truth internal                  # truth on the internal split only
  python hmda_figure.py --methods Kallus grid hit-and-run vertex SCIP
  python hmda_figure.py --methods grid SCIP --param grid=5
  python hmda_figure.py --separate                        # one figure per metric type
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

from hmda_common import OUT

PANELS = [  # (title, file suffix, metrics)
    ("Difference Metrics", "Difference Metrics", ["DD 1-0", "TPRD 1-0", "PPVD 1-0", "F1 diff 1-0",
                     "max DD gap"]),
    ("Ratio Metrics", "ratios", ["DI 1/0", "worst-case DI"]),
    ("Mutual Information", "mi", ["I(D;A)", "I(D;A|Y)"]),
]
LABEL = {"DD 1-0": "DD", "TPRD 1-0": "TPR\ndiff.", "PPVD 1-0": "PPV\ndiff.",
         "BA diff 1-0": "BA\ndiff.", "F1 diff 1-0": "F1\ndiff.", "max DD gap": "max DD\ngap",
         "DI 1/0": "DI", "worst-case DI": "worst-case\nDI", "I(D;A)": r"$I(\hat{Y};A)$",
         "I(D;A|Y)": r"$I(\hat{Y};A\mid Y)$"}
PAIR_TITLE = {"Black-White": "Black vs. White", "API-White": "API vs. White",
              "three": "Three groups\n(Black vs. White)"}
# Baseline in neutral grey, our methods in colorblind-safe Okabe-Ito colors; orange is kept
# for the true-value star so it never clashes with a method.
COLORS = {"Kallus": "#7A7A7A", "grid": "darkorange", "hit-and-run": "#CC79A7",
          "vertex": "#56B4E9", "SCIP": "#0072B2"}
NAMES = {"Kallus": "Kallus et al.", "grid": "Feasible set (grid)", "hit-and-run": "Hit-and-run",
         "vertex": "Vertex enumeration", "SCIP": "Global optimisation (SCIP)"}
INK = "#222222"          # text, marker edges
TRUTH = "#E69F00"        # true-value star
REF = "#BDBDBD"          # reference lines (0 and 1)
GRID = "#ECECEC"         # horizontal gridlines

STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 8,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 0,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def method_bounds(row, method, fixed=None):
    """(lower, upper, param) for a method: the given param, or the largest with both bounds."""
    if method == "Kallus":
        lo, hi = row.get("Kallus_lower"), row.get("Kallus_upper")
        return (lo, hi, "") if pd.notna(lo) and pd.notna(hi) else (np.nan, np.nan, None)
    pat = re.compile(rf"{re.escape(method)}_([0-9.]+)_lower")
    params = sorted({m.group(1) for c in row.index if (m := pat.fullmatch(c))},
                    key=float, reverse=True)
    if fixed is not None:
        params = [p for p in params if float(p) == float(fixed)]
    for p in params:
        lo, hi = row.get(f"{method}_{p}_lower"), row.get(f"{method}_{p}_upper")
        if pd.notna(lo) and pd.notna(hi):
            return lo, hi, p
    return np.nan, np.nan, None


def draw_panel(ax, d, title, metrics, a, fixed, used, step):
    """Error bars of the selected methods for each metric in this panel, plus truth (and CI)."""
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    if title == "Ratios":
        ax.axhline(1, color=REF, lw=0.8, zorder=1)
    else:
        ax.axhline(0, color=REF, lw=0.8, zorder=1)
    for x, metric in enumerate(metrics):
        if metric not in d.index:
            continue
        row = d.loc[metric]
        intervals = []
        for name in a.methods:
            lo_, hi_, p = method_bounds(row, name, fixed.get(name))
            if p is not None:
                intervals.append((name, lo_, hi_))
                used[name].add(p)
        offsets = (np.arange(len(intervals)) - (len(intervals) - 1) / 2) * step
        star_x = x  # the true value goes on the grid interval (centre if grid is absent)
        for (name, lo_, hi_), off in zip(intervals, offsets):
            if name == "grid":
                star_x = x + off
            mid = (lo_ + hi_) / 2
            ax.errorbar(x + off, mid, yerr=[[mid - lo_], [hi_ - mid]], fmt="none",
                        ecolor=COLORS[name], elinewidth=1.3, capsize=3, capthick=1.3,
                        zorder=3)
        truth = row[f"truth_{a.truth}"]
        if pd.notna(truth):
            ax.plot(star_x, truth, "*", ms=8.5, color=TRUTH, mec=INK, mew=0.5, zorder=5)
        if a.show_ci and pd.notna(row.get("CI_point")):
            right = offsets[-1] if len(offsets) else 0.0
            ax.plot(x + right + step * 0.9, row["CI_point"], "o", ms=3.5, mfc="white",
                    mec=INK, mew=0.8, zorder=5)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([LABEL[m] for m in metrics])
    ax.tick_params(axis="x", length=2.5)
    ax.set_xlim(-0.55, len(metrics) - 0.45)
    ax.margins(y=0.08)


def legend_handles(a, used):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COLORS[m], lw=1.3, marker="_", ms=6, mew=1.3, label=NAMES[m])
         for m in a.methods if used[m]]
    h.append(Line2D([], [], ls="", marker="*", ms=8.5, color=TRUTH, mec=INK, mew=0.5,
                    label="True value"))
    if a.show_ci:
        h.append(Line2D([], [], ls="", marker="o", ms=3.5, mfc="white", mec=INK, mew=0.8,
                        label="Conditional independence"))
    return h


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decision", default="model", choices=["model", "lender"])
    ap.add_argument("--cut-source", default="external", choices=["external", "full"])
    ap.add_argument("--proxy", default="county", choices=["county", "income", "both"])
    ap.add_argument("--version", default="split", choices=["split", "exact"])
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--settings", nargs="+", default=["Black-White", "API-White"])
    ap.add_argument("--truth", default="pooled", choices=["pooled", "internal"])
    ap.add_argument("--show-ci", action="store_true", help="add the CI point estimate")
    ap.add_argument("--methods", nargs="+", default=["Kallus", "grid"],
                    choices=list(COLORS), help="methods to show, in this order")
    ap.add_argument("--param", nargs="+", default=[],
                    help="fix a method's parameter, e.g. grid=10 hit-and-run=10000")
    ap.add_argument("--separate", action="store_true",
                    help="one figure per metric type (default: all types side by side)")
    ap.add_argument("--width", type=float, default=7.0,
                    help="figure width in inches (7.0 ~ two-column text width)")
    ap.add_argument("--metric-width", type=float, default=0.75,
                    help="inches per metric in the separate figures, so they share one scale")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    fixed = dict(p.split("=", 1) for p in a.param)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w = pd.read_csv(os.path.join(a.out, "bounds_wide.csv"))
    w = w[(w.decision == a.decision) & (w.cut_source == a.cut_source) & (w.proxy == a.proxy)
          & (w.version == a.version) & (w.rep == a.rep)]
    settings = [s for s in a.settings if s in set(w.setting)]
    if not settings:
        raise SystemExit("no rows for this selection; check the options and run hmda_collect.py")
    data = {s: w[w.setting == s].drop_duplicates("metric").set_index("metric") for s in settings}

    plt.rcParams.update(STYLE)
    used = {m: set() for m in a.methods}
    step = min(0.7 / max(len(a.methods), 1), 0.22)
    n = len(settings)
    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)
    tag = "" if a.methods == ["Kallus", "grid"] else "_" + "-".join(a.methods)
    stem = os.path.join(a.out, "figures",
                        f"bounds_{a.decision}_{a.cut_source}_{a.proxy}_{a.version}_rep{a.rep}{tag}")

    def finish(fig, path):
        h = legend_handles(a, used)
        ncol = len(h) if not a.separate or len(h) <= 3 else 2
        fig.legend(handles=h, loc="outside upper center", ncol=ncol, frameon=False,
                   handlelength=1.4, columnspacing=1.2)
        for ext in ("png", "pdf"):
            fig.savefig(f"{path}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"saved {path}.png and .pdf")

    if not a.separate:
        fig, axes = plt.subplots(n, len(PANELS), figsize=(a.width, 1.9 * n + 0.45),
                                 squeeze=False, sharey="col", layout="constrained",
                                 gridspec_kw=dict(width_ratios=[len(m) for *_, m in PANELS]))
        for r_, s in enumerate(settings):
            for c_, (title, _, metrics) in enumerate(PANELS):
                ax = axes[r_][c_]
                draw_panel(ax, data[s], title, metrics, a, fixed, used, step)
                if r_ < n - 1:
                    ax.set_xticklabels([])
                if c_ == 0:
                    ax.set_ylabel(PAIR_TITLE.get(s, s), fontweight="bold")
                if r_ == 0:
                    ax.set_title(title, loc="left", pad=4)
        finish(fig, stem)
    else:
        for title, suffix, metrics in PANELS:
            width = 0.9 + a.metric_width * len(metrics)
            fig, axes = plt.subplots(n, 1, figsize=(width, 1.9 * n + 0.6), squeeze=False,
                                     sharey=True, layout="constrained")
            for r_, s in enumerate(settings):
                ax = axes[r_][0]
                draw_panel(ax, data[s], title, metrics, a, fixed, used, step)
                if r_ < n - 1:
                    ax.set_xticklabels([])
                ax.set_ylabel(PAIR_TITLE.get(s, s), fontweight="bold")
            axes[0][0].set_title(title, loc="left", pad=4)
            finish(fig, f"{stem}_{suffix}")

    for m in a.methods:
        print(f"  {m}: parameter used {sorted(used[m]) if used[m] else 'no results'}")


if __name__ == "__main__":
    main()