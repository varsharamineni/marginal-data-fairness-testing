"""
Figure of the HMDA bounds for one setting: for each metric, the intervals of the selected
methods side by side (by default Kallus et al. where they apply, and the grid at its largest
feasible num); the true value is a cross on the intervals (optionally the
conditional-independence estimate as a dot).

Methods (--methods, shown in the given order): Kallus, grid, hit-and-run, vertex, SCIP.
For grid, hit-and-run, vertex and SCIP the largest parameter that was run is used (largest
num, most samples, largest vertex cap, longest SCIP time limit) unless --param fixes it, e.g.
--param grid=10 hit-and-run=10000.

One row per group pair; three panels per row because the metrics live on different scales:
differences, ratios, and mutual information.

Reads <out>/bounds_wide.csv (run hmda_collect.py first) and writes
<out>/figures/bounds_<decision>_<cut_source>_<proxy>_<version>_rep<rep>.png and .pdf

Usage
  python hmda_figure.py                                   # classifier, county, split margins
  python hmda_figure.py --decision lender --proxy both --version exact --show-ci
  python hmda_figure.py --truth internal                  # truth on the internal split only
  python hmda_figure.py --methods Kallus grid hit-and-run vertex SCIP
  python hmda_figure.py --methods grid SCIP --param grid=5
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

from hmda_common import OUT

PANELS = [
    ("Differences", ["DD 1-0", "TPRD 1-0", "PPVD 1-0", "BA diff 1-0", "F1 diff 1-0",
                     "max DD gap"]),
    ("Ratios", ["DI 1/0", "worst-case DI"]),
    ("Mutual information (bits)", ["I(D;A)", "I(D;A|Y)"]),
]
LABEL = {"DD 1-0": "DD", "TPRD 1-0": "TPR\ndiff.", "PPVD 1-0": "PPV\ndiff.",
         "BA diff 1-0": "BA\ndiff.", "F1 diff 1-0": "F1\ndiff.", "max DD gap": "max DD\ngap",
         "DI 1/0": "DI", "worst-case DI": "worst-case\nDI", "I(D;A)": r"$I(\hat{Y};A)$",
         "I(D;A|Y)": r"$I(\hat{Y};A\mid Y)$"}
PAIR_TITLE = {"Black-White": "Black vs White", "API-White": "API vs White",
              "three": "Three groups (Black vs White)"}
COLORS = {"Kallus": "black", "grid": "tab:orange", "hit-and-run": "tab:blue",
          "vertex": "tab:green", "SCIP": "tab:purple"}
NAMES = {"Kallus": "Kallus et al.", "grid": "Feasible set (grid)", "hit-and-run": "Hit-and-run",
         "vertex": "Vertex enumeration", "SCIP": "Global optimisation (SCIP)"}


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
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    fixed = dict(p.split("=", 1) for p in a.param)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    w = pd.read_csv(os.path.join(a.out, "bounds_wide.csv"))
    w = w[(w.decision == a.decision) & (w.cut_source == a.cut_source) & (w.proxy == a.proxy)
          & (w.version == a.version) & (w.rep == a.rep)]
    settings = [s for s in a.settings if s in set(w.setting)]
    if not settings:
        raise SystemExit("no rows for this selection; check the options and run hmda_collect.py")

    widths = [len(m) for _, m in PANELS]
    fig, axes = plt.subplots(len(settings), len(PANELS), figsize=(13, 3.4 * len(settings)),
                             squeeze=False, gridspec_kw=dict(width_ratios=widths))
    used = {m: set() for m in a.methods}
    k = len(a.methods)
    step = min(0.8 / max(k, 1), 0.26)
    for r_, setting in enumerate(settings):
        d = w[w.setting == setting].drop_duplicates("metric").set_index("metric")
        for c_, (title, metrics) in enumerate(PANELS):
            ax = axes[r_][c_]
            for x, metric in enumerate(metrics):
                if metric not in d.index:
                    continue
                row = d.loc[metric]
                truth = row[f"truth_{a.truth}"]
                intervals = []
                for name in a.methods:
                    lo_, hi_, p = method_bounds(row, name, fixed.get(name))
                    if p is not None:
                        intervals.append((name, lo_, hi_))
                        used[name].add(p)
                offsets = (np.arange(len(intervals)) - (len(intervals) - 1) / 2) * step
                cap = min(0.07, step * 0.3)
                for (name, lo_, hi_), off in zip(intervals, offsets):
                    ax.plot([x + off, x + off], [lo_, hi_], color=COLORS[name], lw=2.2,
                            solid_capstyle="butt")
                    ax.plot([x + off - cap, x + off + cap], [lo_, lo_], color=COLORS[name],
                            lw=1.5)
                    ax.plot([x + off - cap, x + off + cap], [hi_, hi_], color=COLORS[name],
                            lw=1.5)
                    ax.plot(x + off, truth, "x", color="tab:red", ms=7, mew=2, zorder=5)
                if a.show_ci and pd.notna(row.get("CI_point")):
                    right = offsets[-1] if len(offsets) else 0.0
                    ax.plot(x + right + step * 0.8, row["CI_point"], "o", color="tab:cyan",
                            mec="k", mew=0.5, ms=5, zorder=5)
            ax.set_xticks(range(len(metrics)))
            ax.set_xticklabels([LABEL[m] for m in metrics], fontsize=8)
            ax.set_xlim(-0.6, len(metrics) - 0.4)
            if c_ == 0:
                ax.axhline(0, color="grey", lw=0.6, ls=":")
                ax.set_ylabel(PAIR_TITLE.get(setting, setting), fontsize=10)
            if c_ == 1:
                ax.axhline(1, color="grey", lw=0.6, ls=":")
            if r_ == 0:
                ax.set_title(title, fontsize=10)
    handles = [Line2D([], [], color=COLORS[m], lw=2.2, label=NAMES[m]) for m in a.methods
               if used[m]]
    handles.append(Line2D([], [], color="tab:red", marker="x", ls="", ms=7, mew=2,
                          label="True value"))
    if a.show_ci:
        handles.append(Line2D([], [], color="tab:cyan", marker="o", mec="k", mew=0.5, ls="",
                              ms=5, label="Conditional independence"))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4), fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08 if len(handles) > 4 else 0.05, 1, 1))
    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)
    tag = "" if a.methods == ["Kallus", "grid"] else "_" + "-".join(a.methods)
    stem = os.path.join(a.out, "figures",
                        f"bounds_{a.decision}_{a.cut_source}_{a.proxy}_{a.version}_rep{a.rep}{tag}")
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved {stem}.png and .pdf")
    for m in a.methods:
        print(f"  {m}: parameter used {sorted(used[m]) if used[m] else 'no results'}")


if __name__ == "__main__":
    main()