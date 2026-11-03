"""
Figure of the HMDA bounds for one setting: for each metric, the grid interval (at the largest
feasible num) and, where Kallus et al. apply, the Kallus interval side by side; the true value
is a cross on the intervals (optionally the conditional-independence estimate as a dot).

One row per group pair; three panels per row because the metrics live on different scales:
differences, ratios, and mutual information.

Reads <out>/bounds_wide.csv (run hmda_collect.py first) and writes
<out>/figures/bounds_<decision>_<cut_source>_<proxy>_<version>_rep<rep>.png and .pdf

Usage
  python hmda_figure.py                                   # classifier, county, split margins
  python hmda_figure.py --decision lender --proxy both --version exact --show-ci
  python hmda_figure.py --truth internal                  # truth on the internal split only
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
COLORS = {"grid": "tab:orange", "Kallus": "black"}


def grid_bounds(row):
    """(lower, upper, num) of the grid at the largest num with both bounds."""
    nums = sorted({int(m.group(1)) for c in row.index
                   if (m := re.fullmatch(r"grid_(\d+)_lower", c))}, reverse=True)
    for n in nums:
        lo, hi = row.get(f"grid_{n}_lower"), row.get(f"grid_{n}_upper")
        if pd.notna(lo) and pd.notna(hi):
            return lo, hi, n
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
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

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
    nums_used = set()
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
                if pd.notna(row.get("Kallus_lower")):
                    intervals.append(("Kallus", row["Kallus_lower"], row["Kallus_upper"]))
                lo, hi, n = grid_bounds(row)
                if n is not None:
                    intervals.append(("grid", lo, hi))
                    nums_used.add(n)
                offsets = [0.0] if len(intervals) == 1 else [-0.13, 0.13]
                for (name, lo_, hi_), off in zip(intervals, offsets):
                    ax.plot([x + off, x + off], [lo_, hi_], color=COLORS[name], lw=2.2,
                            solid_capstyle="butt")
                    ax.plot([x + off - 0.07, x + off + 0.07], [lo_, lo_], color=COLORS[name],
                            lw=1.5)
                    ax.plot([x + off - 0.07, x + off + 0.07], [hi_, hi_], color=COLORS[name],
                            lw=1.5)
                    ax.plot(x + off, truth, "x", color="tab:red", ms=8, mew=2, zorder=5)
                if a.show_ci and pd.notna(row.get("CI_point")):
                    ax.plot(x + offsets[-1] + 0.2, row["CI_point"], "o", color="tab:blue",
                            ms=4, zorder=5)
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
    handles = [Line2D([], [], color=COLORS["Kallus"], lw=2.2, label="Kallus et al."),
               Line2D([], [], color=COLORS["grid"], lw=2.2, label="Feasible set (grid)"),
               Line2D([], [], color="tab:red", marker="x", ls="", ms=8, mew=2,
                      label="True value")]
    if a.show_ci:
        handles.append(Line2D([], [], color="tab:blue", marker="o", ls="", ms=4,
                              label="Conditional independence"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)
    stem = os.path.join(a.out, "figures",
                        f"bounds_{a.decision}_{a.cut_source}_{a.proxy}_{a.version}_rep{a.rep}")
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved {stem}.png and .pdf (grid num used: {sorted(nums_used)})")


if __name__ == "__main__":
    main()