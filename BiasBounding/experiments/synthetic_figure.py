"""
Single summary figure for the synthetic experiments: the classifier comparison (separate bounds
and bounds on the difference) and the cost of the grid (size of the feasible set and accuracy
of an interior bound), drawn from the CSVs written by synthetic_limit.py.

Usage
  python experiments/synthetic_limit.py --exp compare cost   # if the CSVs are stale
  python experiments/synthetic_figure.py
Outputs: <out>/figures/synthetic_summary.png/.pdf
"""
import argparse
import os

import pandas as pd

from fs_common import C_A, C_B, C_DIFF, INK, REF, STYLE, TRUTH
from synthetic_limit import interval, save, setup, star

K_COLS = {2: "#0072B2", 3: "#009E73", 4: "#E69F00", 6: "#CC79A7", 8: "#D55E00"}


def panels_compare(ax_sep, ax_diff, res):
    setup(ax_sep); ax_sep.axhline(0, color=REF, lw=0.8)
    setup(ax_diff); ax_diff.axhline(0, color=INK, lw=0.8, ls=":")
    for k, row in res.iterrows():
        interval(ax_sep, k - 0.14, row.A_lower, row.A_upper, C_A)
        interval(ax_sep, k + 0.14, row.B_lower, row.B_upper, C_B)
        star(ax_sep, k - 0.14, row.truth_A); star(ax_sep, k + 0.14, row.truth_B)
        interval(ax_diff, k, row.diff_lower, row.diff_upper, C_DIFF, lw=1.8, cap=4)
        star(ax_diff, k, row.truth_diff)
    for ax in (ax_sep, ax_diff):
        ax.set_xticks(range(len(res))); ax.set_xticklabels([f"{s:g}" for s in res.strength])
        ax.set_xlabel("Proxy strength $s$")
    ax_sep.set_ylabel("Demographic disparity")
    ax_sep.set_title("(a) DD of each rule", loc="left", pad=4)
    ax_diff.set_title("(b) DD(A) $-$ DD(B)", loc="left", pad=4)


def panels_cost(ax_size, ax_acc, res, max_joints):
    K = res[(res.kind == "K") & res.feasible]
    setup(ax_size); setup(ax_acc)
    for k, g in K.groupby("K"):
        ax_size.plot(g.num, g.n_joints, "o-", ms=2.5, lw=1, color=K_COLS[k], label=f"K = {k}")
        ax_acc.plot(g.num, g.shortfall.clip(lower=1e-6), "o-", ms=2.5, lw=1, color=K_COLS[k])
    ax_size.axhline(max_joints, color=INK, lw=0.8, ls="--")
    for ax in (ax_size, ax_acc):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Grid size (num)")
    ax_size.set_ylabel("Joints enumerated")
    ax_acc.set_ylabel(r"Shortfall in min $|$DD$|$")
    ax_size.set_title("(c) Size of the feasible set", loc="left", pad=4)
    ax_acc.set_title("(d) Interior bound accuracy", loc="left", pad=4)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-joints", type=int, default=2_000_000)
    ap.add_argument("--out", default="results/synthetic")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update(STYLE)
    compare = pd.read_csv(os.path.join(a.out, "synthetic_compare.csv"))
    cost = pd.read_csv(os.path.join(a.out, "synthetic_cost.csv"))
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.2), layout="constrained",
                             gridspec_kw=dict(width_ratios=[1.25, 1, 1, 1]))
    panels_compare(axes[0], axes[1], compare)
    panels_cost(axes[2], axes[3], cost, a.max_joints)
    h = [Line2D([], [], color=C_A, lw=1.3, label="Rule A"),
         Line2D([], [], color=C_B, lw=1.3, label="Rule B"),
         Line2D([], [], color=C_DIFF, lw=1.8, label="A $-$ B"),
         Line2D([], [], ls="", marker="*", ms=7, color=TRUTH, mec=INK, mew=0.5,
                label="True value")]
    h += [Line2D([], [], color=c, marker="o", ms=2.5, lw=1, label=f"$K$ = {k}")
          for k, c in K_COLS.items()]
    fig.legend(handles=h, loc="outside upper center", ncol=9, frameon=False, fontsize=6.5,
               handlelength=1.4, columnspacing=0.9)
    save(fig, a.out, "synthetic_summary")
    plt.close(fig)
    print(f"saved {a.out}/figures/synthetic_summary.png/.pdf")


if __name__ == "__main__":
    main()
