"""Simultaneous achievability of fairness criteria over the feasible joint distributions.

Kallus bounds are separate intervals for each metric: both intervals can contain 0 without
any single joint having both metrics near 0. The joints method evaluates all metrics on the
same joint, so it can answer: is there a feasible joint under which the classifier
satisfies several criteria at once?

For a set S of criteria (each a disparity that should be 0):
    eps*(S) = min over feasible joints of max_{m in S} |m(joint)|
is the smallest tolerance at which all of S hold together for some feasible joint.
On a grid this is an upper bound on the exact eps* (a finer grid can only lower it).

Usage:  python scripts/simultaneous_fairness.py [--eps 0.05] [--out-dir outputs/simultaneous]
"""
import argparse
import itertools
import os
import sys

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fairness import methods  # noqa: E402
from metric_comparison import from_config, group_table, synthetic  # noqa: E402


# ------------------------------------------------------------------ criteria on T[n, a, yhat, y]
def criteria(T):
    """Disparities (group a minus group b) for every joint; each is 0 when the criterion holds."""
    with np.errstate(divide="ignore", invalid="ignore"):
        P = T.sum(axis=(2, 3))
        pos = T[:, :, 1, :].sum(2) / P                                  # P(Yhat=1 | A)
        tpr = T[:, :, 1, 1] / T[:, :, :, 1].sum(2)                      # P(Yhat=1 | Y=1, A)
        fpr = T[:, :, 1, 0] / T[:, :, :, 0].sum(2)                      # P(Yhat=1 | Y=0, A)
        ppv = T[:, :, 1, 1] / T[:, :, 1, :].sum(2)                      # P(Y=1 | Yhat=1, A)
        acc = (T[:, :, 0, 0] + T[:, :, 1, 1]) / P                       # P(Yhat=Y | A)
    d = lambda r: r[:, 0] - r[:, 1]                                     # noqa: E731
    return pd.DataFrame({"DD": d(pos), "TPRD": d(tpr), "FPRD": d(fpr), "PPV diff": d(ppv),
                         "ACC diff": d(acc)})


NAMES = {"DD": "demographic parity", "TPRD": "equal opportunity", "FPRD": "FPR parity",
         "PPV diff": "predictive parity", "ACC diff": "accuracy parity"}
SETS = {"equalized odds (TPRD, FPRD)": ["TPRD", "FPRD"],
        "equalized odds + predictive parity": ["TPRD", "FPRD", "PPV diff"],
        "demographic parity + equalized odds": ["DD", "TPRD", "FPRD"],
        "all five": list(NAMES)}


def eps_star(vals, subset):
    worst = vals[subset].abs().max(axis=1, skipna=False)               # undefined -> excluded
    return worst.min(), worst


def kallus_box(prim, aux, a_levels):
    b = {m: methods.kallus_bounds(prim, aux, a_levels, m) for m in ("DD", "TPRD", "TNRD")}
    b["FPRD"] = (-b["TNRD"][1], -b["TNRD"][0])                         # FPR = 1 - TNR
    return b


# ------------------------------------------------------------------ experiment
def analyse(name, prim, aux, a_levels, num, eps):
    joints = methods.feasible_joints(prim, aux, a_levels, method="grid", num=num)
    T = methods.joint_metrics(joints, lambda j: group_table(j, a_levels))
    vals = criteria(T)
    box = kallus_box(prim, aux, a_levels)
    print(f"\n{name}: {len(joints)} feasible joints")

    ms = list(NAMES)
    pair = pd.DataFrame(np.nan, index=ms, columns=ms)
    rows = []
    for m1, m2 in itertools.combinations_with_replacement(ms, 2):
        subset = [m1] if m1 == m2 else [m1, m2]
        e, worst = eps_star(vals, subset)
        pair.loc[m1, m2] = pair.loc[m2, m1] = e
        row = {"dataset": name, "criteria": " + ".join(subset), "eps_star": e,
               f"n_joints_within_{eps}": int((worst <= eps).sum()), "n_joints": len(vals)}
        if all(m in box for m in subset):                              # what Kallus alone says
            row["kallus_each_contains_0"] = all(box[m][0] <= 0 <= box[m][1] for m in subset)
        rows.append(row)
    for label, subset in SETS.items():
        e, worst = eps_star(vals, subset)
        rows.append({"dataset": name, "criteria": label, "eps_star": e,
                     f"n_joints_within_{eps}": int((worst <= eps).sum()), "n_joints": len(vals),
                     "kallus_each_contains_0": (all(box[m][0] <= 0 <= box[m][1] for m in subset)
                                                if all(m in box for m in subset) else np.nan)})
    return pd.DataFrame(rows), pair, vals, box


# ------------------------------------------------------------------ figures
BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, MUTED, GRID, ORANGE, ACCENT = "#1f1f1e", "#6b6a63", "#e4e3dc", "#eb6834", "#1baf7a"


def _style():
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False})


def plot_pairs(pairs, path):
    _style()
    cmap = LinearSegmentedColormap.from_list("seq", BLUE)
    vmax = max(p.to_numpy().max() for p in pairs.values())
    fig, axes = plt.subplots(1, len(pairs), figsize=(5.2 * len(pairs) + 1, 4.6), constrained_layout=True)
    for ax, (name, p) in zip(np.atleast_1d(axes), pairs.items()):
        im = ax.imshow(p.to_numpy(), cmap=cmap, vmin=0, vmax=vmax)
        for (i, j), v in np.ndenumerate(p.to_numpy()):
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.5 * vmax else INK)
        ax.set_xticks(range(len(p)), p.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(p)), p.index)
        ax.set_title(f"{name}", loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.8)
    cb.set_label("ε* = smallest tolerance at which both hold for some feasible joint")
    cb.outline.set_visible(False)
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


def plot_scatter(name, vals, box, eps, path):
    _style()
    pairs = [("DD", "TPRD"), ("TPRD", "FPRD"), ("TPRD", "PPV diff")]
    fig, axes = plt.subplots(1, len(pairs), figsize=(13, 4.3), constrained_layout=True)
    for ax, (mx, my) in zip(axes, pairs):
        v = vals[[mx, my]].dropna()
        ok = (v.abs() <= eps).all(axis=1)
        ax.scatter(v.loc[~ok, mx], v.loc[~ok, my], s=4, color=BLUE[3], alpha=0.35, linewidths=0,
                   label="feasible joint")
        ax.scatter(v.loc[ok, mx], v.loc[ok, my], s=10, color=ACCENT, linewidths=0,
                   label=f"both within ±{eps}")
        ax.add_patch(plt.Rectangle((-eps, -eps), 2 * eps, 2 * eps, fill=False, color=INK, lw=1.2))
        if mx in box and my in box:
            (x0, x1), (y0, y1) = box[mx], box[my]
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color=ORANGE,
                                       lw=1.5, ls="--", label="Kallus bounds (each separately)"))
        ax.axhline(0, color=GRID, lw=0.8, zorder=0)
        ax.axvline(0, color=GRID, lw=0.8, zorder=0)
        ax.set(xlabel=f"{mx} ({NAMES[mx]})", ylabel=f"{my} ({NAMES[my]})")
        ax.set_title(f"{int(ok.sum())} of {len(v)} joints satisfy both", loc="left")
    axes[0].legend(loc="upper left", fontsize=8, markerscale=2)
    fig.suptitle(f"{name}: pairs of disparities on the same feasible joint", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=0.05, help="tolerance for 'criterion holds'")
    ap.add_argument("--num-synthetic", type=int, default=3)
    ap.add_argument("--num-german", type=int, default=5)
    ap.add_argument("--out-dir", default="outputs/simultaneous")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tables, pairs = [], {}
    for name, (prim, aux, _, a_levels), num in [
            ("synthetic", synthetic(), args.num_synthetic),
            ("german", from_config("configs/german.yaml"), args.num_german)]:
        table, pair, vals, box = analyse(name, prim, aux, a_levels, num, args.eps)
        tables.append(table)
        pairs[name] = pair
        plot_scatter(name, vals, box, args.eps, os.path.join(args.out_dir, f"{name}_pairs.png"))
    table = pd.concat(tables)
    table.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    plot_pairs(pairs, os.path.join(args.out_dir, "eps_star.png"))
    pd.set_option("display.width", 250)
    print(table.round(4).to_string(index=False))
