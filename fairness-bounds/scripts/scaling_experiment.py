"""How feasible_joints scales with the number of Z levels K.

Grid:   number of joints (grid points per Z level, multiplied over Z levels),
        runtime, and coverage of the Kallus bounds.
Random: runtime and coverage for different numbers of samples.

coverage = (max - min of the metric over the joints) / (kallus_upper - kallus_lower);
1 means the joints reach the sharp bounds.

Usage:  python scripts/scaling_experiment.py [--out-dir outputs/scaling] [--quick]
"""
import argparse
import os
import sys
import time

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fairness import data, methods  # noqa: E402

A = ["a", "b"]
MAX_BUILD = 20_000          # grids larger than this are counted but not built


def make_data(K, n=20_000, seed=0):
    cfg = dict(seed=seed, n_primary=n, n_auxiliary=n, n_z=K, p_z=None, p_a_given_z=None,
               y_intercept=0.0, y_coef_a=0.5, y_coef_z=1.0, x1_coef_y=1.5, x1_coef_a=0.5, x2_coef_y=1.0)
    prim, aux = data.generate_synthetic(cfg)
    prim["yhat"] = (prim["x1"] > 0.75).astype(int)
    prim.pop("a")
    return prim, aux


def evaluate(prim, aux, metrics, **kw):
    t = time.perf_counter()
    joints = methods.feasible_joints(prim, aux, A, max_joints=np.inf, **kw)
    row = {"n_joints": len(joints), "build_s": time.perf_counter() - t}
    t = time.perf_counter()
    for m in metrics:
        v = methods.joint_metrics(joints, lambda j: methods.joint_disparity(j, A, m))
        lo, hi = methods.kallus_bounds(prim, aux, A, m)
        row[f"coverage_{m}"] = (np.nanmax(v) - np.nanmin(v)) / (hi - lo)
    row["metrics_s"] = time.perf_counter() - t
    return row


GRID_SETTINGS = [(("yhat",), 3, ["DD"]), (("yhat",), 5, ["DD"]), (("yhat",), 9, ["DD"]),
                 (("yhat", "y"), 3, list(methods.METRICS)), (("yhat", "y"), 5, list(methods.METRICS))]


def run_grid(Ks):
    rows = []
    for K in Ks:
        prim, aux = make_data(K)
        for outcome, num, metrics in GRID_SETTINGS:
            row = {"K": K, "outcome": ",".join(outcome), "num": num,
                   "grid_size": methods.grid_size(prim, aux, A, outcome, num)}
            if row["grid_size"] <= MAX_BUILD:
                row |= evaluate(prim, aux, metrics, outcome=outcome, method="grid", num=num)
            rows.append(row)
            print(row, flush=True)
    return pd.DataFrame(rows)


def run_random(Ks, sample_sizes):
    rows = []
    for K in Ks:
        prim, aux = make_data(K)
        for n in sample_sizes:
            row = {"K": K, "n_samples": n}
            row |= evaluate(prim, aux, list(methods.METRICS), method="random", n_samples=n)
            rows.append(row)
            print(row, flush=True)
    return pd.DataFrame(rows)


def plot(grid, rand, path):
    series = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    ink, muted, grid_c = "#1f1f1e", "#6b6a63", "#e4e3dc"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "axes.titlecolor": ink, "xtick.color": muted, "ytick.color": muted,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": grid_c, "grid.linewidth": 0.6, "legend.frameon": False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    groups = list(grid.groupby(["outcome", "num"], sort=False))

    ax = axes[0]
    for i, ((outcome, num), g) in enumerate(groups):
        ax.plot(g["K"], g["grid_size"], marker="o", ms=5, lw=2, color=series[i],
                label=f"O=({outcome}), num={num}")
    ax.axhline(100_000, color=muted, lw=1, ls="--")
    ax.annotate("default max_joints", (grid["K"].max(), 100_000), xytext=(0, 4),
                textcoords="offset points", ha="right", va="bottom", color=muted, fontsize=8)
    ax.set(yscale="log", xlabel="Z levels K", ylabel="grid joints", title="Grid size grows exponentially in K")
    ax.legend(fontsize=8)

    ax = axes[1]
    for i, (_, g) in enumerate(groups):
        g = g.dropna(subset=["build_s"])
        ax.plot(g["n_joints"], g["build_s"] + g["metrics_s"], marker="o", ms=6, lw=0, color=series[i])
    ax.plot(rand["n_joints"], rand["build_s"] + rand["metrics_s"], marker="s", ms=6, lw=0,
            color=muted, label="random")
    ax.set(xscale="log", yscale="log", xlabel="joints built",
           ylabel="seconds (build + all metrics)", title="Runtime is linear in joints built")
    ax.legend(fontsize=8)

    ax = axes[2]
    for i, (n, g) in enumerate(rand.groupby("n_samples")):
        ax.plot(g["K"], g["coverage_DD"], marker="o", ms=5, lw=2, color=series[i], label=f"{n} samples")
    ax.axhline(1.0, color=muted, lw=1, ls="--")
    ax.set(xscale="log", ylim=(0, 1.05), xlabel="Z levels K",
           ylabel="DD range / Kallus width", title="Random sampling covers less as K grows")
    ax.legend(fontsize=8)

    fig.savefig(path, dpi=150)
    print(f"saved {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/scaling")
    ap.add_argument("--quick", action="store_true", help="fewer settings, for a fast check")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    grid_Ks = [1, 2, 3] if args.quick else [1, 2, 3, 4, 5, 6, 8, 10]
    rand_Ks = [1, 5] if args.quick else [1, 2, 3, 5, 10, 20, 50]
    sizes = [100] if args.quick else [100, 1000, 3000]

    grid = run_grid(grid_Ks)
    rand = run_random(rand_Ks, sizes)
    grid.to_csv(os.path.join(args.out_dir, "grid.csv"), index=False)
    rand.to_csv(os.path.join(args.out_dir, "random.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\nGrid\n", grid.round(3).to_string(index=False))
    print("\nRandom\n", rand.round(3).to_string(index=False))
    plot(grid, rand, os.path.join(args.out_dir, "scaling.png"))
