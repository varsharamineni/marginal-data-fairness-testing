"""
Feasible regions of pairs of fairness metrics for one HMDA setting, from the grid method.

The joints consistent with the setting's internal and external tables are enumerated once
(grid_joint.solve_joint_distribution) and every metric is evaluated on every joint. The figure
is a grid of panels over all pairs of metrics:
  below the diagonal  the values of the two metrics reached together by the joints (the shading
                      is the number of joints in each cell, on a log scale); this is the joint
                      feasible region of the pair, which per-metric bounds do not show
  diagonal            the distribution of each metric across the joints; its range is the grid's
                      interval for that metric
  red cross / line    the true value (computed with the withheld race labels)
With internal variable DY, the decision-only metrics (DD, DI, ...) are computed from the same
joints by summing over Y, so all ten metrics are shown together.

Usage (run hmda_prepare.py first)
  python hmda_regions.py                                        # Black-White, county, split, DY
  python hmda_regions.py --setting API-White --proxy both --num 5
  python hmda_regions.py --metrics "DD 1-0" "DI 1/0" "I(D;A)" "TPRD 1-0"
Output: <out>/figures/regions_<decision>_<cut_source>_<setting>_<proxy>_<version>_<run>_num<num>_rep<rep>.png/.pdf
"""
import argparse
import contextlib
import io
import os
import pickle

import numpy as np

from hmda_common import OUT, PREPARED
from grid_joint import solve_joint_distribution  # noqa: E402
from synth import metrics  # noqa: E402

LABEL = {"DD 1-0": "DD", "DI 1/0": "DI", "worst-case DI": "worst-case DI",
         "max DD gap": "max DD gap", "I(D;A)": r"$I(\hat{Y};A)$", "TPRD 1-0": "TPR diff.",
         "PPVD 1-0": "PPV diff.", "BA diff 1-0": "BA diff.", "F1 diff 1-0": "F1 diff.",
         "I(D;A|Y)": r"$I(\hat{Y};A\mid Y)$"}
ORDER = ["DD 1-0", "DI 1/0", "worst-case DI", "max DD gap", "I(D;A)",
         "TPRD 1-0", "PPVD 1-0", "BA diff 1-0", "F1 diff 1-0", "I(D;A|Y)"]
PAIR = {"Black-White": "Black vs White", "API-White": "API vs White",
        "three": "three groups (pairwise: Black vs White)"}


def enumerate_joints(r, m, nums, max_joints, method):
    """Joints at the largest num in `nums` that stays under the cap: (J, num)."""
    K, I = r.shape
    G = m.shape[1]
    for num in sorted(nums, reverse=True):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                J, _ = solve_joint_distribution(
                    {(i, z): r[z, i] for z in range(K) for i in range(I)},
                    {(z, g): m[z, g] for z in range(K) for g in range(G)},
                    num=num, method=method, max_joints=max_joints, as_dicts=False)
            return J, num
        except SystemExit:
            continue
    raise SystemExit(f"no num in {nums} stays under {max_joints:,} joints")


def all_metrics(J, run):
    """{metric: values over the joints}; for DY also the decision-only metrics."""
    vals = dict(metrics(J, run))
    if run == "DY":                    # cells TP, FP, FN, TN -> D = 0 (FN + TN), D = 1 (TP + FP)
        JD = np.stack([J[:, :, 2] + J[:, :, 3], J[:, :, 0] + J[:, :, 1]], axis=2)
        vals.update(metrics(JD, "D"))
    return vals


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decision", default="model", choices=["model", "lender"])
    ap.add_argument("--cut-source", default="external", choices=["external", "full"])
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--setting", default="Black-White",
                    choices=["Black-White", "API-White", "three"])
    ap.add_argument("--proxy", default="county", choices=["county", "income", "both"])
    ap.add_argument("--version", default="split", choices=["split", "exact"])
    ap.add_argument("--run", default="DY", choices=["D", "DY"])
    ap.add_argument("--num", type=int, nargs="+", default=[2, 3, 4, 5, 10, 20],
                    help="grid resolution(s); the largest under --max-joints is used")
    ap.add_argument("--max-joints", type=int, default=2_000_000)
    ap.add_argument("--method", default="sequential")
    ap.add_argument("--metrics", nargs="+", default=None, help="subset of metrics, in order")
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--truth", default="pooled", choices=["pooled", "internal"])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if a.decision == "lender" and a.run == "DY":
        raise SystemExit("with decision = lender only --run D is defined")

    with open(PREPARED, "rb") as f:
        units = pickle.load(f)
    key = (a.decision, a.cut_source, a.rep, a.setting, a.proxy, a.run, a.version)
    if key not in units:
        raise SystemExit(f"unit {key} not prepared; run hmda_prepare.py --reps {a.rep}")
    u = units[key]
    truth = dict(u[f"truth_{a.truth}"])
    if a.run == "DY":                                   # decision-only truths from the D unit
        kd = key[:5] + ("D",) + key[6:]
        if kd in units:
            truth.update(units[kd][f"truth_{a.truth}"])

    J, num = enumerate_joints(u["r"], u["m"], a.num, a.max_joints, a.method)
    vals = all_metrics(J, a.run)
    names = [m_ for m_ in (a.metrics or ORDER) if m_ in vals]
    print(f"{len(J):,} joints at num={num}; metrics: {', '.join(names)}")
    for n_ in names:
        v = vals[n_][np.isfinite(vals[n_])]
        print(f"  {n_:14s} [{v.min():8.4f}, {v.max():8.4f}]   truth {truth.get(n_, np.nan):.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    k = len(names)
    fig, axes = plt.subplots(k, k, figsize=(1.75 * k + 1, 1.75 * k + 1), squeeze=False)
    ranges = {}
    for n_ in names:
        v = vals[n_][np.isfinite(vals[n_])]
        lo, hi = v.min(), v.max()
        pad = 0.04 * (hi - lo) if hi > lo else 0.01
        ranges[n_] = (lo - pad, hi + pad)
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            ax = axes[i][j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                v = vals[ni][np.isfinite(vals[ni])]
                ax.hist(v, bins=a.bins, range=ranges[ni], color="tab:orange", alpha=0.8)
                ax.axvline(truth.get(ni, np.nan), color="tab:red", lw=1.5)
                ax.set_yticks([])
                ax.set_xlim(ranges[ni])
            else:
                x, y = vals[nj], vals[ni]
                ok = np.isfinite(x) & np.isfinite(y)
                H, xe, ye = np.histogram2d(x[ok], y[ok], bins=a.bins,
                                           range=[ranges[nj], ranges[ni]])
                H = np.ma.masked_equal(H, 0)
                ax.pcolormesh(xe, ye, H.T, cmap="Oranges", norm=LogNorm(vmin=1, vmax=H.max()),
                              rasterized=True)
                ax.plot(truth.get(nj, np.nan), truth.get(ni, np.nan), "x", color="tab:red",
                        ms=6, mew=1.8)
                ax.set_xlim(ranges[nj])
                ax.set_ylim(ranges[ni])
            ax.tick_params(labelsize=6)
            if i == k - 1:
                ax.set_xlabel(LABEL.get(nj, nj), fontsize=8)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(LABEL.get(ni, ni), fontsize=8)
            elif j > 0:
                ax.set_yticklabels([])
    fig.suptitle(f"Feasible regions, {PAIR.get(a.setting, a.setting)}; common variable: "
                 f"{a.proxy}; {len(J):,} joints (num = {num})", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)
    stem = os.path.join(a.out, "figures",
                        f"regions_{a.decision}_{a.cut_source}_{a.setting}_{a.proxy}_{a.version}_"
                        f"{a.run}_num{num}_rep{a.rep}")
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved {stem}.png and .pdf")


if __name__ == "__main__":
    main()