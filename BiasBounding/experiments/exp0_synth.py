"""
Experiment 0 - validation of the grid method against Kallus et al. (2022) where both apply.

Question: on problems where Kallus' closed forms are sharp (binary protected attribute, no
extra assumptions), do the grid bounds stay inside the sharp interval and converge to it as
`num` grows, and what does that cost?

Design
  * Synthetic POPULATIONS (exact probabilities, no sampling), so any gap between the methods is
    due to the methods, not to estimation noise. The true disparity is known.
  * Each population: proxy Z with K levels, binary group A, binary label Y, binary decision D.
    `strength` controls how informative Z is about A (0 = useless, 1 = very informative).
  * Part A  demographic disparity, internal = D        (1 free entry per proxy level, many levels)
    Part B  TPR and PPV disparity, internal = (D, Y)   (3 free entries per proxy level, few levels)
  * References: Kallus closed form (sharp) and Kallus LP (should match the closed form).
  * Grid: both methods (sweep, sequential), num in a list; runs over max_joints are recorded
    as skipped, which shows where the grid stops scaling.

Layout: methods/ (grid_joint.py, kallus_bounds.py), experiments/ (this file),
results in experiment_results/exp0/.

Both codes get the same P(Z) here because primary and auxiliary come from one population
(on real data, align them: grid takes P(Z) from the auxiliary data, Kallus from the primary).

Outputs (in --out):
  results.csv        one row per (population, part, metric, reference/method, num)
  summary.csv        mean / max gap to the sharp bound per (part, metric, method, num)
  convergence.png    bounds vs num for the first population, sharp bounds as lines
  runtime.png        runtime and number of joints vs num

Usage
  python exp0_validation.py                       # defaults
  python exp0_validation.py --pops 20 --strength 0.8 --nums-dd 2 3 5 9 17 33
"""
import argparse
import contextlib
import io
import os
import time

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "methods"))

from grid_joint import solve_joint_distribution
from kallus_bounds import closed_form_bounds, lp_bounds

INTERNAL_DY = [(0, 0), (0, 1), (1, 0), (1, 1)]          # (decision, label)

# metric -> (target, cond) as functions of internal value (d, y) or d
METRICS = {
    "DD":   (lambda d, y: d == 1,            lambda d, y: True),
    "TPRD": (lambda d, y: d == 1 and y == 1, lambda d, y: y == 1),
    "PPVD": (lambda d, y: d == 1 and y == 1, lambda d, y: d == 1),
}


# ------------------------------------------------------------------------- population
def make_population(K, strength, seed):
    """
    Exact joint p[z, a, y, d]. P(A=1 | Z=z) is spread around 0.5 by `strength`; Y depends on
    A and Z; D depends on Y, A and Z, so all disparities are non-zero in general.
    """
    rng = np.random.default_rng(seed)
    pz = rng.dirichlet(np.full(K, 3.0))
    pa1 = np.clip(0.5 + strength * (rng.uniform(size=K) - 0.5), 0.02, 0.98)
    zeff = rng.normal(size=K)
    sig = lambda x: 1 / (1 + np.exp(-x))
    p = np.zeros((K, 2, 2, 2))
    for z in range(K):
        for a in range(2):
            pa = pa1[z] if a == 1 else 1 - pa1[z]
            py1 = sig(-0.3 + 0.6 * a + 0.8 * zeff[z])
            for y in range(2):
                py = py1 if y == 1 else 1 - py1
                pd1 = sig(-0.5 + 1.8 * y - 0.7 * a + 0.5 * zeff[z])
                for d in range(2):
                    p[z, a, y, d] = pz[z] * pa * py * (pd1 if d == 1 else 1 - pd1)
    return p


def split(p, internal):
    """Primary r[z, i] and auxiliary m[z, a] (Kallus format) plus dicts for the grid code."""
    K = p.shape[0]
    if internal == "D":
        I = [0, 1]
        r = p.sum((1, 2))                                          # [z, d]
    else:
        I = INTERNAL_DY
        r = np.stack([p[:, :, y, d].sum(1) for d, y in I], 1)      # [z, (d, y)]
    m = p.sum((2, 3))                                              # [z, a]
    p_primary = {(i, z): r[z, k] for z in range(K) for k, i in enumerate(I)}
    p_aux = {(z, a): m[z, a] for z in range(K) for a in range(2)}
    return I, r, m, p_primary, p_aux


def masks(I, metric):
    t, c = METRICS[metric]
    dy = [(i, None) if not isinstance(i, tuple) else i for i in I]
    return (np.array([bool(t(d, y)) for d, y in dy]),
            np.array([bool(c(d, y)) for d, y in dy]))


def true_disparity(p, metric):
    """mu_1 - mu_0 computed from the full joint."""
    t, c = METRICS[metric]
    mu = []
    for a in range(2):
        num = sum(p[:, a, y, d].sum() for d in range(2) for y in range(2) if t(d, y))
        den = sum(p[:, a, y, d].sum() for d in range(2) for y in range(2) if c(d, y))
        mu.append(num / den)
    return mu[1] - mu[0]


# ------------------------------------------------------------------------- grid evaluation
def grid_disparity(J, I_grid, metric):
    """mu_1 - mu_0 for every joint J [N, common, internal, external]."""
    target, cond = masks(I_grid, metric)
    theta = J.sum(1)                                   # [N, internal, external]
    T = theta[:, target, :].sum(1)                     # [N, external]
    X = theta[:, cond, :].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = T / X
    diff = mu[:, 1] - mu[:, 0]
    return diff[np.isfinite(diff)]


def run_grid(p_primary, p_aux, num, method, max_joints):
    """Returns (J, I, seconds) or (None, None, seconds) if the run exceeds max_joints."""
    t0 = time.perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()):      # silence the solver's prints
            J, (I, C, E) = solve_joint_distribution(p_primary, p_aux, num=num, method=method,
                                                    max_joints=max_joints, as_dicts=False)
        assert list(E) == [0, 1], E
        return J, I, time.perf_counter() - t0
    except SystemExit:
        return None, None, time.perf_counter() - t0


# ------------------------------------------------------------------------- experiment
def run_pops(part, pops, nums, methods, metrics, max_joints, use_lp):
    """
    pops: list of (name, p, info) with p[z, a, y, d] an exact joint (any K, binary a, y, d)
    and info a dict of extra columns for the results (e.g. proxy, groups).
    Part A uses internal = D, part B internal = (D, Y).
    """
    rows = []
    internal = "D" if part == "A" else "DY"
    for name, p, info in pops:
        K = p.shape[0]
        I, r, m, p_primary, p_aux = split(p, internal)
        for metric in metrics:
            target, cond = masks(I, metric)
            truth = true_disparity(p, metric)
            lo, hi = closed_form_bounds(r, m, target, cond)
            base = dict(pop=name, part=part, K=K, metric=metric, truth=truth, **info)
            rows.append(dict(base, method="kallus_closed_form", num=np.nan, lower=lo, upper=hi,
                             n_joints=np.nan, seconds=np.nan))
            if use_lp:
                t0 = time.perf_counter()
                df = lp_bounds(r, m, target, cond)
                rows.append(dict(base, method="kallus_lp", num=np.nan,
                                 lower=df.lower[0], upper=df.upper[0], n_joints=np.nan,
                                 seconds=time.perf_counter() - t0))
        for method in methods:
            for num in nums:
                J, I_grid, sec = run_grid(p_primary, p_aux, num, method, max_joints)
                for metric in metrics:
                    base = dict(pop=name, part=part, K=K, metric=metric,
                                truth=true_disparity(p, metric), method=f"grid_{method}",
                                num=num, seconds=sec, **info)
                    if J is None:
                        rows.append(dict(base, lower=np.nan, upper=np.nan, n_joints=np.nan,
                                         skipped=True))
                        continue
                    d = grid_disparity(J, I_grid, metric)
                    rows.append(dict(base, lower=d.min(), upper=d.max(), n_joints=len(J),
                                     skipped=False))
                status = "skipped" if J is None else f"{len(J):,} joints"
                print(f"  part {part} {str(name):24s} {method:10s} num={num:<3d} "
                      f"{status:>18s} {sec:7.2f}s")
    return rows


def run_part(part, pops, K, strength, nums, methods, metrics, max_joints, use_lp, seed0):
    """Synthetic populations, then run_pops."""
    populations = [(s, make_population(K, strength, seed0 + s), {}) for s in range(pops)]
    return run_pops(part, populations, nums, methods, metrics, max_joints, use_lp)


def add_gaps(res):
    """Gaps to the sharp (closed-form) bound; positive = grid interval is inside."""
    sharp = res[res.method == "kallus_closed_form"].set_index(["pop", "part", "metric"])
    key = list(zip(res["pop"], res["part"], res["metric"]))
    res["sharp_lower"] = [sharp.loc[k, "lower"] for k in key]
    res["sharp_upper"] = [sharp.loc[k, "upper"] for k in key]
    res["gap_lower"] = res.lower - res.sharp_lower          # >= 0 expected for the grid
    res["gap_upper"] = res.sharp_upper - res.upper          # >= 0 expected for the grid
    res["width"] = res.upper - res.lower
    res["covers_truth"] = (res.lower <= res.truth + 1e-9) & (res.truth <= res.upper + 1e-9)
    return res


def check(res, tol=1e-7):
    """Sanity checks that should hold by construction."""
    msgs = []
    g = res[res.method.str.startswith("grid") & res.lower.notna()]
    if (g.gap_lower < -tol).any() or (g.gap_upper < -tol).any():
        msgs.append("FAIL: some grid bound lies outside the sharp interval")
    lp = res[res.method == "kallus_lp"]
    if len(lp) and ((lp.gap_lower.abs() > 1e-5) | (lp.gap_upper.abs() > 1e-5)).any():
        msgs.append("WARN: Kallus LP differs from the closed form by more than 1e-5")
    if not res[res.method.str.startswith("kallus")].covers_truth.all():
        msgs.append("FAIL: a Kallus interval does not contain the true disparity")
    miss = g[g.covers_truth == False]
    if len(miss):      # expected for coarse grids: the grid is an inner approximation
        msgs.append(f"note: {len(miss)} grid intervals miss the truth (inner approximation), "
                    f"at num = {sorted(miss.num.unique().astype(int).tolist())}; "
                    "see the coverage column")
    return msgs or ["all checks passed: grid inside sharp interval, LP = closed form, "
                    "truth covered"]


def summarise(res):
    g = res[res.method.str.startswith("grid")].copy()
    g["covers_truth"] = g.covers_truth.where(g.lower.notna())      # NaN for skipped runs
    g["gap_max"] = g[["gap_lower", "gap_upper"]].max(axis=1)
    g["rel_width"] = g.width / (g.sharp_upper - g.sharp_lower)
    return (g.groupby(["part", "metric", "method", "num"])
             .agg(runs=("pop", "size"), skipped=("skipped", "sum"),
                  mean_gap=("gap_max", "mean"), max_gap=("gap_max", "max"),
                  mean_rel_width=("rel_width", "mean"), coverage=("covers_truth", "mean"), mean_joints=("n_joints", "mean"),
                  mean_seconds=("seconds", "mean"))
             .reset_index())


def plot(res, out, pop=None, suffix=""):
    """Convergence plot for one population (default: the first) and a runtime plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pop = res["pop"].iloc[0] if pop is None else pop
    r0 = res[res["pop"] == pop]
    panels = list(r0.groupby(["part", "metric"]).groups)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6), squeeze=False)
    for ax, (part, metric) in zip(axes[0], panels):
        d = r0[(r0.part == part) & (r0.metric == metric)]
        cf = d[d.method == "kallus_closed_form"].iloc[0]
        for b in ("lower", "upper"):
            ax.axhline(cf[b], color="k", lw=1, ls="--",
                       label="Kallus (sharp)" if b == "lower" else None)
        ax.axhline(cf.truth, color="tab:green", lw=1, label="truth")
        for method, mk in (("grid_sweep", "o"), ("grid_sequential", "s")):
            g = d[(d.method == method) & d.lower.notna()].sort_values("num")
            if len(g):
                ax.plot(g.num, g.lower, mk + "-", ms=4, label=method)
                ax.plot(g.num, g.upper, mk + "-", ms=4, color=ax.lines[-1].get_color())
        ax.set_xscale("log")
        ticks = sorted(d.num.dropna().unique())
        ax.set_xticks(ticks); ax.set_xticklabels([str(int(t)) for t in ticks])
        ax.minorticks_off()
        ax.set_title(f"{pop}\npart {part}: {metric} (K={int(cf.K)})" if isinstance(pop, str)
                     else f"Part {part}: {metric} (K={int(cf.K)})", fontsize=9)
        ax.set_xlabel("num (grid values per free entry)")
    axes[0][0].set_ylabel(r"bound on $\mu_1-\mu_0$")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, f"convergence{suffix}.png"), dpi=150)
    plt.close(fig)
    if suffix:                     # runtime plot only once
        return

    g = res[res.method.str.startswith("grid") & res.lower.notna()]
    g = g.groupby(["part", "method", "num"]).agg(sec=("seconds", "mean"),
                                                 joints=("n_joints", "mean")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for (part, method), d in g.groupby(["part", "method"]):
        axes[0].plot(d.num, d.sec, "o-", ms=4, label=f"part {part}, {method}")
        axes[1].plot(d.num, d.joints, "o-", ms=4, label=f"part {part}, {method}")
    for ax, lab in zip(axes, ("seconds", "number of joints")):
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("num"); ax.set_ylabel(lab)
    axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "runtime.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pops", type=int, default=10, help="number of random populations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strength", type=float, default=0.8, help="how informative Z is about A")
    ap.add_argument("--K-dd", type=int, default=4, help="proxy levels in part A")
    ap.add_argument("--K-rates", type=int, default=2, help="proxy levels in part B")
    ap.add_argument("--nums-dd", type=int, nargs="+", default=[2, 3, 5, 9, 17, 33, 65],
                    help="nested grids (2^k + 1) so each grid contains the previous one")
    ap.add_argument("--nums-rates", type=int, nargs="+", default=[2, 3, 5, 9, 17],
                    help="nested grids (2^k + 1)")
    ap.add_argument("--methods", nargs="+", default=["sweep", "sequential"])
    ap.add_argument("--max-joints", type=int, default=5_000_000)
    ap.add_argument("--no-lp", action="store_true", help="skip the Kallus LP reference")
    ap.add_argument("--skip-part", choices=["A", "B"])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent
                                         / "experiment_results" / "exp0"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = []
    if a.skip_part != "A":
        print(f"Part A: demographic disparity, K={a.K_dd}")
        rows += run_part("A", a.pops, a.K_dd, a.strength, a.nums_dd, a.methods, ["DD"],
                         a.max_joints, not a.no_lp, a.seed)
    if a.skip_part != "B":
        print(f"Part B: TPR and PPV disparity, K={a.K_rates}")
        rows += run_part("B", a.pops, a.K_rates, a.strength, a.nums_rates, a.methods,
                         ["TPRD", "PPVD"], a.max_joints, not a.no_lp, a.seed + 1000)

    res = add_gaps(pd.DataFrame(rows))
    res.to_csv(os.path.join(a.out, "results.csv"), index=False)
    summ = summarise(res)
    summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    plot(res, a.out)

    print("\n" + "\n".join(check(res)))
    with pd.option_context("display.width", 160, "display.max_rows", 200):
        print("\n" + summ.round(5).to_string(index=False))
    print(f"\nsaved results to {a.out}/")


if __name__ == "__main__":
    main()