"""
All methods across systematically varied synthetic scenarios.

Scenarios (as in exp_synthetic.py): exact populations p(z, a, y, d), one factor varied at a time
from the base case (strength 1, minority share 0.3, 2 groups, 3 proxy levels):
  strength   0, 0.5, 1, 2, 4        informativeness of the proxy about the group
  share      0.05, 0.15, 0.3, 0.5   total minority share
  groups     2, 3, 4                number of protected groups
  levels     1, 2, 3, 5, 8          number of proxy levels
For each population, both internal variables: D (5 metrics) and (D, Y) (5 metrics); metric
definitions and bound classes are those of exp_synthetic.py.

Methods
  CI           conditional-independence point estimate          methods/ci_baseline.py
  Kallus       sharp LP bounds (DD; TPRD/PPVD up to 3 groups)   methods/kallus_bounds.py
  grid         feasible set approach at the largest feasible num methods/grid_joint.py
  hit-and-run  min / max over uniform random joints             methods/hit_and_run.py
  vertex       min / max over vertex combinations               methods/vertex_sampling.py
  SCIP         certified global optimum per bound               methods/global_opt.py

Reference for each bound (lower / upper of a metric)
  exact          Kallus' sharp bound, or SCIP's value when SCIP proves it optimal
                 (|value - proven bound| <= 1e-5)
  best attained  otherwise: the most extreme value attained by any method (all are feasible
                 joints, so this is an inner bound); SCIP's proven bound is kept as 'ref_outer',
                 so the true bound lies between ref and ref_outer
Gap of an interval method = how far its bound lies inside the reference (>= 0).

Outputs (experiment_results/baselines/)
  results.csv        one row per population x internal x metric x side x method; 'level' is
                     the value of the varied factor, 'value' the method's bound (appended
                     after every population, so an interrupted run resumes where it stopped)
  summary_class.csv  gap by method and bound class
  summary_factor.csv gap by factor value and method
  runtime.csv        seconds per method
  ci_error.csv       conditional-independence error |CI - truth| relative to the bound width
  fig_gap_by_class.png, fig_gap_vs_factor.png, fig_runtime.png

Usage
  python exp_baselines.py --quick                     # small test (a few minutes)
  python exp_baselines.py                             # full run (can take hours with SCIP)
  python exp_baselines.py --methods grid vertex hit-and-run --seeds 3   # without SCIP
  python exp_baselines.py --time-limit 30 --factors groups levels
"""
import argparse
import contextlib
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "methods"))
from ci_baseline import ci_joint  # noqa: E402
from grid_joint import solve_joint_distribution  # noqa: E402
from hit_and_run import bounds as sample_bounds, hit_and_run_joints  # noqa: E402
from vertex_sampling import vertex_bounds  # noqa: E402

from synth import (BASE, CLASS, CLASS_ORDER, SWEEPS, joint_table,  # noqa: E402
                           kallus_refs, make_population, metrics)

try:
    from global_opt import global_bounds  # noqa: E402
except ImportError as e:
    print(f"SCIP unavailable ({e}); it will be skipped")
    global_bounds = None

INTERVAL_METHODS = ["Kallus", "grid", "hit-and-run", "vertex", "SCIP"]
CERT_TOL = 1e-5


# ------------------------------------------------------------------------- one population
def run_methods(r, m, run, a):
    """{method: {metric: (lower, upper)}}, SCIP proven bounds, CI values, seconds per method."""
    K, I = r.shape
    G = m.shape[1]
    fn = lambda Q: metrics(Q, run)
    res, sec, scip_bound = {}, {}, {}

    t0 = time.perf_counter()
    ci = {k: float(v[0]) for k, v in fn(ci_joint(r, m)).items()}
    sec["CI"] = time.perf_counter() - t0

    if "Kallus" in a.methods:
        t0 = time.perf_counter()
        res["Kallus"] = kallus_refs(r, m, run, G, a.kallus_max_groups, 21)
        sec["Kallus"] = time.perf_counter() - t0

    if "grid" in a.methods:
        t0 = time.perf_counter()
        for num in sorted(a.nums, reverse=True):                 # largest feasible num
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    J, _ = solve_joint_distribution(
                        {(i, z): r[z, i] for z in range(K) for i in range(I)},
                        {(z, g): m[z, g] for z in range(K) for g in range(G)},
                        num=num, method=a.grid_method, max_joints=a.max_joints, as_dicts=False)
                res["grid"] = sample_bounds(J, fn)
                res["grid_num"] = num
                break
            except SystemExit:
                continue
        sec["grid"] = time.perf_counter() - t0

    if "hit-and-run" in a.methods:
        t0 = time.perf_counter()
        res["hit-and-run"] = sample_bounds(hit_and_run_joints(r, m, n=a.n_hr, seed=0), fn)
        sec["hit-and-run"] = time.perf_counter() - t0

    if "vertex" in a.methods:
        t0 = time.perf_counter()
        res["vertex"], info = vertex_bounds(r, m, fn, n=a.n_vertex, cap=a.vertex_cap)
        res["vertex_full"] = info["full_enumeration"]
        sec["vertex"] = time.perf_counter() - t0

    if "SCIP" in a.methods and global_bounds is not None:
        t0 = time.perf_counter()
        gb = global_bounds(r, m, run, time_limit=a.time_limit)
        res["SCIP"] = {k: (v["lower"]["value"], v["upper"]["value"]) for k, v in gb.items()}
        scip_bound = {k: (v["lower"]["bound"], v["upper"]["bound"]) for k, v in gb.items()}
        sec["SCIP"] = time.perf_counter() - t0
    return res, scip_bound, ci, sec


def population_rows(cfg, seed, run, a):
    p = make_population(cfg["groups"], cfg["share"], cfg["levels"], cfg["strength"], seed)
    q = joint_table(p, run)
    r, m = q.sum(2), q.sum(1)
    truth = {k: float(v[0]) for k, v in metrics(q[None], run).items()}
    res, scip_bound, ci, sec = run_methods(r, m, run, a)
    rows = []
    for metric in truth:
        for j, side in enumerate(("lower", "upper")):
            pick = min if side == "lower" else max
            vals = {meth: res[meth][metric][j] for meth in INTERVAL_METHODS
                    if meth in res and metric in res[meth] and np.isfinite(res[meth][metric][j])}
            # reference: exact if Kallus has it or SCIP certifies it, else best attained
            sb = scip_bound.get(metric, (np.nan, np.nan))[j]
            exact = None
            if "Kallus" in vals:
                exact = vals["Kallus"]
            elif "SCIP" in vals and np.isfinite(sb) and abs(vals["SCIP"] - sb) <= CERT_TOL:
                exact = vals["SCIP"]
            if exact is not None:
                ref, ref_type = exact, "exact"
            elif vals:
                ref, ref_type = pick(vals.values()), "best attained"
            else:
                ref, ref_type = np.nan, "none"
            for meth, v in vals.items():
                gap = (v - ref) if side == "lower" else (ref - v)
                rows.append(dict(**cfg, seed=seed, run=run, metric=metric, side=side,
                                 cls=CLASS[metric][j], method=meth, value=v, ref=ref,
                                 ref_type=ref_type, ref_outer=sb, gap=gap, truth=truth[metric],
                                 seconds=sec.get(meth, np.nan),
                                 grid_num=res.get("grid_num", np.nan) if meth == "grid" else np.nan,
                                 vertex_full=res.get("vertex_full") if meth == "vertex" else None))
        rows.append(dict(**cfg, seed=seed, run=run, metric=metric, side="point", cls="",
                         method="CI", value=ci[metric], ref=np.nan, ref_type="", ref_outer=np.nan,
                         gap=np.nan, truth=truth[metric], seconds=sec["CI"]))
    return rows


# ------------------------------------------------------------------------- summaries and plots
def summarise(df, out):
    iv = df[df.method != "CI"].copy()
    iv["exact_hit"] = iv.gap.abs() <= 1e-6
    by_class = (iv.groupby(["run", "cls", "method"])
                  .agg(bounds=("gap", "size"), mean_gap=("gap", "mean"), max_gap=("gap", "max"),
                       share_exact=("exact_hit", "mean"),
                       share_ref_exact=("ref_type", lambda s: (s == "exact").mean()))
                  .reset_index())
    by_class.to_csv(os.path.join(out, "summary_class.csv"), index=False)
    by_factor = (iv.groupby(["factor", "level", "method"])
                   .agg(mean_gap=("gap", "mean"), max_gap=("gap", "max"),
                        share_exact=("exact_hit", "mean")).reset_index())
    by_factor.to_csv(os.path.join(out, "summary_factor.csv"), index=False)
    rt = (df.drop_duplicates(["factor", "level", "seed", "run", "method"])
            .groupby(["run", "groups", "method"]).seconds.mean().unstack("method"))
    rt.to_csv(os.path.join(out, "runtime.csv"))

    # conditional independence: error relative to the sharp / best-known interval width
    width = (df[df.method != "CI"].groupby(["factor", "level", "seed", "run", "metric", "side"])
               .ref.first().unstack("side"))
    width = (width["upper"] - width["lower"]).rename("width")
    ci = df[df.method == "CI"].set_index(["factor", "level", "seed", "run", "metric"])
    ci = ci.join(width)
    ci["abs_error"] = (ci.value - ci.truth).abs()
    ci["rel_error"] = ci.abs_error / ci.width.replace(0, np.nan)
    ci_sum = ci.reset_index().groupby(["factor", "level", "metric"])[["abs_error", "rel_error"]].mean()
    ci_sum.to_csv(os.path.join(out, "ci_error.csv"))
    return by_class, by_factor, rt


def plot(df, by_class, by_factor, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = [mm for mm in ["Kallus", "grid", "vertex", "hit-and-run", "SCIP"]
               if mm in set(df.method)]
    colors = dict(zip(["Kallus", "grid", "vertex", "hit-and-run", "SCIP"],
                      ["k", "tab:orange", "tab:green", "tab:blue", "tab:purple"]))

    # fig 1: mean gap by bound class and method
    classes = [c for c in CLASS_ORDER if c in set(by_class.cls)]
    agg = by_class.groupby(["cls", "method"]).apply(
        lambda d: np.average(d.mean_gap, weights=d.bounds)).unstack("method")
    fig, ax = plt.subplots(figsize=(10, 4))
    w = 0.8 / len(methods)
    for k, mm in enumerate(methods):
        if mm in agg:
            y = [agg.loc[c, mm] if c in agg.index else np.nan for c in classes]
            ax.bar(np.arange(len(classes)) + k * w, np.maximum(y, 1e-7), w, label=mm,
                   color=colors[mm])
    ax.set_xticks(np.arange(len(classes)) + w * (len(methods) - 1) / 2)
    ax.set_xticklabels(classes, fontsize=8); ax.set_yscale("log")
    ax.set_ylabel("mean gap to reference (floor 1e-7)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_gap_by_class.png"), dpi=150)
    plt.close(fig)

    # fig 2: gap vs each factor
    factors = list(dict.fromkeys(by_factor.factor))
    fig, axes = plt.subplots(1, len(factors), figsize=(4.2 * len(factors), 3.6), squeeze=False,
                             sharey=True)
    for ax, f in zip(axes[0], factors):
        d = by_factor[by_factor.factor == f]
        for mm in methods:
            g = d[d.method == mm].sort_values("level")
            if len(g):
                ax.plot(g.level, np.maximum(g.mean_gap, 1e-7), "o-", ms=4, color=colors[mm],
                        label=mm)
        ax.set_yscale("log"); ax.set_xlabel(f)
    axes[0][0].set_ylabel("mean gap (floor 1e-7)"); axes[0][0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_gap_vs_factor.png"), dpi=150)
    plt.close(fig)

    # fig 3: runtime
    rt = (df.drop_duplicates(["factor", "level", "seed", "run", "method"])
            .groupby(["run", "method"]).seconds.mean().unstack("run"))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    runs = list(rt.columns)
    for k, run in enumerate(runs):
        ax.bar(np.arange(len(rt)) + k * 0.4, rt[run].fillna(0).clip(lower=1e-4), 0.4,
               label=f"internal {run}")
    ax.set_xticks(np.arange(len(rt)) + 0.2); ax.set_xticklabels(rt.index, fontsize=8)
    ax.set_yscale("log"); ax.set_ylabel("seconds per population"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_runtime.png"), dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--factors", nargs="+", default=list(SWEEPS), choices=list(SWEEPS))
    ap.add_argument("--runs", nargs="+", default=["D", "DY"], choices=["D", "DY"])
    ap.add_argument("--methods", nargs="+", default=["Kallus", "grid", "hit-and-run", "vertex",
                                                      "SCIP"])
    ap.add_argument("--grid-method", default="sequential")
    ap.add_argument("--nums", type=int, nargs="+", default=[2, 3, 5, 9, 17, 33])
    ap.add_argument("--max-joints", type=int, default=2_000_000)
    ap.add_argument("--n-hr", type=int, default=5_000, help="hit-and-run samples")
    ap.add_argument("--n-vertex", type=int, default=100_000)
    ap.add_argument("--vertex-cap", type=int, default=500_000)
    ap.add_argument("--kallus-max-groups", type=int, default=3)
    ap.add_argument("--time-limit", type=float, default=10.0, help="SCIP seconds per bound")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="ignore existing results.csv")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent
                                         / "experiment_results" / "baselines"))
    a = ap.parse_args()
    sweeps = {f: SWEEPS[f] for f in a.factors}
    if a.quick:
        a.seeds, a.n_hr, a.time_limit = 1, 2000, 5.0
        a.nums = [2, 3, 5, 9]
        sweeps = {"strength": [0.0, 1.0, 4.0], "groups": [2, 3], "levels": [1, 3]}
        a.out = a.out + "_quick"
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "results.csv")
    done = set()
    if os.path.exists(path) and not a.fresh:
        old = pd.read_csv(path)
        done = set(zip(old.factor, old.level.astype(float), old.seed, old.run))
        print(f"resuming: {len(done)} population runs already in {path}")
    elif os.path.exists(path):
        os.remove(path)

    t_start = time.perf_counter()
    for factor, values in sweeps.items():
        for value in values:
            cfg = dict(BASE, **{factor: value})
            for seed in range(a.seeds):
                for run in a.runs:
                    if (factor, float(value), seed, run) in done:
                        continue
                    t0 = time.perf_counter()
                    rows = population_rows(cfg, seed, run, a)
                    pd.DataFrame([dict(r_, factor=factor, level=value) for r_ in rows]).to_csv(
                        path, mode="a", header=not os.path.exists(path), index=False)
                    print(f"  {factor}={value} seed={seed} internal={run}: "
                          f"{time.perf_counter() - t0:.1f}s "
                          f"(total {time.perf_counter() - t_start:.0f}s)", flush=True)

    df = pd.read_csv(path)
    by_class, by_factor, rt = summarise(df, a.out)
    plot(df, by_class, by_factor, a.out)
    neg = df[(df.method != "CI") & (df.ref_type == "exact") & (df.gap < -1e-5)]
    print(f"\nbounds beyond an exact reference (should be none): {len(neg)}")
    if len(neg):
        print(neg[["factor", "level", "run", "metric", "side", "method", "value", "ref"]]
              .head(10).to_string(index=False))
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        t = by_class.pivot_table(index=["run", "cls"], columns="method", values="mean_gap")
        print("\nmean gap to the reference, by bound class:\n" + t.to_string(float_format="%.4f"))
        print("\nmean seconds per population:\n" + rt.round(2).to_string())
    print(f"\nsaved to {a.out}/")


if __name__ == "__main__":
    main()