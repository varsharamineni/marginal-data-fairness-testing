"""
Compare all methods on one synthetic population (from exp_synthetic.py):

  truth            metric computed from the full population
  CI               conditional-independence point estimate      (methods/ci_baseline.py)
  Kallus           sharp LP bounds, linear / linear-fractional   (methods/kallus_bounds.py)
  grid             feasible set approach                         (methods/grid_joint.py)
  hit-and-run      min / max over uniform random joints          (methods/hit_and_run.py)
  vertex           min / max over vertices (all, or a sample)    (methods/vertex_sampling.py)
  SCIP             certified global optimum per bound            (methods/global_opt.py)
                   shown as value, with '*' if the proven dual bound differs by more than 1e-4

Every method except CI gives an interval; the grid, hit-and-run and vertex intervals are inner
approximations (every value in them is attainable), Kallus' are sharp where available.

Usage
  python compare_baselines.py                               # base case
  python compare_baselines.py --groups 3 --levels 3 --run DY --num 3 --n-hr 50000
"""
import argparse
import contextlib
import io
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

try:
    from global_opt import global_bounds  # noqa: E402
except ImportError:                       # PySCIPOpt not installed
    global_bounds = None

from synth import joint_table, kallus_refs, make_population, metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", type=int, default=2)
    ap.add_argument("--share", type=float, default=0.3)
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run", choices=["D", "DY"], default="D")
    ap.add_argument("--num", type=int, default=9, help="grid resolution")
    ap.add_argument("--method", default="sequential", help="grid method")
    ap.add_argument("--n-hr", type=int, default=20_000, help="hit-and-run samples")
    ap.add_argument("--n-vertex", type=int, default=200_000,
                    help="vertex samples if full enumeration exceeds --vertex-cap")
    ap.add_argument("--vertex-cap", type=int, default=2_000_000)
    ap.add_argument("--no-global", action="store_true", help="skip SCIP")
    ap.add_argument("--time-limit", type=float, default=60.0, help="SCIP seconds per bound")
    a = ap.parse_args()

    p = make_population(a.groups, a.share, a.levels, a.strength, a.seed)
    q = joint_table(p, a.run)
    r, m = q.sum(2), q.sum(1)
    K, I, G = q.shape
    fn = lambda Q: metrics(Q, a.run)
    truth = {k: float(v[0]) for k, v in fn(q[None]).items()}
    times, res = {}, {}

    t0 = time.perf_counter()
    ci = {k: float(v[0]) for k, v in fn(ci_joint(r, m)).items()}
    times["CI"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    res["Kallus"] = kallus_refs(r, m, a.run, G, max_groups=3, n_grid=21)
    times["Kallus"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        J, _ = solve_joint_distribution({(i, z): r[z, i] for z in range(K) for i in range(I)},
                                        {(z, g): m[z, g] for z in range(K) for g in range(G)},
                                        num=a.num, method=a.method, max_joints=5_000_000,
                                        as_dicts=False)
    res["grid"] = sample_bounds(J, fn)
    times["grid"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    res["hit-and-run"] = sample_bounds(hit_and_run_joints(r, m, n=a.n_hr, seed=a.seed), fn)
    times["hit-and-run"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    res["vertex"], info = vertex_bounds(r, m, fn, n=a.n_vertex, cap=a.vertex_cap, seed=a.seed)
    times["vertex"] = time.perf_counter() - t0

    if global_bounds is not None and not a.no_global:
        t0 = time.perf_counter()
        gb = global_bounds(r, m, a.run, time_limit=a.time_limit)
        flag = lambda d: "*" if not abs(d["value"] - d["bound"]) <= 1e-4 else ""
        for k, v in gb.items():
            for side in ("lower", "upper"):
                if str(v[side]["status"]).startswith("error") or flag(v[side]):
                    print(f"  SCIP {k} {side}: status {v[side]['status']}, value "
                          f"{v[side]['value']:.4f}, proven bound {v[side]['bound']:.4f}")
        res["SCIP"] = {k: (v["lower"]["value"], v["upper"]["value"],
                           flag(v["lower"]), flag(v["upper"])) for k, v in gb.items()}
        times["SCIP"] = time.perf_counter() - t0

    def fmt(iv):
        if iv is None:
            return ""
        s = f"[{iv[0]:7.3f}, {iv[1]:7.3f}]"
        return s if len(iv) == 2 else f"[{iv[0]:7.3f}{iv[2]}, {iv[1]:7.3f}{iv[3]}]"
    rows = []
    for k in truth:
        rows.append(dict(metric=k, truth=round(truth[k], 3), CI=round(ci[k], 3),
                         **{name: fmt(res[name].get(k)) for name in res}))
    print(f"population: groups={a.groups} share={a.share} levels={a.levels} "
          f"strength={a.strength} seed={a.seed}; internal = {a.run}")
    print(f"grid: {a.method}, num={a.num}, {len(J):,} joints | hit-and-run: {a.n_hr:,} samples | "
          f"vertex: {'all' if info['full_enumeration'] else 'sampled'} "
          f"{info['evaluated']:,} of {info['combinations']:,} combinations "
          f"(vertices per level {info['vertices_per_level']})")
    with pd.option_context("display.width", 200):
        print(pd.DataFrame(rows).to_string(index=False))
    print("seconds: " + ", ".join(f"{k} {v:.2f}" for k, v in times.items()))


if __name__ == "__main__":
    main()