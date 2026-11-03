"""
Step 1 of the HMDA full experiment: feasible set (grid) bounds.

For every selected unit and every num, the joints consistent with the unit's primary and
auxiliary tables are enumerated once (grid_joint.solve_joint_distribution) and all metrics are
evaluated on that same set of joints; the bounds are their min and max. Once a num exceeds the
joint cap for a unit, larger nums are recorded as infeasible and skipped.

Output: <out>/bounds_grid.csv (param = num; n_joints; feasible)

Usage
  python hmda_grid.py                                   # all prepared units, nums 2 3 4 5 10 20
  python hmda_grid.py --nums 2 3 --settings Black-White --runs D
  python hmda_grid.py --method sweep                    # the sweep variant instead
"""
import contextlib
import io
import os
import time

import numpy as np


from hmda_common import (append_rows, base_parser, bounds_rows, done_keys, key_tuple,
                         load_units)
from grid_joint import solve_joint_distribution  # noqa: E402
from hit_and_run import bounds as sample_bounds  # noqa: E402
from synth import metrics  # noqa: E402


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--nums", type=int, nargs="+", default=[2, 3, 4, 5, 10, 20])
    ap.add_argument("--max-joints", type=int, default=2_000_000)
    ap.add_argument("--method", default="sequential", help="grid_joint method")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "bounds_grid.csv")
    done = done_keys(path, ["param", "grid_method"])
    for k, u in load_units(a):
        r, m = u["r"], u["m"]
        K, I = r.shape
        G = m.shape[1]
        too_big = False
        for num in sorted(a.nums):
            if key_tuple(k, num, a.method) in done:
                continue
            if too_big:
                append_rows(path, [dict(**k, method="grid", param=num, grid_method=a.method,
                                        metric="", side="", value=np.nan, seconds=0.0,
                                        n_joints=np.nan, feasible=False)])
                continue
            t0 = time.perf_counter()
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    J, _ = solve_joint_distribution(
                        {(i, z): r[z, i] for z in range(K) for i in range(I)},
                        {(z, g): m[z, g] for z in range(K) for g in range(G)},
                        num=num, method=a.method, max_joints=a.max_joints, as_dicts=False)
            except SystemExit:
                too_big = True
                append_rows(path, [dict(**k, method="grid", param=num, grid_method=a.method,
                                        metric="", side="", value=np.nan,
                                        seconds=time.perf_counter() - t0, n_joints=np.nan,
                                        feasible=False)])
                print(f"  {key_tuple(k)} num={num}: above {a.max_joints:,} joints, stopping")
                continue
            b = sample_bounds(J, lambda Q: metrics(Q, k["run"]))
            sec = time.perf_counter() - t0
            append_rows(path, bounds_rows(k, "grid", num, b, sec, grid_method=a.method,
                                          n_joints=len(J), feasible=True))
            print(f"  {key_tuple(k)} num={num}: {len(J):,} joints, {sec:.2f}s", flush=True)
    print(f"saved {path}")


if __name__ == "__main__":
    main()