"""
Step 6 of the HMDA full experiment: SCIP bounds (global optimisation of each metric).

For every selected unit, metric and side, SCIP minimises or maximises the metric over the
joints consistent with the unit's tables (global_opt.global_bound). Each row records the best
value found ('value', attained by a feasible joint), SCIP's proven bound ('proven_bound') and
the solver status; when the two agree the bound is certified.

Output: <out>/bounds_scip.csv (param = time limit per bound, seconds)

Usage
  python hmda_scip.py
  python hmda_scip.py --time-limit 60 --runs D
"""
import os
import time

from hmda_common import append_rows, base_parser, done_keys, key_tuple, load_units
from global_opt import METRICS, global_bound  # noqa: E402


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--gap", type=float, default=1e-6)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "bounds_scip.csv")
    done = done_keys(path, ["param", "metric", "side"])
    for k, u in load_units(a):
        for metric in METRICS[k["run"]]:
            for side in ("lower", "upper"):
                if key_tuple(k, a.time_limit, metric, side) in done:
                    continue
                t0 = time.perf_counter()
                res = global_bound(u["r"], u["m"], k["run"], metric, side, a.time_limit, a.gap)
                append_rows(path, [dict(**k, method="SCIP", param=a.time_limit, metric=metric,
                                        side=side, value=res["value"],
                                        proven_bound=res["bound"], status=res["status"],
                                        seconds=time.perf_counter() - t0)])
                print(f"  {key_tuple(k)} {metric} {side}: {res['value']:.4f} "
                      f"(proven {res['bound']:.4f}, {res['status']}, "
                      f"{time.perf_counter() - t0:.1f}s)", flush=True)
    print(f"saved {path}")


if __name__ == "__main__":
    main()