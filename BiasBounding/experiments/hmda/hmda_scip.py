"""
Step 6 of the HMDA full experiment: SCIP bounds (global optimisation of each metric).

For every selected unit, metric and side, SCIP minimises or maximises the metric over the
joints consistent with the unit's tables (global_opt.global_bound). Each row records the best
value found ('value', attained by a feasible joint), SCIP's proven bound ('proven_bound') and
the solver status; when the two agree the bound is certified.

Each solve runs in its own process, so a crash inside SCIP's C library (e.g. "double free or
corruption") only ends that solve: it is recorded with status 'crashed' and the script moves
on. A solve that does not return within --hard-timeout seconds is stopped and recorded as
'killed'. Crashed or killed bounds are saved like any other, so rerunning skips them; delete
their rows from bounds_scip.csv to retry them.

Output: <out>/bounds_scip.csv (param = time limit per bound, seconds)

Usage
  python hmda_scip.py
  python hmda_scip.py --time-limit 60 --runs D
"""
import multiprocessing as mp
import os
import time

import numpy as np

from hmda_common import append_rows, base_parser, done_keys, key_tuple, load_units
from global_opt import METRICS, global_bound  # noqa: E402


def _worker(queue, r, m, run, metric, side, time_limit, gap):
    try:
        res = global_bound(r, m, run, metric, side, time_limit, gap)
        queue.put(dict(value=res["value"], bound=res["bound"], status=res["status"]))
    except Exception as e:                       # Python-level errors
        queue.put(dict(value=np.nan, bound=np.nan, status=f"error: {e}"))


def isolated_bound(r, m, run, metric, side, time_limit, gap, hard_timeout):
    """global_bound in a separate process; survives crashes in SCIP's C code."""
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(queue, r, m, run, metric, side, time_limit, gap))
    proc.start()
    proc.join(hard_timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return dict(value=np.nan, bound=np.nan, status=f"killed after {hard_timeout:.0f}s")
    if proc.exitcode != 0:
        return dict(value=np.nan, bound=np.nan, status=f"crashed (exit code {proc.exitcode})")
    try:
        return queue.get(timeout=10)
    except Exception:
        return dict(value=np.nan, bound=np.nan, status="no result returned")


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--gap", type=float, default=1e-6)
    ap.add_argument("--hard-timeout", type=float, default=None,
                    help="stop a solve after this many seconds (default: 10 x time limit + 60;"
                         " some bounds run one solve per pair of groups)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "bounds_scip.csv")
    done = done_keys(path, ["param", "metric", "side"])
    hard = a.hard_timeout or 10 * a.time_limit + 60
    for k, u in load_units(a):
        for metric in METRICS[k["run"]]:
            for side in ("lower", "upper"):
                if key_tuple(k, a.time_limit, metric, side) in done:
                    continue
                t0 = time.perf_counter()
                res = isolated_bound(u["r"], u["m"], k["run"], metric, side, a.time_limit,
                                     a.gap, hard)
                append_rows(path, [dict(**k, method="SCIP", param=a.time_limit, metric=metric,
                                        side=side, value=res["value"],
                                        proven_bound=res["bound"], status=res["status"],
                                        seconds=time.perf_counter() - t0)])
                print(f"  {key_tuple(k)} {metric} {side}: {res['value']:.4f} "
                      f"(proven {res['bound']:.4f}, {res['status']}, "
                      f"{time.perf_counter() - t0:.1f}s)", flush=True)
    print(f"saved {path}")


if __name__ == "__main__":
    mp.freeze_support()
    main()