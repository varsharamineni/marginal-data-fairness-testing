"""
Step 4 of the HMDA full experiment: hit-and-run bounds.

For every selected unit, n joints are drawn uniformly from the feasible set once
(hit_and_run.hit_and_run_joints) and all metrics are evaluated on those same samples; the
bounds are their min and max.

Output: <out>/bounds_hit_and_run.csv (param = number of samples)

Usage
  python hmda_hit_and_run.py
  python hmda_hit_and_run.py --n 20000 --burn 2000 --thin 5
"""
import os
import time

from hmda_common import append_rows, base_parser, bounds_rows, done_keys, key_tuple, load_units
from hit_and_run import bounds as sample_bounds, hit_and_run_joints  # noqa: E402
from synth import metrics  # noqa: E402


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--burn", type=int, default=1_000)
    ap.add_argument("--thin", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "bounds_hit_and_run.csv")
    done = done_keys(path, ["param"])
    for k, u in load_units(a):
        if key_tuple(k, a.n) in done:
            continue
        t0 = time.perf_counter()
        Q = hit_and_run_joints(u["r"], u["m"], n=a.n, burn=a.burn, thin=a.thin, seed=a.seed)
        b = sample_bounds(Q, lambda X: metrics(X, k["run"]))
        sec = time.perf_counter() - t0
        append_rows(path, bounds_rows(k, "hit-and-run", a.n, b, sec))
        print(f"  {key_tuple(k)}: {sec:.1f}s", flush=True)
    print(f"saved {path}")


if __name__ == "__main__":
    main()