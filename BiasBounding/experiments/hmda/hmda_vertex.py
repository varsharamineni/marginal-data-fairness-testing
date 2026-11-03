"""
Step 5 of the HMDA full experiment: vertex bounds.

For every selected unit, the vertices of each proxy level's feasible table are enumerated and
combined once (vertex_sampling.iter_vertex_joints): all combinations if there are at most
--cap, otherwise --n random combinations. All metrics are evaluated on those same joints; the
bounds are their min and max.

Output: <out>/bounds_vertex.csv (param = cap; full_enumeration; combinations; evaluated)

Usage
  python hmda_vertex.py
  python hmda_vertex.py --cap 1000000 --n 200000
"""
import os
import time

from hmda_common import append_rows, base_parser, bounds_rows, done_keys, key_tuple, load_units
from vertex_sampling import vertex_bounds  # noqa: E402
from synth import metrics  # noqa: E402


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--cap", type=int, default=500_000, help="full enumeration up to this many")
    ap.add_argument("--n", type=int, default=100_000, help="random combinations above the cap")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "bounds_vertex.csv")
    done = done_keys(path, ["param"])
    for k, u in load_units(a):
        if key_tuple(k, a.cap) in done:
            continue
        t0 = time.perf_counter()
        b, info = vertex_bounds(u["r"], u["m"], lambda X: metrics(X, k["run"]), n=a.n,
                                cap=a.cap, seed=a.seed)
        sec = time.perf_counter() - t0
        append_rows(path, bounds_rows(k, "vertex", a.cap, b, sec,
                                      full_enumeration=info["full_enumeration"],
                                      combinations=info["combinations"],
                                      evaluated=info["evaluated"]))
        print(f"  {key_tuple(k)}: {info['evaluated']:,} of {info['combinations']:,} "
              f"combinations, {sec:.2f}s", flush=True)
    print(f"saved {path}")


if __name__ == "__main__":
    main()