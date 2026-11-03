"""
Step 3 of the HMDA full experiment: Kallus et al. bounds, for the metrics they cover.

  internal D    DD 1-0 (sharp LP, any number of groups)
  internal DY   TPRD 1-0, PPVD 1-0 (sharp LP over couplings with a grid over group masses;
                up to --max-groups groups)
Other metrics have no Kallus et al. bounds and are not recorded.

Output: <out>/bounds_kallus.csv

Usage
  python hmda_kallus.py
  python hmda_kallus.py --lp-grid 41
"""
import os
import time

from hmda_common import append_rows, base_parser, bounds_rows, done_keys, key_tuple, load_units
from synth import kallus_refs  # noqa: E402


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--lp-grid", type=int, default=21, help="grid over group masses (3+ groups)")
    ap.add_argument("--max-groups", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "bounds_kallus.csv")
    done = done_keys(path)
    for k, u in load_units(a):
        if key_tuple(k) in done:
            continue
        t0 = time.perf_counter()
        b = kallus_refs(u["r"], u["m"], k["run"], u["m"].shape[1], a.max_groups, a.lp_grid)
        sec = time.perf_counter() - t0
        rows = bounds_rows(k, "Kallus", "", b, sec)
        if not rows:     # nothing covered: mark the unit as done
            rows = [dict(**k, method="Kallus", param="", metric="", side="", value=float("nan"),
                         seconds=sec)]
        append_rows(path, rows)
        print(f"  {key_tuple(k)}: {sec:.2f}s", flush=True)
    print(f"saved {path}")


if __name__ == "__main__":
    main()