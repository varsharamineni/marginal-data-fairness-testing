"""
Step 2 of the HMDA full experiment: conditional-independence point values.

Assumes A is independent of the internal variable given the proxy, so the joint is
q(z, i, a) = p(z, i) p(a | z); every metric is evaluated on that single joint.

Output: <out>/bounds_ci.csv (side = point)

Usage
  python hmda_ci.py
"""
import os
import time

from hmda_common import append_rows, base_parser, done_keys, key_tuple, load_units
from ci_baseline import ci_joint  # noqa: E402
from synth import metrics  # noqa: E402


def main():
    a = base_parser(__doc__).parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "bounds_ci.csv")
    done = done_keys(path)
    for k, u in load_units(a):
        if key_tuple(k) in done:
            continue
        t0 = time.perf_counter()
        vals = {m_: float(v[0]) for m_, v in metrics(ci_joint(u["r"], u["m"]), k["run"]).items()}
        sec = time.perf_counter() - t0
        append_rows(path, [dict(**k, method="CI", param="", metric=m_, side="point", value=v,
                                seconds=sec) for m_, v in vals.items()])
    print(f"saved {path}")


if __name__ == "__main__":
    main()