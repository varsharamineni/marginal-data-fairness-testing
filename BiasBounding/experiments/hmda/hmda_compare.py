"""
How much do the methods' bounds differ? Reads <out>/bounds_wide.csv (run hmda_collect.py first).

Every method except SCIP evaluates feasible joint distributions only, so its interval lies inside
the true bounds. The reference for each unit x metric is therefore the envelope: the smallest
lower and the largest upper bound over all interval methods (Kallus, grid, hit-and-run, vertex,
SCIP). With --reference <method>, that method is the reference instead (for grid, SCIP,
vertex and hit-and-run its largest parameter that ran for each unit x metric); rows where it is
missing are dropped. --reference grid compares every other method with the feasible-set grid.

For each method and parameter, and each unit x metric:
  gap_lower   lower - ref_lower      (>= 0: the method's lower bound is too high)
  gap_upper   ref_upper - upper      (>= 0: the method's upper bound is too low)
  max_gap     max(|gap_lower|, |gap_upper|)
  rel_width   width / reference width (1 = same width; < 1 = too narrow)
  matches     max_gap <= --tol

Outputs (in <out>/compare/)
  compare_long.csv      one row per unit x metric x method x param
  compare_summary.csv   per method, param and metric: number of intervals, share matching,
                        median and max of max_gap, median and min of rel_width
  kallus_agreement.csv  per method and param, for the metrics Kallus et al. cover: the largest
                        absolute difference from the Kallus bounds

Usage
  python hmda_compare.py                                   # classifier, external cuts, split
  python hmda_compare.py --proxy county                    # the figure's setting
  python hmda_compare.py --reference grid                  # every method against the grid
  python hmda_compare.py --reference SCIP --tol 1e-4
  python hmda_compare.py --decision lender --version exact --proxy both
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

from hmda_common import OUT

METHODS = ["Kallus", "grid", "hit-and-run", "vertex", "SCIP"]
COL = re.compile(r"^(Kallus|grid|hit-and-run|vertex|SCIP)(?:_([0-9.]+))?_(lower|upper)$")
KEYS = ["decision", "cut_source", "rep", "setting", "proxy", "run", "version", "metric"]


def long_format(w):
    """One row per unit x metric x method x param with lower and upper."""
    cols = {}
    for c in w.columns:
        m = COL.match(c)
        if m:
            method, param, side = m.group(1), m.group(2) or "", m.group(3)
            cols.setdefault((method, param), {})[side] = c
    parts = []
    for (method, param), sides in cols.items():
        if set(sides) != {"lower", "upper"}:
            continue
        p = w[KEYS + ["truth_pooled"]].copy()
        p["method"], p["param"] = method, param
        p["lower"], p["upper"] = w[sides["lower"]].to_numpy(), w[sides["upper"]].to_numpy()
        parts.append(p)
    d = pd.concat(parts, ignore_index=True)
    return d[np.isfinite(d.lower) & np.isfinite(d.upper)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decision", nargs="+", default=["model"])
    ap.add_argument("--cut-source", nargs="+", default=["external"])
    ap.add_argument("--version", nargs="+", default=["split"])
    ap.add_argument("--proxy", nargs="+", default=["county", "income", "both"])
    ap.add_argument("--settings", nargs="+", default=["Black-White", "API-White", "three"])
    ap.add_argument("--reps", type=int, nargs="+", default=None, help="default: all")
    ap.add_argument("--reference", default="envelope",
                    choices=["envelope", "grid", "SCIP", "Kallus", "vertex", "hit-and-run"],
                    help="grid: the finest grid that ran for each unit x metric")
    ap.add_argument("--tol", type=float, default=1e-4, help="max_gap counted as a match")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    w = pd.read_csv(os.path.join(a.out, "bounds_wide.csv"))
    sel = (w.decision.isin(a.decision) & w.cut_source.isin(a.cut_source)
           & w.version.isin(a.version) & w.proxy.isin(a.proxy) & w.setting.isin(a.settings))
    if a.reps is not None:
        sel &= w.rep.isin(a.reps)
    w = w[sel]
    if w.empty:
        raise SystemExit("no rows for this selection")
    d = long_format(w)

    if a.reference == "envelope":
        ref = d.groupby(KEYS).agg(ref_lower=("lower", "min"), ref_upper=("upper", "max"))
    else:
        r = d[d.method == a.reference]
        if a.reference != "Kallus":            # largest parameter, e.g. longest time limit
            r = r.assign(p=r.param.astype(float)).sort_values("p").groupby(KEYS).tail(1)
        ref = r.set_index(KEYS)[["lower", "upper"]].rename(
            columns={"lower": "ref_lower", "upper": "ref_upper"})
    d = d.join(ref, on=KEYS, how="inner")

    d["width"] = d.upper - d.lower
    ref_w = d.ref_upper - d.ref_lower
    d["gap_lower"] = d.lower - d.ref_lower
    d["gap_upper"] = d.ref_upper - d.upper
    d["max_gap"] = np.maximum(d.gap_lower.abs(), d.gap_upper.abs())
    d["rel_width"] = np.where(ref_w > 1e-12, d.width / ref_w.where(ref_w > 1e-12, 1), np.nan)
    d["matches"] = d.max_gap <= a.tol
    # signed comparison: how far the method's interval extends beyond the reference (wider)
    # and how far it falls short of it (narrower), at the worse of the two ends
    d["wider"] = np.maximum(-d.gap_lower, -d.gap_upper).clip(lower=0)
    d["narrower"] = np.maximum(d.gap_lower, d.gap_upper).clip(lower=0)
    d["covers_truth"] = (d.lower <= d.truth_pooled + 1e-9) & (d.truth_pooled <= d.upper + 1e-9)

    out = os.path.join(a.out, "compare")
    os.makedirs(out, exist_ok=True)
    d.to_csv(os.path.join(out, "compare_long.csv"), index=False)

    g = d.groupby(["method", "param", "metric"])
    s = g.agg(intervals=("max_gap", "size"), share_match=("matches", "mean"),
              median_max_gap=("max_gap", "median"), max_max_gap=("max_gap", "max"),
              median_rel_width=("rel_width", "median"), min_rel_width=("rel_width", "min"),
              max_rel_width=("rel_width", "max"),
              share_wider=("wider", lambda x: float((x > a.tol).mean())),
              max_wider=("wider", "max"),
              share_narrower=("narrower", lambda x: float((x > a.tol).mean())),
              max_narrower=("narrower", "max"),
              covers_truth=("covers_truth", "mean")).reset_index()
    if a.reference == "grid":     # the reference itself and coarser grids are not of interest
        s = s[s.method != "grid"]
    s["p"] = pd.to_numeric(s.param, errors="coerce")
    s["m"] = s.method.map({m: i for i, m in enumerate(METHODS)})
    s = s.sort_values(["m", "p", "metric"]).drop(columns=["m", "p"])
    s.to_csv(os.path.join(out, "compare_summary.csv"), index=False)

    # agreement with Kallus et al. on the metrics they cover
    k = d[d.method == "Kallus"].set_index(KEYS)[["lower", "upper"]].rename(
        columns={"lower": "k_lower", "upper": "k_upper"})
    kk = d[d.method != "Kallus"].join(k, on=KEYS, how="inner")
    kk["diff"] = np.maximum((kk.lower - kk.k_lower).abs(), (kk.upper - kk.k_upper).abs())
    agree = (kk.groupby(["method", "param"])
               .agg(intervals=("diff", "size"), max_abs_diff=("diff", "max"),
                    median_abs_diff=("diff", "median")).reset_index())
    agree.to_csv(os.path.join(out, "kallus_agreement.csv"), index=False)

    with pd.option_context("display.width", 220, "display.max_rows", 400):
        print(f"reference: {a.reference}, tolerance {a.tol:g}, {len(w)} unit x metric rows\n")
        print("Agreement with Kallus et al. (DD, TPR and PPV differences):")
        print(agree.to_string(index=False, float_format=lambda x: f"{x:.2e}"))
        print("\nShare of bounds matching the reference, by method and metric:")
        print(s.pivot_table(index="metric", columns=["method", "param"], values="share_match")
               .round(2).to_string())
        print("\nMedian width relative to the reference (1 = same width):")
        print(s.pivot_table(index="metric", columns=["method", "param"],
                            values="median_rel_width").round(3).to_string())
        print("\nShare of intervals wider than the reference (beyond the tolerance):")
        print(s.pivot_table(index="metric", columns=["method", "param"], values="share_wider")
               .round(2).to_string())
        print("\nLargest amount wider than the reference:")
        print(s.pivot_table(index="metric", columns=["method", "param"], values="max_wider")
               .round(4).to_string())
        print("\nShare of intervals narrower than the reference (beyond the tolerance):")
        print(s.pivot_table(index="metric", columns=["method", "param"], values="share_narrower")
               .round(2).to_string())
        print("\nLargest amount narrower than the reference:")
        print(s.pivot_table(index="metric", columns=["method", "param"], values="max_narrower")
               .round(4).to_string())
        print("\nLargest gap from the reference bound:")
        print(s.pivot_table(index="metric", columns=["method", "param"], values="max_max_gap")
               .round(3).to_string())
    print(f"\nwrote compare_long.csv, compare_summary.csv, kallus_agreement.csv to {out}/")


if __name__ == "__main__":
    main()