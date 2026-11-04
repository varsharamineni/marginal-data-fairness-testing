"""
Step 7 of the HMDA full experiment: combine whatever bounds have been computed.

Reads every bounds_*.csv in the output folder (grid, CI, Kallus, hit-and-run, vertex, SCIP),
attaches the true values from the prepared units, and writes
  bounds_long.csv   one row per unit x metric x method x param x side (+ truths)
  bounds_wide.csv   one row per unit x metric: truth_pooled, truth_internal, and for each
                    method/param its lower and upper bound (CI: its point value)
  coverage.csv      per method, param, version and internal variable: the share of complete
                    intervals that contain truth_pooled and truth_internal (interval methods
                    only); 'incomplete' counts intervals with a missing bound (e.g. a failed
                    SCIP solve)
No method is treated as a reference.

Usage
  python hmda_collect.py
"""
import os
import pickle

import numpy as np
import pandas as pd

from hmda_common import OUT, PREPARED, UNIT_KEYS


def main():
    method_files = ["bounds_grid.csv", "bounds_ci.csv", "bounds_kallus.csv",
                    "bounds_hit_and_run.csv", "bounds_vertex.csv", "bounds_scip.csv", "bounds_mi.csv"]
    files = [os.path.join(OUT, f) for f in method_files if os.path.exists(os.path.join(OUT, f))]
    if not files:
        raise SystemExit(f"no method bounds files in {OUT}")
    df = pd.concat([pd.read_csv(f, dtype={"param": str}) for f in files], ignore_index=True)
    df = df[df.metric.notna() & (df.metric != "")]
    df["param"] = df.param.fillna("").astype(str)
    with open(PREPARED, "rb") as f:
        units = pickle.load(f)
    truth = []
    for key, u in units.items():
        for metric in u["truth_pooled"]:
            truth.append(dict(zip(UNIT_KEYS, key), metric=metric,
                              truth_pooled=u["truth_pooled"][metric],
                              truth_internal=u["truth_internal"][metric]))
    truth = pd.DataFrame(truth)
    for c in UNIT_KEYS:
        df[c] = df[c].astype(str)
        truth[c] = truth[c].astype(str)
    df = df.merge(truth, on=UNIT_KEYS + ["metric"], how="left")
    df.to_csv(os.path.join(OUT, "bounds_long.csv"), index=False)

    df["col"] = df.method + np.where(df.param != "", "_" + df.param, "") + "_" + df.side
    wide = df.pivot_table(index=UNIT_KEYS + ["metric", "truth_pooled", "truth_internal"],
                          columns="col", values="value", aggfunc="first").reset_index()
    wide.to_csv(os.path.join(OUT, "bounds_wide.csv"), index=False)

    iv = df[df.side.isin(["lower", "upper"])]
    iv = iv.pivot_table(index=UNIT_KEYS + ["metric", "method", "param", "truth_pooled",
                                           "truth_internal"],
                        columns="side", values="value", aggfunc="first").reset_index()
    iv["complete"] = np.isfinite(iv.lower) & np.isfinite(iv.upper)
    for t in ("truth_pooled", "truth_internal"):
        covers = (iv.lower <= iv[t] + 1e-9) & (iv[t] <= iv.upper + 1e-9)
        iv[f"covers_{t}"] = covers.where(iv.complete)          # NaN for incomplete intervals
    cov = (iv.groupby(["method", "param", "version", "run"])
             .agg(intervals=("lower", "size"), incomplete=("complete", lambda c: int((~c).sum())),
                  covers_truth_pooled=("covers_truth_pooled", "mean"),
                  covers_truth_internal=("covers_truth_internal", "mean"))
             .reset_index())
    cov.to_csv(os.path.join(OUT, "coverage.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(cov.round(3).to_string(index=False))
    print(f"\nwrote bounds_long.csv, bounds_wide.csv, coverage.csv to {OUT}/")


if __name__ == "__main__":
    main()