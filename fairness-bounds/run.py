"""Pipeline: data -> (classifier for Yhat if needed) -> CI estimate, Kallus bounds and
(optionally) the metric range over feasible joint distributions.

Usage:  python run.py                 (uses config.yaml)
        python run.py my_config.yaml
"""
import os
import sys

import pandas as pd
import yaml

from fairness import classifier, data, methods


def main(config_path="config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cols, a_levels = cfg["columns"], list(cfg["a_levels"])
    synthetic = cfg["data"]["source"] == "synthetic"

    # 1. data
    if synthetic:
        primary, auxiliary = data.generate_synthetic(cfg["data"]["synthetic"])
        cols = {"yhat": "yhat", "y": "y", "a": "a", "z": ["z"], "features": ["x1", "x2"]}
        a_levels = ["a", "b"]
    elif cfg["data"]["source"] == "csv":
        primary, auxiliary = data.load_csv(cfg["data"]["csv"])
    else:
        raise ValueError("data.source must be 'synthetic' or 'csv'")
    true_a = "a" if synthetic else cfg["data"]["csv"].get("true_a")
    primary, auxiliary = data.prepare(primary, auxiliary, cols, a_levels, true_a=true_a)
    print(f"Primary rows: {len(primary)}   Auxiliary rows: {len(auxiliary)}   "
          f"Z levels: {primary['z'].nunique()}")

    # 2. classifier (only if Yhat is not already in the data)
    if "yhat" not in primary.columns:
        primary, info = classifier.add_yhat(primary, cols["features"], cfg["classifier"])
        print(f"Trained {info['model']} on {info['n_train']} rows; Yhat on {info['n_held_out']} "
              f"held-out rows (accuracy {info['held_out_accuracy']:.3f})")

    # 3. methods
    a_true = primary.pop("a_true") if "a_true" in primary.columns else None
    jcfgs = cfg.get("joints") or []
    if isinstance(jcfgs, dict):                       # a single sampler config
        jcfgs = [jcfgs]
    joint_vals = {}                                   # sampler name -> DataFrame (joint x metric)
    for jcfg in jcfgs:
        jcfg = dict(jcfg)
        name = jcfg.pop("name", jcfg.get("method", "grid"))
        joints = methods.feasible_joints(primary, auxiliary, a_levels, **jcfg)
        print(f"Feasible joints ({name}): {len(joints)}")
        joint_vals[name] = pd.DataFrame({m: methods.joint_metrics(
            joints, lambda j, m=m: methods.joint_disparity(j, a_levels, m)) for m in cfg["metrics"]})
    rows = []
    for m in cfg["metrics"]:
        if m not in methods.METRICS:
            raise ValueError(f"unknown metric {m}; choose from {methods.METRICS}")
        lo, hi = methods.kallus_bounds(primary, auxiliary, a_levels, m)
        row = {"metric": m,
               "ci_estimate": methods.ci_estimate(primary, auxiliary, a_levels, m),
               "kallus_lower": lo, "kallus_upper": hi}
        for name, vals in joint_vals.items():
            row[f"{name}_min"], row[f"{name}_max"] = vals[m].min(), vals[m].max()
        if a_true is not None:
            row["truth"] = methods.true_disparity(primary["yhat"], primary["y"], a_true, a_levels, m)
        rows.append(row)
    results = pd.DataFrame(rows).set_index("metric")

    print(f"\nDisparities ({a_levels[0]} minus {a_levels[1]}):")
    print(results.round(4).to_string())

    out = cfg["output_dir"]
    os.makedirs(out, exist_ok=True)
    results.to_csv(os.path.join(out, "results.csv"))
    for name, vals in joint_vals.items():
        vals.to_csv(os.path.join(out, f"joint_metrics_{name}.csv"), index_label="joint")
    primary.to_csv(os.path.join(out, "primary_processed.csv"), index=False)
    auxiliary.to_csv(os.path.join(out, "auxiliary_processed.csv"), index=False)
    print(f"\nSaved results and processed data to {out}/")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
