"""
Shared helpers for the HMDA full experiment (hmda_prepare.py, hmda_<method>.py, hmda_collect.py).

A unit is one combination of
  decision    model (classifier trained on the train split) | lender (D = lender's decision)
  cut_source  external (county cut-offs from the external split) | full (repo's full-data
              county composition)
  rep         random split index (seed)
  setting     Black-White | API-White | three
  proxy       county | income | both
  run         D | DY            (DY is not defined for decision = lender, where Y = D)
  version     split (internal margin from the internal split, external from the external
              split) | exact (both margins from internal + external pooled)
prepared/units.pkl holds, for every unit, the primary table r[z, i], the auxiliary table
m[z, a] (rescaled to the primary p(z)) and the true metric values.

Every method script appends rows to its own CSV in the output folder and skips units (and
parameter values) already done, so it can be stopped, rerun, or extended later.
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent            # .../BiasBounding/experiments/hmda
ROOT = HERE.parent.parent                         # .../BiasBounding
sys.path.insert(0, str(ROOT / "methods"))         # grid_joint.py, kallus_bounds.py, ...
sys.path.insert(0, str(HERE.parent))              # experiments/: exp0_hmda.py, exp_synthetic.py
sys.path.insert(0, str(HERE))                     # experiments/hmda/: hmda_common.py

OUT = ROOT / "experiment_results" / "hmda_full"


PREPARED = OUT / "prepared" / "units.pkl"
UNIT_KEYS = ["decision", "cut_source", "rep", "setting", "proxy", "run", "version"]
CHOICES = dict(decision=["model", "lender"], cut_source=["external", "full"],
               setting=["Black-White", "API-White", "three"], proxy=["county", "income", "both"],
               run=["D", "DY"], version=["split", "exact"])


def add_unit_filters(ap):
    """Command-line filters to run a subset of the prepared units."""
    for k, vals in CHOICES.items():
        ap.add_argument(f"--{k.replace('_', '-')}s" if k != "proxy" else "--proxies",
                        dest=k, nargs="+", choices=vals, default=vals)
    ap.add_argument("--reps", type=int, nargs="+", default=None, help="default: all prepared")
    ap.add_argument("--out", default=str(OUT))
    return ap


def load_units(a):
    """Prepared units matching the filters: list of (key dict, unit dict)."""
    with open(PREPARED, "rb") as f:
        units = pickle.load(f)
    sel = []
    for key, u in units.items():
        k = dict(zip(UNIT_KEYS, key))
        if all(k[c] in getattr(a, c) for c in CHOICES) and (a.reps is None or k["rep"] in a.reps):
            sel.append((k, u))
    return sel


def done_keys(path, extra=()):
    """Set of (unit keys + extra columns) already in a results CSV."""
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    cols = UNIT_KEYS + list(extra)
    return set(map(tuple, df[cols].astype(str).values))


def key_tuple(k, *extra):
    return tuple(str(k[c]) for c in UNIT_KEYS) + tuple(str(e) for e in extra)


def append_rows(path, rows):
    if rows:
        pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def bounds_rows(k, method, param, bounds, seconds, **extra):
    """Rows for {metric: (lower, upper)}."""
    rows = []
    for metric, (lo, hi) in bounds.items():
        for side, v in (("lower", lo), ("upper", hi)):
            rows.append(dict(**k, method=method, param=param, metric=metric, side=side,
                             value=v, seconds=seconds, **extra))
    return rows


def base_parser(description):
    return add_unit_filters(argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter))