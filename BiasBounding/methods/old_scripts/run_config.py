"""
Run fairness-estimation experiments from YAML config files.

Each scenario in a config describes one experiment end to end:

  data        where the data comes from: synthetic, one full csv, or already split csvs
  preprocess  list of cleaning steps (filter, binary, recode, numeric, bin, ...)
  roles       which columns are protected / common / features / label
  split       how full data is divided into training, primary and auxiliary sets
  models      which models produce the decisions (or an observed decision column)
  metrics     which fairness metrics to compute ('all' or a list)
  methods     which estimation methods to run, with their settings
  compare     seeds, tolerance, plots

A file holds either one scenario (keys at top level) or several:
    defaults: {...}             # merged into every scenario
    scenarios:
      - name: a
        ...
      - name: b
        ...

Command line:
  python run_config.py configs/synthetic.yaml
  python run_config.py configs/*.yaml --out results --only feature_dep
  python run_config.py --list          # available metrics, methods, models, steps

Outputs per scenario in <out>/<name>/: results.csv, summary.csv, method_comparison.csv,
interval plots, and config_resolved.yaml (the exact settings used). Across all scenarios:
<out>/all_results.csv and <out>/all_summary.csv.
"""
import argparse
import contextlib
import copy
import importlib
import io
import os

import numpy as np
import pandas as pd
import yaml

import BiasBounding.methods.old_scripts.pipeline as pl
from BiasBounding.methods.old_scripts.fairness_metrics import METRICS
from grid_joint import from_datasets

DEFAULTS = {
    "data": {"type": "synthetic", "params": {}},
    "preprocess": [],
    "roles": {"protected": "race", "common": ["income_bin"],
              "features": ["credit_score", "age"], "label": "repaid"},
    "split": {"type": "random", "train": 0.3, "primary": 0.4, "aux_same_rows": False},
    "models": [{"name": "logit", "type": "logistic", "features": ["features", "common"],
                "rate": 0.5}],
    "metrics": "all",
    "methods": [{"name": "independence"}, {"name": "joint", "num": 4},
                {"name": "kallus"}],
    "compare": {"seeds": [0], "tol": 1e-6, "plots": True},
    "binning": {"max_categories": 20, "bins": 3},
}


# ============================================================================ config handling
def merge(base, over):
    """Deep merge: dicts merge recursively, everything else is replaced."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) \
            else copy.deepcopy(v)
    return out


def load(path):
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    base = merge(DEFAULTS, raw.get("defaults", {}))
    items = raw["scenarios"] if "scenarios" in raw else [raw]
    stem = os.path.splitext(os.path.basename(path))[0]
    out = []
    for k, sc in enumerate(items):
        sc = merge(base, {key: v for key, v in sc.items() if key != "defaults"})
        sc.setdefault("name", stem if len(items) == 1 else f"{stem}_{k}")
        validate(sc)
        out.append(sc)
    return out


def validate(sc):
    known = set(DEFAULTS) | {"name", "description"}
    bad = set(sc) - known
    if bad:
        raise SystemExit(f"[{sc['name']}] unknown section(s) {sorted(bad)}; expected {sorted(known)}")
    if sc["metrics"] != "all":
        bad = [m for m in sc["metrics"] if m not in METRICS]
        if bad:
            raise SystemExit(f"[{sc['name']}] unknown metric(s) {bad}; see --list")
    for m in sc["methods"]:
        if m["name"] not in pl.METHODS:
            raise SystemExit(f"[{sc['name']}] unknown method '{m['name']}'; see --list")
    for m in sc["models"]:
        if m.get("type") not in MODEL_TYPES:
            raise SystemExit(f"[{sc['name']}] model '{m.get('name')}' has unknown type "
                             f"'{m.get('type')}'; see --list")
    for s in sc["preprocess"]:
        if len(s) != 1 or next(iter(s)) not in STEPS:
            raise SystemExit(f"[{sc['name']}] bad preprocessing step {s}; see --list")


def _function(ref):
    """'module:function' or 'path/to/file.py:function' -> callable."""
    mod, fn = ref.rsplit(":", 1)
    if mod.endswith(".py"):
        spec = importlib.util.spec_from_file_location(os.path.basename(mod)[:-3], mod)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(mod)
    return getattr(module, fn)


# ============================================================================ preprocessing
# Each step works on a list of frames (one full frame, or train/primary/aux), so that binning
# edges and recodings are shared. A step is applied to every frame that has its columns.
def _has(df, cols):
    return all(c in df.columns for c in ([cols] if isinstance(cols, str) else cols))


def step_filter(frames, query):
    """Keep rows matching a pandas query, e.g. 'action_taken in [1, 2, 3]'."""
    out = []
    for d in frames:
        try:
            out.append(d.query(query).reset_index(drop=True))
        except Exception:              # columns not in this frame
            out.append(d)
    return out


def step_binary(frames, column, positive, output=None):
    """output = 1 if column is in positive (a value or list), else 0."""
    pos = {str(v) for v in (positive if isinstance(positive, list) else [positive])}
    for d in frames:
        if _has(d, column):
            d[output or column] = d[column].astype(str).isin(pos).astype(int)
    return frames


def step_recode(frames, column, map, default=None, output=None):
    """Map values; values not in map keep their value, or become `default` if given."""
    m = {str(k): v for k, v in map.items()}
    for d in frames:
        if _has(d, column):
            s = d[column].astype(str).map(m)
            d[output or column] = s.fillna(d[column] if default is None else default)
    return frames


def step_numeric(frames, columns, fillna="median"):
    """Coerce to numbers ('Exempt', 'NA' -> missing), then fill: median, zero, drop or a value."""
    cols = [columns] if isinstance(columns, str) else columns
    for d in frames:
        for c in cols:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
    if fillna == "drop":
        return [d.dropna(subset=[c for c in cols if c in d.columns]).reset_index(drop=True)
                for d in frames]
    for c in cols:
        present = [d for d in frames if c in d.columns]
        if not present:
            continue
        v = 0 if fillna == "zero" else \
            pd.concat([d[c] for d in present]).median() if fillna == "median" else fillna
        for d in present:
            d[c] = d[c].fillna(v)
    return frames


def step_bin(frames, column, bins=3, method="quantile", edges=None, output=None):
    """Bin a numeric column with edges shared by all frames (labels 0, 1, ...)."""
    present = [d for d in frames if column in d.columns]
    vals = pd.to_numeric(pd.concat([d[column] for d in present]), errors="coerce").dropna()
    if edges is None:
        edges = (np.quantile(vals, np.linspace(0, 1, bins + 1)) if method == "quantile"
                 else np.linspace(vals.min(), vals.max(), bins + 1))
    edges = np.unique(edges)
    for d in present:
        x = pd.to_numeric(d[column], errors="coerce")
        d[output or column] = pd.cut(x, edges, labels=False, include_lowest=True)
    return [d.dropna(subset=[output or column]).reset_index(drop=True)
            if (output or column) in d.columns else d for d in frames]


def step_string(frames, columns):
    """Treat as categorical text; missing becomes 'NA'."""
    for d in frames:
        for c in ([columns] if isinstance(columns, str) else columns):
            if c in d.columns:
                d[c] = d[c].astype(str).replace({"nan": "NA", "None": "NA"})
    return frames


def step_dropna(frames, columns):
    """Drop rows with missing values in these columns."""
    cols = [columns] if isinstance(columns, str) else columns
    return [d.dropna(subset=[c for c in cols if c in d.columns]).reset_index(drop=True)
            for d in frames]


def step_rename(frames, map):
    """Rename columns: {old: new}."""
    return [d.rename(columns=map) for d in frames]


def step_keep(frames, columns):
    """Keep only these columns (those present in each frame)."""
    return [d[[c for c in columns if c in d.columns]] for d in frames]


def step_sample(frames, n=None, frac=None, seed=0):
    """Random subsample of every frame (for speed)."""
    return [d.sample(n=min(n, len(d)) if n else None, frac=frac, random_state=seed)
            .reset_index(drop=True) for d in frames]


def step_min_count(frames, columns, n):
    """Drop rows whose combination of these columns has fewer than n rows (all frames counted)."""
    cols = [columns] if isinstance(columns, str) else columns
    present = [d for d in frames if _has(d, cols)]
    counts = pd.concat([d[cols] for d in present]).value_counts()
    ok = set(counts[counts >= n].index)
    out = []
    for d in frames:
        if _has(d, cols):
            k = d[cols].apply(tuple, axis=1) if len(cols) > 1 else d[cols[0]]
            d = d[k.isin(ok)].reset_index(drop=True)
        out.append(d)
    return out


def step_custom(frames, function, **kwargs):
    """Your own step: function(df, **kwargs) -> df, given as 'module:function' or 'file.py:function'."""
    f = _function(function)
    return [f(d, **kwargs) for d in frames]


STEPS = {"filter": step_filter, "binary": step_binary, "recode": step_recode,
         "numeric": step_numeric, "bin": step_bin, "string": step_string,
         "dropna": step_dropna, "rename": step_rename, "keep": step_keep,
         "sample": step_sample, "min_count": step_min_count, "custom": step_custom}


def preprocess(frames, steps):
    frames = [d.copy() for d in frames]
    for s in steps:
        name, args = next(iter(s.items()))
        if not isinstance(args, dict):       # shorthand: {filter: "query"}, {string: [cols]}
            args = {"query": args} if name == "filter" else {"columns": args}
        frames = STEPS[name](frames, **args)
    return frames


# ============================================================================ data + split
_CACHE = {}


def read_table(path, columns=None):
    """csv, csv.gz or parquet; optionally only some columns."""
    if str(path).endswith(".parquet"):
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns, low_memory=False)


def _load_and_preprocess(sc, seed):
    """Raw frames after preprocessing. Deterministic inputs are cached, so a large master
    dataset is read and cleaned once and then re-split for every seed and scenario."""
    d = sc["data"]
    cols = d.get("columns")
    key = None
    if d["type"] in ("csv", "file", "presplit"):
        paths = [d.get("path")] if d["type"] != "presplit" else [d.get(k) for k in ("train", "primary", "aux")]
        key = (yaml.safe_dump(paths), yaml.safe_dump(cols), yaml.safe_dump(sc["preprocess"]))
        if key in _CACHE:
            return [None if f is None else f.copy() for f in _CACHE[key]]
    if d["type"] == "synthetic":
        frames = [pl.synthetic(seed, **d.get("params", {}))[0]]
    elif d["type"] in ("csv", "file"):
        frames = [read_table(d["path"], cols)]
    elif d["type"] == "presplit":
        frames = [read_table(d[k]) if d.get(k) else None for k in ("train", "primary", "aux")]
    else:
        raise SystemExit(f"[{sc['name']}] data.type must be synthetic, file or presplit")
    present = preprocess([f for f in frames if f is not None], sc["preprocess"])
    it = iter(present)
    frames = [None if f is None else next(it) for f in frames]
    if key is not None:
        _CACHE.clear()                      # keep only the latest dataset in memory
        _CACHE[key] = frames
        return [None if f is None else f.copy() for f in frames]
    return frames


def _size(x, n):
    """Fraction (float <= 1) or number of rows (int)."""
    return int(round(x * n)) if isinstance(x, float) and x <= 1 else int(x)


def get_data(sc, seed):
    """Returns (train or None, primary, aux, aux weight column or None)."""
    d, sp, roles = sc["data"], sc["split"], sc["roles"]
    frames = _load_and_preprocess(sc, seed)
    if d["type"] == "presplit":
        train, primary, aux = frames
    else:
        full = frames[0]
        if sp["type"] == "random":
            train, primary, aux = pl.split(full, roles, seed,
                                           sp.get("train", 0.3), sp.get("primary", 0.4),
                                           sp.get("aux_same_rows", False))
            if sp.get("train", 0.3) == 0:
                train = None
        elif sp["type"] == "sample":
            n = len(full)
            idx = np.random.default_rng(seed).permutation(n)
            n_pr, n_tr = _size(sp["primary"], n), _size(sp.get("train", 0), n)
            if n_pr + n_tr > n:
                raise SystemExit(f"[{sc['name']}] primary + train ({n_pr + n_tr}) > {n} rows")
            mode = sp.get("aux", "rest")
            au = idx if mode == "all" else idx[n_pr + n_tr:] if mode == "rest" else \
                idx[n_pr + n_tr:n_pr + n_tr + _size(mode, n)]
            primary = full.iloc[idx[:n_pr]].reset_index(drop=True)
            train = full.iloc[idx[n_pr:n_pr + n_tr]].reset_index(drop=True) if n_tr else None
            aux = full.iloc[au].reset_index(drop=True)
        elif sp["type"] == "column":
            col, v = sp["column"], sp["values"]
            pick = lambda key: full[full[col].astype(str).isin([str(x) for x in v[key]])] \
                .reset_index(drop=True) if v.get(key) is not None else None
            train, primary, aux = pick("train"), pick("primary"), pick("aux")
        else:
            raise SystemExit(f"[{sc['name']}] split.type must be random, sample or column")
    keep = roles["common"] + pl.plist(roles["protected"])
    wa = d.get("weight_aux")
    if sp.get("aggregate_aux"):
        w = aux[wa] if wa else 1
        aux = aux.assign(_w=w).groupby(keep, observed=True)["_w"].sum().rename("count") \
            .reset_index()
        wa = "count"
    aux = aux[keep + ([wa] if wa and wa in aux.columns else [])]
    return train, primary, aux, wa


# ============================================================================ models
def _model_columns(spec, roles):
    out = []
    for f in spec.get("features", ["features", "common"]):
        out += pl._columns(roles, [f]) if f in ("features", "common", "protected") else [f]
    return list(dict.fromkeys(out))


def _estimator(kind, params, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, **params))
    if kind == "gbm":
        return HistGradientBoostingClassifier(random_state=seed, **params)
    if kind == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
    return _function(params.pop("function"))(**params)      # kind == 'custom'


def make_model(spec, roles):
    """fit_predict(train, primary, roles, rate, seed) -> 0/1 decisions for the primary rows."""
    if spec["type"] == "observed":
        return pl.observed(spec["column"])

    def fit_predict(train, primary, roles_, rate, seed):
        cols = _model_columns(spec, roles)
        cat = roles["common"] + pl.plist(roles["protected"])
        Xtr = pl._design(train, cols, cat)
        Xpr = pl._design(primary, cols, cat, Xtr.columns)
        est = _estimator(spec["type"], dict(spec.get("params", {})), seed)
        est.fit(Xtr, train[roles["label"]].astype(int))
        s_tr, s_pr = est.predict_proba(Xtr)[:, 1], est.predict_proba(Xpr)[:, 1]
        thr = spec["threshold"] if "threshold" in spec else \
            np.quantile(s_tr, 1 - spec.get("rate", 0.5))
        return (s_pr >= thr).astype(int)
    return fit_predict


MODEL_TYPES = {"logistic": "logistic regression (params passed to sklearn)",
               "gbm": "histogram gradient boosting",
               "random_forest": "random forest",
               "observed": "an existing decision column: {column: approved}",
               "custom": "params: {function: 'file.py:make_estimator', ...} returning an "
                         "sklearn-style classifier"}


# ============================================================================ run one scenario
def run_scenario(sc, out_root, verbose=False):
    name, roles = sc["name"], dict(sc["roles"], decision="yhat")
    roles.setdefault("features", [])
    roles.setdefault("label", None)
    roles["common"] = pl.plist(roles["common"])
    lab, com, dec = roles["label"], roles["common"], "yhat"
    prot = pl.plist(roles["protected"])
    extra = pl.plist(roles.get("internal"))       # further internal variables in the grid
    metrics = list(METRICS) if sc["metrics"] == "all" else list(sc["metrics"])
    if lab is None:
        dropped = [m for m in metrics if METRICS[m]["needs_label"]]
        metrics = [m for m in metrics if not METRICS[m]["needs_label"]]
        if dropped and sc["metrics"] != "all":
            print(f"[{name}] no label: dropping {dropped}")
    cmp_ = sc["compare"]
    quiet = (lambda: contextlib.nullcontext()) if verbose else \
        (lambda: contextlib.redirect_stdout(io.StringIO()))

    rows = []
    for seed in cmp_.get("seeds", [0]):
        train, primary, aux, wa = get_data(sc, seed)
        with quiet():
            primary, aux = pl._discretise(primary, aux, com, sc["binning"]["max_categories"],
                                          sc["binning"]["bins"])
        internal_cols = [dec] + ([lab] if lab else []) + extra
        print(f"[{name}] seed {seed}: primary {len(primary):,} rows, auxiliary {len(aux):,} "
              f"rows{' (aggregated)' if wa == 'count' else ''}"
              f"{'' if train is None else f', train {len(train):,} rows'}")
        for spec in sc["models"]:
            mname = spec["name"]
            print(f"[{name}] seed {seed} | model {mname}")
            if set(prot) & set(_model_columns(spec, roles)) and not set(prot) <= set(primary.columns):
                print("    skipped: model uses the protected attribute, not in the primary data")
                continue
            fp = make_model(spec, roles)
            if spec["type"] != "observed" and lab is None:
                raise SystemExit(f"[{name}] model '{mname}' needs roles.label")
            primary[dec] = fp(train, primary, roles, None, seed) if spec["type"] == "observed" \
                else (pl.cross_fit(fp, primary, roles, None, seed) if train is None
                      else fp(train, primary, roles, None, seed))
            with quiet():
                p_primary, p_aux, internal, _, _ = from_datasets(
                    primary[internal_cols + com], aux, internal_cols, com, prot, weight_aux=wa)
            with quiet():
                true = pl.truth(primary, roles, internal)
            for m in sc["methods"]:
                cfg = dict(num=m.get("num", 4), grid_method=m.get("grid_method", "sequential"),
                           max_joints=m.get("max_joints", 2_000_000),
                           kallus_extended=m.get("extended", False), metrics=metrics)
                label = m.get("label", m["name"])
                try:
                    with quiet():
                        res = pl.METHODS[m["name"]](p_primary, p_aux, internal, roles, cfg)
                except SystemExit as e:
                    print(f"    {label} skipped: {e}")
                    continue
                for r in res[res.metric.isin(metrics)].itertuples(index=False):
                    rows.append(dict(scenario=name, seed=seed, classifier=mname, method=label,
                                     metric=r.metric, lower=r.lower, upper=r.upper,
                                     truth=true.get(r.metric, np.nan)))
    res = pd.DataFrame(rows)
    tol = cmp_.get("tol", 1e-6)
    res["width"] = res.upper - res.lower
    res["covered"] = ((res.lower - tol <= res.truth) & (res.truth <= res.upper + tol)).astype(float)
    res.loc[res.truth.isna(), "covered"] = np.nan
    res["miss"] = np.maximum(0, np.maximum(res.lower - res.truth, res.truth - res.upper))
    res["midpoint_error"] = (res.lower + res.upper) / 2 - res.truth

    out = os.path.join(out_root, name)
    os.makedirs(out, exist_ok=True)
    res.to_csv(os.path.join(out, "results.csv"), index=False)
    summ = pl.summarise(res)
    summ.to_csv(os.path.join(out, "summary.csv"), index=False)
    pl.compare_methods(res, tol).to_csv(os.path.join(out, "method_comparison.csv"), index=False)
    if cmp_.get("plots", True) and len(res):
        pl.plot(res, out)
    with open(os.path.join(out, "config_resolved.yaml"), "w") as f:
        yaml.safe_dump(sc, f, sort_keys=False)
    return res, summ


# ============================================================================ command line
def list_options():
    print("metrics:")
    for k, v in METRICS.items():
        print(f"  {k:26s}{'(needs label) ' if v['needs_label'] else ''}{v['description']}")
    print("\nmethods: " + ", ".join(pl.METHODS) +
          "   (joint: num, grid_method, max_joints; kallus: extended; any: label)")
    print("\nmodel types:")
    for k, v in MODEL_TYPES.items():
        print(f"  {k:15s}{v}")
    print("\npreprocessing steps:")
    for k, f in STEPS.items():
        print(f"  {k:9s}{' '.join((f.__doc__ or '').split())}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="*")
    ap.add_argument("--out", default="results")
    ap.add_argument("--only", nargs="+", help="run only these scenario names")
    ap.add_argument("--list", action="store_true", help="show available options")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    if a.list:
        return list_options()
    if not a.configs:
        ap.error("give at least one config file (or --list)")

    scenarios = [sc for path in a.configs for sc in load(path)]
    names = [sc["name"] for sc in scenarios]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate scenario names: {names}")
    if a.only:
        scenarios = [sc for sc in scenarios if sc["name"] in a.only]
    all_res, all_sum = [], []
    for sc in scenarios:
        r, s = run_scenario(sc, a.out, a.verbose)
        all_res.append(r)
        all_sum.append(s)
    os.makedirs(a.out, exist_ok=True)
    res = pd.concat(all_res, ignore_index=True)
    res.to_csv(os.path.join(a.out, "all_results.csv"), index=False)
    pd.concat(all_sum, ignore_index=True).to_csv(os.path.join(a.out, "all_summary.csv"),
                                                 index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 500):
        print("\nmean distance from the truth to the interval (0 = covered; point estimates: |error|):")
        print(res.pivot_table(index="metric", columns=["scenario", "method"], values="miss",
                              aggfunc="mean").round(3).to_string())
    print(f"\nsaved to {a.out}/")


if __name__ == "__main__":
    main()