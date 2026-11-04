"""
Supplement to the HMDA full experiment: sensitivity of the Kallus et al. bounds to the binning
of the proxies.

Everything except the binning matches hmda_prepare.py and hmda_kallus.py: the same random split
(seed = rep), the same classifier (trained on the 20% train split), bin boundaries computed from
the external split and applied to every record, the same joint tables, truths and
incompatibility check, and the Kallus et al. bounds from synth.kallus_refs.

Schemes (the scheme hmda_prepare.py uses is called "main")
  county   none      one level (no proxy)
           main      share >= 0.30 (Black) or >= 0.15 (API); 3 levels for three groups
           cut:a,b   fixed cut-offs on the county's share of group 1          (pairs only)
           q<k>      k quantile bins of that share, quantiles over the external
                     applicants of group 1                                      (pairs only)
           full      every county its own level
  income   none, main (external median, = q2), q<k> (external quantiles), full (every value)
  both     <county scheme>+<income scheme>, e.g. main+main, q4+q4, full+full

Versions
  exact   r and m both from internal + external pooled: the bounds reflect only what the
          binning keeps (the truth always lies inside)
  split   r from internal, m from external, as in the main experiment: adds sampling noise
          and incompatible levels (internal records but no external records); such units are
          recorded with no bounds, with n_incompatible and incompatible_mass filled in

Output
  <out>/binning_long.csv     one row per bound (as bounds_kallus.csv) plus scheme columns;
                             rerunning skips units already in this file
  <out>/binning_summary.csv  one row per unit and metric: n_levels, smallest level in each
                             split, truth, lower, upper, width, width with no proxy,
                             narrowing, and whether the bounds contain the pooled truth

Usage
  python hmda_binning.py --repo ~/kallus
  python hmda_binning.py --repo ~/kallus --versions exact --decisions lender
  python hmda_binning.py --repo ~/kallus --county none main q4 --income none main --both main+main
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from hmda_prepare import (CUT, GROUPS, classifier_decision, county_shares,  # noqa: E402
                          population, split_data)
from hmda_common import UNIT_KEYS, bounds_rows  # noqa: E402
from ci_baseline import harmonize  # noqa: E402
from exp0_hmda import load_hmda  # noqa: E402
from synth import joint_table, kallus_refs, metrics  # noqa: E402

COUNTY_DEFAULT = ["none", "main", "cut:0.10", "cut:0.20", "q2", "q4", "q8", "q16", "full"]
INCOME_DEFAULT = ["none", "main", "q3", "q4", "q8", "q16", "full"]
BOTH_DEFAULT = ["main+main", "q4+q4", "q8+q8", "q16+q16", "full+full"]
INFO_COLS = ["n_levels", "min_cell_internal", "min_cell_external", "n_internal", "n_external",
             "n_incompatible", "incompatible_mass"]


def qcuts(x, k):
    return np.unique(np.quantile(x, np.linspace(0, 1, k + 1)[1:-1]))


def income_levels(d, split, scheme):
    x = d.income.to_numpy(float)
    if scheme == "none":
        return np.zeros(len(d), int)
    if scheme == "full":
        return pd.factorize(d.income, sort=True)[0]
    if scheme == "main" or (scheme[0] == "q" and scheme[1:].isdigit()):
        k = 2 if scheme == "main" else int(scheme[1:])
        # right=True: a record exactly at a cut-off goes to the lower bin, matching
        # income > median in hmda_prepare.py
        return np.digitize(x, qcuts(x[split == "external"], k), right=True)
    raise ValueError(f"bad income scheme '{scheme}'")


def county_levels(d, split, groups, scheme):
    if scheme == "none":
        return np.zeros(len(d), int)
    if scheme == "full":
        return pd.factorize(d.county_id, sort=True)[0]
    sh = county_shares(d, split, groups, "external")
    if scheme == "main":                      # exactly proxy_levels() in hmda_prepare.py
        if len(groups) == 2:
            return (sh[groups[1]] >= CUT[groups[1]]).astype(int).to_numpy()
        return np.where(sh["Black"] >= CUT["Black"], 1,
                        np.where(sh["API"] >= CUT["API"], 2, 0))
    if len(groups) != 2:
        raise ValueError(f"county scheme '{scheme}' is defined for pairs only")
    s = sh[groups[1]].to_numpy(float)
    if scheme.startswith("cut:"):
        cuts = [float(c) for c in scheme[4:].split(",")]
    elif scheme[0] == "q" and scheme[1:].isdigit():
        members = (split == "external") & (d.race == groups[1]).to_numpy()
        cuts = qcuts(s[members], int(scheme[1:]))
    else:
        raise ValueError(f"bad county scheme '{scheme}'")
    return np.digitize(s, cuts)               # share >= cut goes up, as in "main"


def proxy_z(d, split, groups, proxy, scheme):
    if proxy == "income":
        return income_levels(d, split, scheme)
    if proxy == "county":
        return county_levels(d, split, groups, scheme)
    c, i = scheme.split("+")
    zc = county_levels(d, split, groups, c).astype(np.int64)
    zi = income_levels(d, split, i).astype(np.int64)
    return zc * (int(zi.max()) + 1) + zi


def pairs_only(proxy, scheme):
    parts = scheme.split("+")[:1] if proxy == "both" else ([scheme] if proxy == "county" else [])
    return any(p not in ("none", "main", "full") for p in parts)


def append_csv(path, rows):
    df = pd.DataFrame(rows)
    if os.path.exists(path):
        cols = pd.read_csv(path, nrows=0).columns
        df.reindex(columns=cols).to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def summarize(long_path, out):
    df = pd.read_csv(long_path)
    keys = list(UNIT_KEYS) + ["scheme"]
    has = df.metric.notna() & (df.metric.astype(str) != "")
    agg = dict(lower=("value", "min"), upper=("value", "max"),
               truth_pooled=("truth_pooled", "first"),
               truth_internal=("truth_internal", "first"), seconds=("seconds", "first"),
               **{c: (c, "first") for c in INFO_COLS})
    s = df[has].groupby(keys + ["metric"], dropna=False).agg(**agg).reset_index()
    s["width"] = s.upper - s.lower
    tol = 1e-9
    s["covers_pooled"] = (s.lower - tol <= s.truth_pooled) & (s.truth_pooled <= s.upper + tol)
    group = ["decision", "rep", "setting", "run", "version", "metric"]
    none = (s[s.scheme == "none"].drop_duplicates(group)[group + ["width"]]
            .rename(columns={"width": "width_none"}))
    s = s.merge(none, on=group, how="left")
    s["narrowing"] = s.width_none - s.width
    skipped = df[~has][keys + INFO_COLS].drop_duplicates()
    s = pd.concat([s, skipped], ignore_index=True)
    s.to_csv(os.path.join(out, "binning_summary.csv"), index=False)
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="~/kallus")
    ap.add_argument("--reps", type=int, nargs="+", default=[0])
    ap.add_argument("--settings", nargs="+", default=["Black-White", "API-White"],
                    choices=list(GROUPS))
    ap.add_argument("--decisions", nargs="+", default=["model", "lender"],
                    choices=["model", "lender"])
    ap.add_argument("--versions", nargs="+", default=["exact", "split"],
                    choices=["exact", "split"])
    ap.add_argument("--county", nargs="+", default=COUNTY_DEFAULT)
    ap.add_argument("--income", nargs="+", default=INCOME_DEFAULT)
    ap.add_argument("--both", nargs="+", default=BOTH_DEFAULT)
    ap.add_argument("--lp-grid", type=int, default=21, help="grid over group masses (3+ groups)")
    ap.add_argument("--max-groups", type=int, default=3)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent
                                         / "experiment_results" / "hmda_binning"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    long_path = os.path.join(a.out, "binning_long.csv")

    schemes = ([("county", s) for s in a.county] + [("income", s) for s in a.income]
               + [("both", s) for s in a.both])
    for proxy, s in schemes:                  # fail early on a typo
        if proxy == "both" and "+" not in s:
            raise SystemExit(f"both scheme '{s}' must look like county+income, e.g. main+main")

    done = set()
    if os.path.exists(long_path):
        old = pd.read_csv(long_path)
        done = set(zip(old.rep, old.setting, old.decision, old.run, old.version, old.proxy,
                       old.scheme))

    d = load_hmda(a.repo)
    print(f"loaded {len(d):,} records")
    y_all = d.outcome.to_numpy(int)
    for rep in a.reps:
        split = split_data(d, rep)
        dec_model, _ = classifier_decision(d, split)
        decisions = {"model": dec_model, "lender": y_all}
        for setting in a.settings:
            groups = GROUPS[setting]
            keep = d.race.isin(groups).to_numpy()
            a_all = d.race.map({g: j for j, g in enumerate(groups)}).fillna(-1).to_numpy(int)
            sel = {p: keep & (split == p) for p in ("internal", "external")}
            n = {p: int(sel[p].sum()) for p in sel}
            for proxy, scheme in schemes:
                if len(groups) > 2 and pairs_only(proxy, scheme):
                    print(f"  skip {setting} {proxy} {scheme}: pairs only")
                    continue
                z = proxy_z(d, split, groups, proxy, scheme)
                z_all = np.full(len(d), -1)
                z_all[keep] = pd.factorize(z[keep], sort=True)[0]
                K = int(z_all.max()) + 1
                cnt = {p: np.bincount(z_all[sel[p]], minlength=K) for p in sel}
                info = dict(scheme=scheme, n_levels=K,
                            min_cell_internal=int(cnt["internal"].min()),
                            min_cell_external=int(cnt["external"].min()),
                            n_internal=n["internal"], n_external=n["external"])
                for decision in a.decisions:
                    dec_all = decisions[decision]
                    parts = {p: population(z_all[sel[p]], a_all[sel[p]], y_all[sel[p]],
                                           dec_all[sel[p]], K, len(groups)) for p in sel}
                    pooled = ((parts["internal"] * n["internal"]
                               + parts["external"] * n["external"]) / sum(n.values()))
                    for run in (["D"] if decision == "lender" else ["D", "DY"]):
                        todo = [v for v in a.versions if (rep, setting, decision, run, v, proxy,
                                                          scheme) not in done]
                        if not todo:
                            continue
                        q_int = joint_table(parts["internal"], run)
                        q_ext = joint_table(parts["external"], run)
                        q_pool = joint_table(pooled, run)
                        truth_pool = {k: float(v[0]) for k, v in metrics(q_pool[None], run).items()}
                        truth_int = {k: float(v[0]) for k, v in metrics(q_int[None], run).items()}
                        tables = {"exact": (q_pool.sum(2), q_pool.sum(1)),
                                  "split": harmonize(q_int.sum(2), q_ext.sum(1))}
                        for version in todo:
                            r, m = tables[version]
                            bad = (r.sum(1) > 0) & (m.sum(1) <= 0)
                            k = dict(zip(UNIT_KEYS, (decision, "external", rep, setting, proxy,
                                                     run, version)))
                            extra = dict(info, n_incompatible=int(bad.sum()),
                                         incompatible_mass=float(r.sum(1)[bad].sum()))
                            if bad.any():
                                sec = 0.0
                                rows = [dict(**k, method="Kallus", param="", metric="", side="",
                                             value=float("nan"), seconds=sec)]
                            else:
                                t0 = time.perf_counter()
                                b = kallus_refs(r, m, run, m.shape[1], a.max_groups, a.lp_grid)
                                sec = time.perf_counter() - t0
                                rows = bounds_rows(k, "Kallus", "", b, sec)
                            for row in rows:
                                row.update(extra)
                                row["truth_pooled"] = truth_pool.get(row.get("metric"), np.nan)
                                row["truth_internal"] = truth_int.get(row.get("metric"), np.nan)
                            append_csv(long_path, rows)
                            print(f"  rep {rep} {setting} {decision} {run} {version} {proxy} "
                                  f"{scheme}: {K} levels, {int(bad.sum())} incompatible, "
                                  f"{sec:.2f}s", flush=True)

    s = summarize(long_path, a.out)
    show = s[(s.version == "exact") & s.metric.notna()]
    for (dec, setting, metric), t in show.groupby(["decision", "setting", "metric"], sort=False):
        print(f"\n{dec}, {setting}, {metric}   truth = {t.truth_pooled.iloc[0]:.3f}")
        print(t[["proxy", "scheme", "n_levels", "min_cell_internal", "lower", "upper", "width",
                 "narrowing"]].round(3).to_string(index=False))
    print(f"\nsaved {long_path} and {os.path.join(a.out, 'binning_summary.csv')}")


if __name__ == "__main__":
    main()