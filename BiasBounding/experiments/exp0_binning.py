"""
Experiment 0 (supplement) - sharp bounds under different income and county binnings.

Two separate tables, one per proxy, each varying only that proxy's binning:
  income   number of quantile bins, or fixed income cut-offs (in $1000s)
  county   cut-offs on the county's share of the protected group (among the two groups),
           or quantile bins of that share
Bounds are the SHARP bounds (Kallus et al. closed form), on the same HMDA population as
exp0_hmda.py (empirical distribution, two groups at a time, label Y = outcome).

Metrics
  DD_lender   demographic disparity of the lender's decision (D = outcome). D is not a
              function of the proxy, so this is the clean measure of what a binning keeps.
  DD, TPRD, PPVD   disparities of the classifier D = 1[yhat1 >= t] used in exp0_hmda.py.
              The classifier is a function of county and income, so at fine resolutions its
              bounds shrink partly because D becomes (nearly) known given the proxy level.

Each row also reports the smallest proxy level (min_cell, in records): at fine resolutions
small levels have empirical P(A | Z) near 0 or 1, which narrows the bounds through sparsity
rather than information.

Scheme syntax (repeatable)
  none        one level (no proxy)
  q<k>        k quantile bins (income: of income; county: of the share, among the protected
              group's members, each county kept in one bin)
  cut:a,b,..  fixed cut-offs (income in $1000s, county as a share in [0, 1])
  full        every distinct value (every county / every income value), with the
              empirical P(A | Z) of each level (small levels make this near 0 or 1)
  fitted      no binning, as in Kallus et al.: every record is its own level, with P(A | Z)
              and P(D = 1 | Z) from the replication code's fitted models (county: the
              county's full-data race composition and approval rate; income: multinomial
              and logistic regressions on income). Only for DD_lender, since the repo has no
              fitted P(classifier decision, label | Z); other metrics are left empty.

Usage
  python exp0_binning_tables.py --repo ~/kallus
  python exp0_binning_tables.py --income none q2 q4 cut:50,100 cut:40,80,150 \
         --county none cut:0.15 cut:0.05,0.15,0.30 cut:0.1,0.3,0.5 q4 --metrics DD_lender DD
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "methods"))
from kallus_bounds import closed_form_bounds  # noqa: E402

from exp0_hmda import county_share, load_hmda, population, qbins  # noqa: E402

INTERNAL = [(0, 0), (0, 1), (1, 0), (1, 1)]                 # (decision, label)
RATES = {
    "DD":   (lambda d, y: d == 1,            lambda d, y: True),
    "TPRD": (lambda d, y: d == 1 and y == 1, lambda d, y: y == 1),
    "PPVD": (lambda d, y: d == 1 and y == 1, lambda d, y: d == 1),
}
INCOME_DEFAULT = ["none", "q2", "q3", "q4", "q8", "q16", "cut:50,100", "cut:40,80,150", "full",
                  "fitted"]
COUNTY_DEFAULT = ["none", "cut:0.10", "cut:0.15", "cut:0.30", "cut:0.05,0.15,0.30",
                  "cut:0.05,0.10,0.20", "cut:0.10,0.30,0.50", "q2", "q4", "q8", "full", "fitted"]
FITTED_FILE = {"income": "small_proxy_income.csv", "county": "small_proxy_county.csv"}


def load_fitted(repo, d):
    """Attach the repo's fitted P(approved | Z) and P(race | Z) for each proxy, by record id."""
    base = os.path.join(os.path.expanduser(repo), "HMDA")
    for proxy, f in FITTED_FILE.items():
        fit = pd.read_csv(os.path.join(base, f), usecols=["id", "yhat1", "White", "Black", "API"])
        fit = fit.rename(columns={c: f"fit_{proxy}_{c}" for c in fit.columns if c != "id"})
        d = d.merge(fit, on="id", how="left", validate="one_to_one")
    return d


def fitted_bounds(d, proxy, g0, g1):
    """
    Kallus closed form with every record as its own proxy level (weight 1/n):
    P(D = 1 | z_i) = fitted approval probability, P(A = g1 | z_i) = fitted share of g1 among
    {g0, g1}. Equivalent to the paper's plug-in: lower P(D=1, A=1) = mean max(0, eta_D + eta_A - 1),
    upper = mean min(eta_D, eta_A).
    """
    eta_d = d[f"fit_{proxy}_yhat1"].to_numpy(float)
    p1, p0 = (d[f"fit_{proxy}_{g}"].to_numpy(float) for g in (g1, g0))
    eta_a = p1 / (p1 + p0)
    n = len(d)
    r = np.stack([1 - eta_d, eta_d], 1) / n                     # [record, d]
    m = np.stack([1 - eta_a, eta_a], 1) / n                     # [record, a]
    return closed_form_bounds(r, m, np.array([False, True]), np.array([True, True]))


def income_levels(d, scheme):
    if scheme == "none":
        return np.zeros(len(d), int)
    if scheme == "full":
        return pd.factorize(d.income, sort=True)[0]
    if scheme.startswith("q"):
        return qbins(d.income, int(scheme[1:]))
    if scheme.startswith("cut:"):
        return np.digitize(d.income, [float(c) for c in scheme[4:].split(",")])
    raise ValueError(f"bad income scheme {scheme}")


def county_levels(d, scheme, g0, g1):
    if scheme == "none":
        return np.zeros(len(d), int)
    if scheme == "full":
        return pd.factorize(d.county_id, sort=True)[0]
    s = county_share(d, g0, g1)
    if scheme.startswith("q"):
        k = int(scheme[1:])
        cuts = np.unique(np.quantile(s[(d.race == g1).to_numpy()], np.linspace(0, 1, k + 1)[1:-1]))
    elif scheme.startswith("cut:"):
        cuts = [float(c) for c in scheme[4:].split(",")]
    else:
        raise ValueError(f"bad county scheme {scheme}")
    return np.digitize(s, cuts)


def bounds(p, metric):
    t, c = RATES[metric]
    r = np.stack([p[:, :, y, d].sum(1) for d, y in INTERNAL], 1)
    m = p.sum((2, 3))
    target = np.array([bool(t(d, y)) for d, y in INTERNAL])
    cond = np.array([bool(c(d, y)) for d, y in INTERNAL])
    lo, hi = closed_form_bounds(r, m, target, cond)
    mu = [sum(p[:, a, y, d].sum() for d, y in INTERNAL if t(d, y))
          / sum(p[:, a, y, d].sum() for d, y in INTERNAL if c(d, y)) for a in (0, 1)]
    return lo, hi, mu[1] - mu[0]


def table(d_all, pairs, proxy, schemes, metrics):
    rows = []
    for g0, g1 in pairs:
        d = d_all[d_all.race.isin((g0, g1))].reset_index(drop=True)
        a = (d.race == g1).to_numpy(int)
        y = d.outcome.to_numpy(int)
        model = (d.yhat1 >= np.quantile(d.yhat1, 1 - y.mean())).to_numpy(int)
        for scheme in schemes:
            if scheme == "fitted":
                truth = bounds(population(d, np.zeros(len(d), int), a, y, y), "DD")[2]
                for metric in metrics:
                    lo, hi = (fitted_bounds(d, proxy, g0, g1) if metric == "DD_lender"
                              else (np.nan, np.nan))
                    if metric != "DD_lender":
                        truth_m = np.nan
                    else:
                        truth_m = truth
                    rows.append(dict(proxy=proxy, groups=f"{g1}-{g0}", metric=metric,
                                     scheme=scheme, levels=len(d), min_cell=np.nan,
                                     truth=truth_m, lower=lo, upper=hi, width=hi - lo))
                continue
            z = (income_levels(d, scheme) if proxy == "income"
                 else county_levels(d, scheme, g0, g1))
            z = pd.factorize(z, sort=True)[0]
            sizes = np.bincount(z)
            for metric in metrics:
                if metric == "DD_lender":
                    p = population(d, z, a, y, y)
                    lo, hi, truth = bounds(p, "DD")
                else:
                    p = population(d, z, a, y, model)
                    lo, hi, truth = bounds(p, metric)
                rows.append(dict(proxy=proxy, groups=f"{g1}-{g0}", metric=metric,
                                 scheme=scheme, levels=len(sizes), min_cell=int(sizes.min()),
                                 truth=truth, lower=lo, upper=hi, width=hi - lo))
    res = pd.DataFrame(rows)
    none = res[res.scheme == "none"].set_index(["groups", "metric"]).width
    res["narrowing"] = [none[(g, m)] - w for g, m, w in zip(res.groups, res.metric, res.width)]
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="~/kallus")
    ap.add_argument("--pairs", nargs="+", default=["White,Black", "White,API"])
    ap.add_argument("--income", nargs="+", default=INCOME_DEFAULT, help="income schemes")
    ap.add_argument("--county", nargs="+", default=COUNTY_DEFAULT, help="county schemes")
    ap.add_argument("--metrics", nargs="+", default=["DD_lender", "DD", "TPRD", "PPVD"],
                    choices=["DD_lender", "DD", "TPRD", "PPVD"])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent
                                         / "experiment_results" / "exp0_binning_tables"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for s in a.income + a.county:
        if not (s in ("none", "full", "fitted") or s[0] == "q" or s.startswith("cut:")):
            raise SystemExit(f"bad scheme '{s}' (use none, full, fitted, q<k> or cut:a,b,...)")
    if "none" not in a.income:
        a.income = ["none"] + a.income
    if "none" not in a.county:
        a.county = ["none"] + a.county

    d = load_fitted(a.repo, load_hmda(a.repo))
    pairs = [tuple(p.split(",")) for p in a.pairs]
    for proxy, schemes in (("income", a.income), ("county", a.county)):
        res = table(d, pairs, proxy, schemes, a.metrics)
        res.to_csv(os.path.join(a.out, f"{proxy}_binning.csv"), index=False)
        print(f"\n=== {proxy.upper()} binning (sharp bounds; narrowing = width(none) - width) ===")
        for (g, m), t in res.groupby(["groups", "metric"], sort=False):
            print(f"\n{g.replace('-', ' vs ')}, {m}   truth = {t.truth.iloc[0]:.3f}")
            t = t[t.lower.notna()]
            print(t[["scheme", "levels", "min_cell", "lower", "upper", "width", "narrowing"]]
                  .round(3).to_string(index=False))
    print(f"\nsaved {a.out}/income_binning.csv and county_binning.csv")


if __name__ == "__main__":
    main()