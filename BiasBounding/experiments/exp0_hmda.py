"""
Experiment 0 on HMDA - grid method vs Kallus et al. (2022) sharp bounds on real data.

Data: the 1% HMDA 2011-12 sample from the Kallus replication repo
  <repo>/HMDA/small_proxy_county_income.csv   id, county_id, outcome (1 = approved), race,
                                               yhat1, applicant_income_000s, ...
  <repo>/HMDA/small_proxy_county.csv          same ids; White/Black/API = the county's racial
                                               composition computed on the FULL HMDA data

Design
  * The empirical distribution of the sample, restricted to two groups, is the POPULATION:
    primary and auxiliary tables come from the same records (race used only on the auxiliary
    side), so both methods see exactly the same P(Z) and P(A | Z) and the true disparity is the
    one computed with race. This isolates the methods; the realistic split is Experiment 1.
  * Two groups, because Kallus' closed forms are sharp only for binary A
    (default pairs: White vs Black, White vs API; group 1 = the second, so mu_1 - mu_0 is
    e.g. P(approved | Black) - P(approved | White)).
  * Proxies, coarsened so that the grid is feasible:
      income   quantile bins of applicant income (equal numbers of records)
      county   counties binned by their full-data share of group 1 (among the two groups)
               at fixed cut-offs, so that high-minority counties form their own bin. Equal-
               count bins average them away: a level only narrows the bounds where the
               group's share exceeds the approval or denial rate there, and with equal-count
               bins no level does (the bounds then equal the no-proxy bounds).
               --county-binning quantile gives the equal-count version for comparison.
      both     county bins x income bins
    A "none" row (one proxy level) gives the no-proxy bounds as a reference.
  * Label Y = outcome (approved). Decision D (default --decision model) = a classifier,
    D = 1[yhat1 >= t], with yhat1 the repo's per-county logistic fit of outcome on income and
    t set so that D approves as many applicants as there are positive labels. The proxy bins
    are coarser than the classifier's inputs, so D varies within a proxy level (if D were a
    function of the proxy, D's disparity would be point identified).
    --decision lender instead takes D = outcome (the paper's setup for demographic disparity).
  * Part A  demographic disparity of D (internal = D).
    Part B  TPR / PPV disparity of D w.r.t. Y (internal = (D, Y)).

Usage
  python exp0_hmda.py                                  # repo in ~/kallus
  python exp0_hmda.py --repo ~/kallus --pairs White,Black --nums-dd 2 3 5 9 17 33
  python exp0_hmda.py --county-binning quantile            # equal-count county bins
  python exp0_hmda.py --decision lender                    # D = outcome, as in the paper
Layout: methods/ (grid_joint.py, kallus_bounds.py), experiments/ (this file and
exp0_validation.py); results in experiment_results/exp0_hmda/.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from exp0_synth import add_gaps, check, plot, run_pops


# ------------------------------------------------------------------------- data
def load_hmda(repo):
    base = os.path.join(os.path.expanduser(repo), "HMDA")
    f_ci, f_c = (os.path.join(base, f) for f in ("small_proxy_county_income.csv",
                                                  "small_proxy_county.csv"))
    for f in (f_ci, f_c):
        if not os.path.exists(f):
            raise SystemExit(f"cannot find {f}; pass --repo with the path of the cloned repo")
    d = pd.read_csv(f_ci, usecols=["id", "county_id", "outcome", "race", "yhat1",
                                   "applicant_income_000s"])
    shares = (pd.read_csv(f_c, usecols=["id", "White", "Black", "API"])
                .rename(columns=lambda c: c if c == "id" else f"share_{c}"))
    d = d.merge(shares, on="id", how="left", validate="one_to_one")
    if d[["share_White", "share_Black", "share_API"]].isna().any().any():
        raise SystemExit("some records have no county composition in small_proxy_county.csv")
    return d.rename(columns={"applicant_income_000s": "income"})


def qbins(x, k):
    """k quantile bins (equal numbers of records), as integer codes 0..k-1."""
    codes = pd.qcut(pd.Series(x).rank(method="first"), k, labels=False)
    return codes.to_numpy(int)


# share cut-offs (group 1 among the two groups) per number of county levels
COUNTY_CUTS = {"Black": {2: [0.15], 4: [0.05, 0.15, 0.30]},
               "API":   {2: [0.10], 4: [0.05, 0.10, 0.20]}}


def county_share(d, g0, g1):
    """Full-data share of g1 among {g0, g1} in each record's county."""
    return (d[f"share_{g1}"] / (d[f"share_{g1}"] + d[f"share_{g0}"])).to_numpy()


def county_codes(d, levels, g0, g1, binning, cuts=None):
    s = county_share(d, g0, g1)
    if binning == "share":
        cuts = cuts or COUNTY_CUTS.get(g1, {}).get(levels)
        if cuts is None:                  # no preset: cut-offs at quantiles of g1's members
            cuts = np.quantile(s[d.race == g1], np.linspace(0, 1, levels + 1)[1:-1])
        return np.digitize(s, cuts)
    # equal-count bins, keeping each county in one bin
    county_s = pd.Series(s).groupby(d.county_id.to_numpy()).first().sort_values(kind="stable")
    w = d.county_id.value_counts().reindex(county_s.index)
    cum = (w.cumsum() / w.sum()).to_numpy()
    county_bin = pd.Series(np.minimum((cum * levels - 1e-12).astype(int), levels - 1),
                           index=county_s.index)
    return d.county_id.map(county_bin).to_numpy(int)


def proxy_codes(d, proxy, levels, g0, g1, binning):
    """Integer proxy level per record. For 'both', levels = (county levels, income levels)."""
    if proxy == "none":
        return np.zeros(len(d), int)
    if proxy == "income":
        return qbins(d.income, levels)
    if proxy == "county":
        return county_codes(d, levels, g0, g1, binning)
    if proxy == "both":
        lc, li = levels
        return county_codes(d, lc, g0, g1, binning) * li + qbins(d.income, li)
    raise ValueError(proxy)


def population(d, z, a, y, dec):
    """Exact joint p[z, a, y, d] of the records (empirical distribution)."""
    K = z.max() + 1
    p = np.zeros((K, 2, 2, 2))
    np.add.at(p, (z, a, y, dec), 1.0)
    return p / p.sum()


def build_pops(d_all, pairs, part, proxies, binning, decision):
    """proxies: list of (proxy, levels). Returns [(name, p, info)]."""
    pops = []
    for pair in pairs:
        g0, g1 = pair
        d = d_all[d_all.race.isin(pair)].reset_index(drop=True)
        a = (d.race == g1).to_numpy(int)
        y = d.outcome.to_numpy(int)                                  # label
        if decision == "model":
            t = np.quantile(d.yhat1, 1 - y.mean())                   # same positive rate
            dec = (d.yhat1 >= t).to_numpy(int)
        else:
            dec = y
        for proxy, lev in proxies:
            z = proxy_codes(d, proxy, lev, g0, g1, binning)
            z = pd.factorize(z, sort=True)[0]                        # drop empty levels
            p = population(d, z, a, y, dec)
            lev_str = f"{lev[0]}x{lev[1]}" if proxy == "both" else str(lev)
            name = f"{g1} vs {g0}, {proxy} ({lev_str})"
            info = dict(groups=f"{g1}-{g0}", proxy=proxy, levels=lev_str, n=len(d),
                        binning=binning if proxy in ("county", "both") else "",
                        decision=decision)
            pops.append((name, p, info))
            pz = p.sum((1, 2, 3)); pa = p.sum((2, 3))[:, 1] / pz
            print(f"  {name:36s} n={len(d):,}  P(Z) = {np.round(pz, 3)}  "
                  f"P({g1} | Z) = {np.round(pa, 3)}")
    return pops


def summarise_by_pop(res):
    g = res[res.method.str.startswith("grid")].copy()
    g["covers_truth"] = g.covers_truth.where(g.lower.notna())
    g["gap_max"] = g[["gap_lower", "gap_upper"]].max(axis=1)
    keep = ["part", "groups", "proxy", "levels", "metric", "method", "num"]
    out = g[keep + ["truth", "sharp_lower", "sharp_upper", "lower", "upper", "gap_max",
                    "covers_truth", "n_joints", "seconds"]]
    return out.sort_values(keep).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="~/kallus", help="cloned FairnessWithUnobservedProtectedClass")
    ap.add_argument("--pairs", nargs="+", default=["White,Black", "White,API"],
                    help="reference,protected (mu_protected - mu_reference)")
    ap.add_argument("--decision", choices=["model", "lender"], default="model",
                    help="D = thresholded classifier (default) or D = outcome (paper's DD)")
    ap.add_argument("--county-binning", choices=["share", "quantile"], default="share")
    ap.add_argument("--levels-dd", type=int, default=4, help="proxy levels in part A")
    ap.add_argument("--levels-rates", type=int, default=2, help="proxy levels in part B")
    ap.add_argument("--both-dd", type=int, nargs=2, default=[4, 2], metavar=("COUNTY", "INCOME"),
                    help="levels of 'both' in part A (default 4 x 2 = 8; num <= 6 feasible)")
    ap.add_argument("--both-rates", type=int, nargs=2, default=[2, 2], metavar=("COUNTY", "INCOME"))
    ap.add_argument("--nums-dd", type=int, nargs="+", default=[2, 3, 5, 9, 17, 33, 65])
    ap.add_argument("--nums-rates", type=int, nargs="+", default=[2, 3, 5, 9, 17])
    ap.add_argument("--methods", nargs="+", default=["sweep", "sequential"])
    ap.add_argument("--max-joints", type=int, default=5_000_000)
    ap.add_argument("--no-lp", action="store_true")
    ap.add_argument("--skip-part", choices=["A", "B"])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent
                                         / "experiment_results" / "exp0_hmda"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    d = load_hmda(a.repo)
    print(f"loaded {len(d):,} records; race counts {d.race.value_counts().to_dict()}")
    pairs = [tuple(p.split(",")) for p in a.pairs]
    for p in pairs:
        if len(p) != 2 or not set(p) <= set(d.race.unique()):
            raise SystemExit(f"bad pair {p}; groups are {sorted(d.race.unique())}")

    def proxies(levels, both):
        return [("none", 1), ("income", levels), ("county", levels), ("both", tuple(both))]

    dname = "a classifier" if a.decision == "model" else "the lender's decision"
    rows = []
    if a.skip_part != "A":
        print(f"\nPart A: demographic disparity of {dname}")
        pops = build_pops(d, pairs, "A", proxies(a.levels_dd, a.both_dd), a.county_binning,
                          a.decision)
        rows += run_pops("A", pops, a.nums_dd, a.methods, ["DD"], a.max_joints, not a.no_lp)
    if a.skip_part != "B":
        print(f"\nPart B: TPR / PPV disparity of {dname}, label = outcome")
        pops = build_pops(d, pairs, "B", proxies(a.levels_rates, a.both_rates),
                          a.county_binning, a.decision)
        rows += run_pops("B", pops, a.nums_rates, a.methods, ["TPRD", "PPVD"], a.max_joints,
                         not a.no_lp)

    res = add_gaps(pd.DataFrame(rows))
    res.to_csv(os.path.join(a.out, "results.csv"), index=False)
    summ = summarise_by_pop(res)
    summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    names = list(dict.fromkeys(res["pop"]))
    plot(res, a.out, pop=names[0])
    for k, name in enumerate(names[1:], 1):
        plot(res, a.out, pop=name, suffix=f"_{k}")

    print("\n" + "\n".join(check(res)))
    ref = res[res.method == "kallus_closed_form"][["part", "groups", "proxy", "levels",
                                                    "metric", "truth", "lower", "upper",
                                                    "width"]]
    with pd.option_context("display.width", 180, "display.max_rows", 400):
        print("\nKallus sharp bounds:\n" + ref.round(4).to_string(index=False))
        best = (summ[summ.lower.notna()].sort_values("num")
                .groupby(["part", "groups", "proxy", "metric", "method"]).tail(1))
        print("\nGrid at the largest feasible num:\n" +
              best.drop(columns=["sharp_lower", "sharp_upper"]).round(4).to_string(index=False))
    print(f"\nsaved results to {a.out}/ (convergence*.png: one per population)")


if __name__ == "__main__":
    main()