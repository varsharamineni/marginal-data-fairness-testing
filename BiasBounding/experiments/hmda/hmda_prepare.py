"""
Step 0 of the HMDA full experiment: build every unit's tables and true values.

For each repetition (random split, seed = rep) of the 1% HMDA 2011-12 sample (149,032 records):
  train 20%     fits the classifier (decision = model) and nothing else
  internal 40%  the lender's data:  D, Y, Z   (race withheld)
  external 40%  census-like data:   Z, A      (D and Y withheld)

Decision D
  model    logistic regression of approval on log income and the county's approval rate
           (smoothed towards the overall rate with weight 20, from the train split only),
           thresholded so that it approves the same share of internal applicants as the lender
  lender   the lender's decision (D = Y); only internal D is defined
Label Y = the lender's decision (outcome).

Groups (A = 0, 1, ... in this order): Black-White (White, Black), API-White (White, API),
three (White, Black, API). Metrics compare group 1 with group 0 where a pair is needed
(Black vs White in 'three'); worst-case DI and the maximum gap use all groups.

Proxy levels Z
  county   share of each minority group among the setting's groups in the applicant's county
           pairs: 2 levels, share >= 0.30 (Black) or >= 0.15 (API)
           three: 3 levels, Black share >= 0.30, else API share >= 0.15, else other
           cut_source external: shares counted in the external split (counties without
             external applicants go to the lowest level)
           cut_source full: the repo's full-data county composition
  income   2 levels, split at the median income (of the external split, or of all records
           for cut_source full)
  both     county level x income level

Tables (internal variable D: i = D; DY: i = TP, FP, FN, TN as in exp_synthetic.py)
  version split  r = p(z, i) from the internal split, m = p(z, a) from the external split
                 rescaled to the internal p(z)
  version exact  r and m both from internal + external pooled (the truth lies in the sharp set)
True values: truth_pooled (internal + external, with race) and truth_internal (internal only).

Usage
  python hmda_prepare.py --repo ~/kallus --reps 0           # first repetition
  python hmda_prepare.py --repo ~/kallus --reps 1 2 3 4     # add repetitions later
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


from hmda_common import CHOICES, PREPARED, UNIT_KEYS  # noqa: F401  (sets sys.path)
from ci_baseline import harmonize  # noqa: E402
from exp0_hmda import load_hmda  # noqa: E402
from synth import joint_table, metrics  # noqa: E402

GROUPS = {"Black-White": ["White", "Black"], "API-White": ["White", "API"],
          "three": ["White", "Black", "API"]}
CUT = {"Black": 0.30, "API": 0.15}
FRACS = (0.2, 0.4, 0.4)


def split_data(d, rep):
    rng = np.random.default_rng(rep)
    u = rng.permutation(len(d)) / len(d)
    return np.where(u < FRACS[0], "train", np.where(u < FRACS[0] + FRACS[1], "internal",
                                                     "external"))


def classifier_decision(d, split):
    """Logistic regression on log income and the smoothed county approval rate (train only)."""
    tr = d[split == "train"]
    prior, k = tr.outcome.mean(), 20.0
    st = tr.groupby("county_id").outcome.agg(["sum", "count"])
    rate = d.county_id.map((st["sum"] + k * prior) / (st["count"] + k)).fillna(prior)
    X = np.column_stack([np.log1p(d.income), rate])
    clf = LogisticRegression().fit(X[split == "train"], tr.outcome)
    score = clf.predict_proba(X)[:, 1]
    internal = split == "internal"
    t = np.quantile(score[internal], 1 - d.outcome[internal].mean())
    return (score >= t).astype(int)


def county_shares(d, split, groups, source):
    """Per-record county share of each group among `groups`."""
    if source == "full":
        s = d[[f"share_{g}" for g in groups]].to_numpy(float)
        tot = s.sum(1, keepdims=True)
        return pd.DataFrame(np.where(tot > 0, s / np.where(tot > 0, tot, 1), 0), columns=groups)
    ext = d[(split == "external") & d.race.isin(groups)]
    counts = ext.groupby(["county_id", "race"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=groups, fill_value=0)
    shares = counts.div(counts.sum(axis=1), axis=0)
    return pd.DataFrame({g: d.county_id.map(shares[g]).fillna(0).to_numpy() for g in groups})


def proxy_levels(d, split, groups, source):
    sh = county_shares(d, split, groups, source)
    if len(groups) == 2:
        county = (sh[groups[1]] >= CUT[groups[1]]).astype(int).to_numpy()
    else:
        county = np.where(sh["Black"] >= CUT["Black"], 1, np.where(sh["API"] >= CUT["API"], 2, 0))
    ref = d.income[split == "external"] if source == "external" else d.income
    income = (d.income > np.median(ref)).astype(int).to_numpy()
    K_county = 2 if len(groups) == 2 else 3
    return {"county": (county, K_county), "income": (income, 2),
            "both": (county * 2 + income, 2 * K_county)}


def population(z, a, y, dec, K, G):
    p = np.zeros((K, G, 2, 2))
    np.add.at(p, (z, a, y, dec), 1.0)
    return p / p.sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="~/kallus")
    ap.add_argument("--reps", type=int, nargs="+", default=[0])
    a = ap.parse_args()
    os.makedirs(PREPARED.parent, exist_ok=True)
    units = {}
    if PREPARED.exists():
        with open(PREPARED, "rb") as f:
            units = pickle.load(f)
    d = load_hmda(a.repo)
    print(f"loaded {len(d):,} records")
    skipped = []
    for rep in a.reps:
        split = split_data(d, rep)
        decisions = {"model": classifier_decision(d, split), "lender": d.outcome.to_numpy(int)}
        print(f"rep {rep}: " + ", ".join(f"{s} {(split == s).sum():,}" for s in
                                         ("train", "internal", "external")))
        for decision, dec_all in decisions.items():
            for source in CHOICES["cut_source"]:
                for setting, groups in GROUPS.items():
                    keep = d.race.isin(groups).to_numpy()
                    a_all = d.race.map({g: j for j, g in enumerate(groups)}).to_numpy()
                    levels = proxy_levels(d, split, groups, source)
                    y_all = d.outcome.to_numpy(int)
                    for proxy, (z_all, K) in levels.items():
                        parts = {}
                        for part in ("internal", "external"):
                            sel = keep & (split == part)
                            parts[part] = population(z_all[sel], a_all[sel].astype(int),
                                                     y_all[sel], dec_all[sel], K, len(groups))
                        n = {part: int((keep & (split == part)).sum()) for part in parts}
                        pooled = (parts["internal"] * n["internal"]
                                  + parts["external"] * n["external"]) / sum(n.values())
                        for run in CHOICES["run"]:
                            if decision == "lender" and run == "DY":
                                continue
                            q_int, q_ext = joint_table(parts["internal"], run), \
                                joint_table(parts["external"], run)
                            q_pool = joint_table(pooled, run)
                            truth_pool = {k: float(v[0]) for k, v in metrics(q_pool[None], run).items()}
                            truth_int = {k: float(v[0]) for k, v in metrics(q_int[None], run).items()}
                            tables = {"exact": (q_pool.sum(2), q_pool.sum(1)),
                                      "split": harmonize(q_int.sum(2), q_ext.sum(1))}
                            for version, (r, m) in tables.items():
                                key = (decision, source, rep, setting, proxy, run, version)
                                # a proxy level with internal records but no external records
                                # (or vice versa) makes the margins incompatible
                                bad = (r.sum(1) > 0) & (m.sum(1) <= 0)
                                if bad.any():
                                    skipped.append(key)
                                    continue
                                units[key] = dict(r=r, m=m, truth_pooled=truth_pool,
                                                  truth_internal=truth_int, n=n, K=K,
                                                  groups=groups)
    with open(PREPARED, "wb") as f:
        pickle.dump(units, f)
    rows = [dict(zip(UNIT_KEYS, k), K=u["K"], n_internal=u["n"]["internal"],
                 n_external=u["n"]["external"]) for k, u in units.items()]
    pd.DataFrame(rows).to_csv(PREPARED.parent / "units.csv", index=False)
    print(f"{len(units)} units saved to {PREPARED}")
    if skipped:
        print(f"skipped {len(skipped)} units with incompatible margins: {skipped[:5]} ...")


if __name__ == "__main__":
    main()