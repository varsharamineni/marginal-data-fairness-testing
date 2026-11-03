"""
Experiment 1 - HMDA with three protected groups (White, Black, API) at once.

Question: with a realistic split into an internal and an external dataset, what can each
method say about disparities between three groups, including metrics that the method of
Kallus et al. (2022) cannot bound?

Data (1% HMDA 2011-12 sample from the Kallus replication repo), split at random into
  train      20%  used only to fit the classifier
  primary    40%  internal data: decision D, label Y, proxy Z      (race hidden)
  auxiliary  40%  external data: proxy Z, race A                    (D, Y hidden)
Y = outcome (approved). D = a logistic-regression classifier of Y on log income and the
county's approval rate (target-encoded on the training split), thresholded so that it approves
as many primary applicants as are approved. The truth is computed on the primary split with
its (hidden) race. Both methods get the same inputs: the auxiliary P(A | Z) is rescaled to the
primary P(Z) (Kallus' convention), so any truth outside an interval is sampling noise between
the two splits.

Proxies
  none     one level
  income   2 quantile bins (income is uninformative about race at any binning, see exp0)
  county   3 levels: counties with Black share >= 0.30, else API share >= 0.15, else other
           (full-data county composition; cut-offs via --county-cuts)
  both     county (3) x income (2) = 6 levels

Runs (each metric uses only the internal variable it needs, to keep enumeration feasible)
  A  internal = D                    2 free entries per proxy level
     group rates P(D=1 | A), demographic disparity (DD) for all pairs, disparate impact (DI)
     ratios Black/White and API/White with the four-fifths rule, maximum pairwise DD gap,
     I(D; A) and I(D; A | Z)
  B  internal = {(D=1,Y=1), (D=1,Y=0), D=0}   4 free entries per level (target cell gridded)
     PPV per group, predictive parity difference (PPVD) for all pairs, maximum PPV gap
  C  internal = (D, Y)               6 free entries per level (smaller proxy settings only)
     TPR and FPR differences, accuracy difference (references with sharp Kallus bounds);
     equalized-odds gap max over pairs of max(|dTPR|, |dFPR|); balanced-accuracy and F1
     differences; separation I(D; A | Y) and sufficiency I(Y; A | D)
Mutual information is in bits.

Added nonlinear / nonconvex measures (none has a Kallus counterpart unless stated)
  A  worst-case DI  min_g mu_g / max_g mu_g  (four-fifths rule across all three groups)
     odds ratio     [mu_g/(1-mu_g)] / [mu_W/(1-mu_W)]  for Black and API vs White
  C  equalized-odds gap (Kallus: sharp upper bound from separate TPR/FPR LPs, lower outer)
     balanced accuracy  BA_g = (TPR_g + TNR_g)/2, difference vs White
     F1             F1_g = 2TP/(2TP+FP+FN) within group g, difference vs White
     I(Y; A | D)    sufficiency (calibration-style) in information form

Kallus et al. (kallus_bounds.py), where available
  DD, PPVD pairs       lp_bounds (sharp, 3 groups, one difference at a time)
  DI                   closed_form_bounds(ratio=True) on the one-vs-rest collapse only, i.e.
                       P(D=1 | g) / P(D=1 | not g): a different estimand from g / White
  maximum gap          only an outer bound assembled from the per-pair LP intervals
  mutual information   not available

Outputs (experiment_results/exp1_hmda/): results.csv (long format), summary.csv, and a printed
summary per proxy setting.

Usage
  python exp1_hmda.py --repo ~/kallus
  python exp1_hmda.py --nums-a 2 3 5 9 --nums-b 2 3 5 --nums-c 2 3 --methods sequential sweep
"""
import argparse
import contextlib
import io
import itertools
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "methods"))
from grid_joint import solve_joint_distribution  # noqa: E402
from kallus_bounds import closed_form_bounds, lp_bounds, one_vs_rest  # noqa: E402

from exp0_hmda import load_hmda, qbins  # noqa: E402

GROUPS = ["White", "Black", "API"]                   # A = 0, 1, 2
PAIRS = [(1, 0), (2, 0), (2, 1)]                     # (g, h): mu_g - mu_h
PAIR_NAME = {p: f"{GROUPS[p[0]]}-{GROUPS[p[1]]}" for p in PAIRS}
# internal coding per run: list of (d, y) cells merged into each internal value
INTERNAL = {
    "A": [[(0, 0), (0, 1)], [(1, 0), (1, 1)]],                        # D = 0, D = 1
    "B": [[(1, 1)], [(1, 0)], [(0, 0), (0, 1)]],                      # target first
    "C": [[(0, 0)], [(0, 1)], [(1, 0)], [(1, 1)]],                    # (D, Y)
}
EPS = 1e-15


# ------------------------------------------------------------------------- data
def prepare(repo, seed, fracs, county_cuts, income_bins):
    d = load_hmda(repo)
    d["A"] = d.race.map({g: k for k, g in enumerate(GROUPS)})
    rng = np.random.default_rng(seed)
    u = rng.uniform(size=len(d))
    d["split"] = np.where(u < fracs[0], "train", np.where(u < fracs[0] + fracs[1], "primary", "aux"))

    # classifier: logistic regression on log income and the county's (training) approval rate
    tr = d[d.split == "train"]
    prior, k = tr.outcome.mean(), 20.0
    st = tr.groupby("county_id").outcome.agg(["sum", "count"])
    enc = (st["sum"] + k * prior) / (st["count"] + k)                 # smoothed target encoding
    d["county_rate"] = d.county_id.map(enc).fillna(prior)
    X = np.column_stack([np.log1p(d.income), d.county_rate])
    clf = LogisticRegression().fit(X[d.split == "train"], tr.outcome)
    d["score"] = clf.predict_proba(X)[:, 1]
    pri = d.split == "primary"
    t = np.quantile(d.score[pri], 1 - d.outcome[pri].mean())
    d["D"] = (d.score >= t).astype(int)
    d["Y"] = d.outcome.astype(int)

    # proxies
    cb, ca = county_cuts
    d["z_county"] = np.where(d.share_Black >= cb, 1, np.where(d.share_API >= ca, 2, 0))
    d["z_income"] = qbins(d.income, income_bins)
    d["z_both"] = d.z_county * income_bins + d.z_income
    d["z_none"] = 0
    return d


def tables(d, proxy, run):
    """Primary r[z, i], auxiliary m[z, a] (rescaled to the primary P(Z)), truth p[z, a, i]."""
    z = f"z_{proxy}"
    pri, aux = d[d.split == "primary"], d[d.split == "aux"]
    levels = np.sort(pd.unique(d[z]))
    zmap = {v: k for k, v in enumerate(levels)}
    K, I = len(levels), len(INTERNAL[run])
    cell = {dy: i for i, cells in enumerate(INTERNAL[run]) for dy in cells}
    truth = np.zeros((K, 3, I))
    for (zz, a, dd, yy), n in pri.groupby([z, "A", "D", "Y"]).size().items():
        truth[zmap[zz], a, cell[(dd, yy)]] += n
    truth /= truth.sum()
    r = truth.sum(1)                                                  # [z, i]
    m = np.zeros((K, 3))
    for (zz, a), n in aux.groupby([z, "A"]).size().items():
        m[zmap[zz], a] += n
    pz = r.sum(1)
    m = m / np.maximum(m.sum(1, keepdims=True), EPS) * pz[:, None]   # P_aux(A|Z) P_pri(Z)
    return r, m, truth


# ------------------------------------------------------------------------- metrics on joints
def mi(p, axis_x, axis_y, cond_axes=()):
    """Mutual information (bits) between two axes of p [N, ...], conditional on cond_axes."""
    keep = lambda axes: tuple(a for a in range(1, p.ndim) if a not in axes)
    pxyz = p
    pxz = p.sum(axis=keep((axis_x,) + tuple(cond_axes)), keepdims=True)
    pyz = p.sum(axis=keep((axis_y,) + tuple(cond_axes)), keepdims=True)
    pz = p.sum(axis=keep(tuple(cond_axes)), keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(pxyz > EPS, pxyz * np.log2(pxyz * pz / (pxz * pyz)), 0.0)
    return term.reshape(len(p), -1).sum(1)


def metrics_A(J):
    """J [N, z, i=D, a] -> dict of metric arrays."""
    th = J.sum(1)                                                     # [N, d, a]
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = th[:, 1, :] / th.sum(1)                                # P(D=1 | A)
    out = {f"P(D=1|{GROUPS[g]})": rate[:, g] for g in range(3)}
    for g, h in PAIRS:
        out[f"DD {PAIR_NAME[(g, h)]}"] = rate[:, g] - rate[:, h]
    for g in (1, 2):
        out[f"DI {GROUPS[g]}/White"] = rate[:, g] / rate[:, 0]
    out["max DD gap"] = np.max(np.abs(np.stack([rate[:, g] - rate[:, h] for g, h in PAIRS])), 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["worst-case DI (min/max)"] = rate.min(1) / rate.max(1)
        odds = rate / (1 - rate)
        for g in (1, 2):
            out[f"OR {GROUPS[g]}/White"] = odds[:, g] / odds[:, 0]
    out["I(D;A)"] = mi(th, 1, 2)
    out["I(D;A|Z)"] = mi(J, 2, 3, cond_axes=(1,))
    return out


def metrics_B(J):
    th = J.sum(1)                                                     # [N, (11, 10, D0), a]
    with np.errstate(divide="ignore", invalid="ignore"):
        ppv = th[:, 0, :] / (th[:, 0, :] + th[:, 1, :])
    out = {f"PPV {GROUPS[g]}": ppv[:, g] for g in range(3)}
    for g, h in PAIRS:
        out[f"PPVD {PAIR_NAME[(g, h)]}"] = ppv[:, g] - ppv[:, h]
    out["max PPV gap"] = np.max(np.abs(np.stack([ppv[:, g] - ppv[:, h] for g, h in PAIRS])), 0)
    return out


def metrics_C(J):
    th = J.sum(1)                                                     # [N, (00,01,10,11), a]
    p = th.reshape(len(th), 2, 2, 3)                                  # [N, d, y, a]
    tp, fp, fn, tn = p[:, 1, 1], p[:, 1, 0], p[:, 0, 1], p[:, 0, 0]   # each [N, a]
    with np.errstate(divide="ignore", invalid="ignore"):
        tpr, fpr = tp / (tp + fn), fp / (fp + tn)
        acc = (tp + tn) / (tp + fp + fn + tn)
        ba = (tpr + 1 - fpr) / 2
        f1 = 2 * tp / (2 * tp + fp + fn)
    out = {}
    for g, h in PAIRS:
        out[f"TPRD {PAIR_NAME[(g, h)]}"] = tpr[:, g] - tpr[:, h]
        out[f"FPRD {PAIR_NAME[(g, h)]}"] = fpr[:, g] - fpr[:, h]
        out[f"ACCD {PAIR_NAME[(g, h)]}"] = acc[:, g] - acc[:, h]
    out["equalized-odds gap"] = np.max(np.abs(np.stack(
        [tpr[:, g] - tpr[:, h] for g, h in PAIRS] + [fpr[:, g] - fpr[:, h] for g, h in PAIRS])), 0)
    for g in (1, 2):
        out[f"BA diff {GROUPS[g]}-White"] = ba[:, g] - ba[:, 0]
        out[f"F1 diff {GROUPS[g]}-White"] = f1[:, g] - f1[:, 0]
    out["I(D;A|Y)"] = mi(p, 1, 3, cond_axes=(2,))
    out["I(Y;A|D)"] = mi(p, 2, 3, cond_axes=(1,))
    return out


METRICS = {"A": metrics_A, "B": metrics_B, "C": metrics_C}


def truth_metrics(run, truth):
    """truth p[z, a, i] -> metrics, using the same functions on a single 'joint'."""
    J = np.transpose(truth, (0, 2, 1))[None]                          # [1, z, i, a]
    return {k: float(v[0]) for k, v in METRICS[run](J).items()}


# ------------------------------------------------------------------------- methods
def grid(r, m, run, num, method, max_joints):
    K, I = r.shape
    p_primary = {(i, z): r[z, i] for z in range(K) for i in range(I)}
    p_aux = {(z, a): m[z, a] for z in range(K) for a in range(3)}
    t0 = time.perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            J, (Ig, C, E) = solve_joint_distribution(p_primary, p_aux, num=num, method=method,
                                                     max_joints=max_joints, as_dicts=False)
    except SystemExit:
        return None, time.perf_counter() - t0
    assert list(Ig) == list(range(I)) and list(E) == [0, 1, 2]
    return METRICS[run](J), time.perf_counter() - t0, len(J)


def kallus(r, m, run, truth, n_grid):
    """Bounds from kallus_bounds where they exist: {metric: (lower, upper, note)}."""
    out = {}
    if run == "C":
        # internal order 0:(d0,y0) 1:(d0,y1) 2:(d1,y0) 3:(d1,y1)
        specs = {"TPRD": ([3], [1, 3]), "FPRD": ([2], [0, 2]), "ACCD": ([0, 3], [0, 1, 2, 3])}
        eo = []
        for name, (t_idx, c_idx) in specs.items():
            target = np.isin(np.arange(4), t_idx); cond = np.isin(np.arange(4), c_idx)
            for ref in (0, 1):
                df = lp_bounds(r, m, target, cond, ref=ref, n_grid=n_grid)
                for _, row in df.iterrows():
                    pair = (int(row.group), ref)
                    if pair in PAIRS:
                        out[f"{name} {PAIR_NAME[pair]}"] = (row.lower, row.upper, "LP, sharp")
                        if name in ("TPRD", "FPRD"):
                            eo.append((row.lower, row.upper))
        out["equalized-odds gap"] = (
            max(0.0 if lo <= 0 <= hi else min(abs(lo), abs(hi)) for lo, hi in eo),
            max(max(abs(lo), abs(hi)) for lo, hi in eo),
            "upper sharp, lower outer (separate LPs)")
        return out
    if run == "A":
        target, cond = np.array([False, True]), np.array([True, True])
    else:
        target, cond = np.array([True, False, False]), np.array([True, True, False])
    name = "DD" if run == "A" else "PPVD"
    iv = {}
    for ref in (0, 1):
        df = lp_bounds(r, m, target, cond, ref=ref, n_grid=n_grid)
        for _, row in df.iterrows():
            pair = (int(row.group), ref)
            if pair in PAIRS:
                iv[pair] = (row.lower, row.upper)
    for pair, (lo, hi) in iv.items():
        out[f"{name} {PAIR_NAME[pair]}"] = (lo, hi, "LP, sharp")
    # outer bound on the maximum gap from the separate pair intervals
    gap_lo = [0.0 if lo <= 0 <= hi else min(abs(lo), abs(hi)) for lo, hi in iv.values()]
    gap_hi = [max(abs(lo), abs(hi)) for lo, hi in iv.values()]
    out[f"max {'DD' if run == 'A' else 'PPV'} gap"] = (max(gap_lo), max(gap_hi),
                                                        "upper sharp, lower outer (separate LPs)")
    if run == "A":                                     # one-vs-rest DI only (other estimand)
        for g in (1, 2):
            lo, hi = closed_form_bounds(r, one_vs_rest(m, g), target, cond, ratio=True)
            p1 = truth[:, g, :]; p_rest = truth.sum(1) - truth[:, g, :]
            t = (p1[:, 1].sum() / p1.sum()) / (p_rest[:, 1].sum() / p_rest.sum())
            out[f"DI {GROUPS[g]}/rest (one-vs-rest)"] = (lo, hi, f"closed form; truth {t:.3f}")
    return out


# ------------------------------------------------------------------------- experiment
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="~/kallus")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fracs", type=float, nargs=3, default=[0.2, 0.4, 0.4],
                    metavar=("TRAIN", "PRIMARY", "AUX"))
    ap.add_argument("--county-cuts", type=float, nargs=2, default=[0.30, 0.15],
                    metavar=("BLACK", "API"))
    ap.add_argument("--income-bins", type=int, default=2)
    ap.add_argument("--proxies", nargs="+", default=["none", "income", "county", "both"])
    ap.add_argument("--nums-a", type=int, nargs="+", default=[2, 3, 5, 9])
    ap.add_argument("--nums-b", type=int, nargs="+", default=[2, 3, 5])
    ap.add_argument("--nums-c", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--methods", nargs="+", default=["sequential", "sequential-all"])
    ap.add_argument("--max-joints", type=int, default=5_000_000)
    ap.add_argument("--lp-grid", type=int, default=41, help="Kallus LP grid for 3 groups")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent
                                         / "experiment_results" / "exp1_hmda"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    d = prepare(a.repo, a.seed, a.fracs, a.county_cuts, a.income_bins)
    n = d.split.value_counts().to_dict()
    print(f"splits: {n}; classifier approves {d.D[d.split == 'primary'].mean():.3f} "
          f"(label {d.Y[d.split == 'primary'].mean():.3f}), "
          f"accuracy {(d.D == d.Y)[d.split == 'primary'].mean():.3f}")

    nums = {"A": a.nums_a, "B": a.nums_b, "C": a.nums_c}
    rows = []
    for proxy in a.proxies:
        for run in ("A", "B", "C"):
            r, m, truth = tables(d, proxy, run)
            tm = truth_metrics(run, truth)
            base = dict(proxy=proxy, run=run, K=r.shape[0])
            t0 = time.perf_counter()
            for metric, (lo, hi, note) in kallus(r, m, run, truth, a.lp_grid).items():
                rows.append(dict(base, metric=metric, method="kallus", num=np.nan,
                                 lower=lo, upper=hi, truth=tm.get(metric, np.nan), note=note))
            tk = time.perf_counter() - t0
            for method in a.methods:
                for num in nums[run]:
                    res = grid(r, m, run, num, method, a.max_joints)
                    if res[0] is None:
                        print(f"  {proxy:6s} run {run} {method} num={num}: skipped (> max_joints)")
                        continue
                    out, sec, nj = res
                    for metric, v in out.items():
                        v = v[~np.isnan(v)]
                        if v.size == 0:
                            print(f"    note: {metric} undefined on every joint ({proxy}, run {run})")
                            continue
                        rows.append(dict(base, metric=metric, method=f"grid_{method}", num=num,
                                         lower=v.min(), upper=v.max(), truth=tm[metric],
                                         n_joints=nj, seconds=sec, note=""))
                    print(f"  {proxy:6s} run {run} {method} num={num}: {nj:,} joints, {sec:.2f}s"
                          + (f" (Kallus {tk:.1f}s)" if num == nums[run][0] and tk > 0.05 else ""))
    res = pd.DataFrame(rows)
    res["covers_truth"] = (res.lower <= res.truth + 1e-9) & (res.truth <= res.upper + 1e-9)
    res.to_csv(os.path.join(a.out, "results.csv"), index=False)

    # summary: grid at its largest feasible num per (proxy, metric, method) next to Kallus
    g = (res[res.method.str.startswith("grid")].sort_values("num")
         .groupby(["proxy", "metric", "method"], sort=False).tail(1))
    k = res[res.method == "kallus"].set_index(["proxy", "metric"])
    order = {m: i for i, m in enumerate(dict.fromkeys(res.metric))}
    summ = []
    for _, row in g.iterrows():
        kk = k.loc[(row.proxy, row.metric)] if (row.proxy, row.metric) in k.index else None
        verdict = ""
        if row.metric.startswith("DI ") or row.metric.startswith("worst-case DI"):
            verdict = ("violation certified" if row.upper < 0.8 else
                       "compliance certified" if row.lower >= 0.8 else "undetermined")
        summ.append(dict(proxy=row.proxy, metric=row.metric, truth=row.truth,
                         grid_lower=row.lower, grid_upper=row.upper, num=int(row.num),
                         method=row.method,
                         kallus_lower=kk.lower if kk is not None else np.nan,
                         kallus_upper=kk.upper if kk is not None else np.nan,
                         kallus_note=kk.note if kk is not None else "not available",
                         four_fifths=verdict))
    for (proxy, metric), kk in k.iterrows():           # Kallus-only rows (one-vs-rest DI)
        if not ((g.proxy == proxy) & (g.metric == metric)).any():
            summ.append(dict(proxy=proxy, metric=metric, truth=np.nan, grid_lower=np.nan,
                             grid_upper=np.nan, num=np.nan, method="",
                             kallus_lower=kk.lower, kallus_upper=kk.upper,
                             kallus_note=kk.note, four_fifths=""))
    summ = pd.DataFrame(summ)
    summ["o"] = summ.metric.map(order)
    summ = summ.sort_values(["proxy", "o"], key=lambda s: s.map(
        {p: i for i, p in enumerate(a.proxies)}) if s.name == "proxy" else s).drop(columns="o")
    summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 500,
                           "display.max_colwidth", 40):
        for proxy, t in summ.groupby("proxy", sort=False):
            print(f"\n=== proxy: {proxy} ===")
            print(t.drop(columns="proxy").round(3).to_string(index=False))
    miss = res[res.method.str.startswith("grid") & ~res.covers_truth]
    print(f"\ngrid intervals missing the truth: {len(miss)} of "
          f"{res.method.str.startswith('grid').sum()} (sampling noise between splits, or a "
          "coarse grid)")
    print(f"saved to {a.out}/")


if __name__ == "__main__":
    main()