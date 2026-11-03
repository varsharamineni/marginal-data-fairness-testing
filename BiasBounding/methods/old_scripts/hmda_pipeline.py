"""
End-to-end partial-identification pipeline on HMDA (the Kallus et al. 1% sample, or any CSV with
the same columns).

  1. truth        load the data; true fairness metrics use the real race labels
  2. split        disjoint INTERNAL data (features, label, proxy; race dropped) and EXTERNAL data
                  (proxy, race only), e.g. a lender's records and a census-like survey
  3. classifiers  trained on part of the internal data; predictions on the rest (evaluation set)
  4. marginals    primary p(proxy, label, prediction) from the internal evaluation set,
                  auxiliary P(race | proxy) from the external data
  5. sets         the feasible joints, three ways:
                    grid       grid_joint 'sequential-all' (full joints; small proxies only)
                    vertices   per-proxy-level vertices, all combinations (small proxies only)
                    aggregate  vertices of the set of feasible AGGREGATED tables
                               P(internal, group) = sum over proxy levels; every metric depends on
                               the joint only through this table, which lives in |internal|-1
                               dimensions, so its vertices are built level by level with convex-hull
                               pruning. Exact and feasible for real proxies.
                  validated against each other and the Kallus closed forms on a coarse proxy
  6. metrics      LP-type metrics (rate disparities, ratios): exact at the vertices
                  hard metrics (equalized odds, sufficiency, conditional MI, treatment equality,
                  lower bounds of gaps): vertices give the upper bound, an interior search
                  (convex combinations + constrained refinement) the lower bound
                  joint regions of metric pairs
                  classifier comparison: one joint over both classifiers' predictions; is one
                  classifier fairer under EVERY feasible joint?
                  informativeness: proxy choice and resolution, external sample size, known group
                  base rates, and which proxy levels drive the width

Usage (repo root):
  python methods/hmda_pipeline.py --data ~/kallus/HMDA/small_proxy_county_income.csv
  python methods/hmda_pipeline.py --data ... --groups Black --stages truth sets metrics
Outputs go to --out (CSV and PNG).

Notes. The label is the lender's historical approval (HMDA has no repayment outcome), so a
classifier's "TPR" is its agreement with past approvals. Disparities are rate(group) - rate(rest).
Predictions use a threshold that matches the training approval rate (a 0.5 threshold predicts
almost everyone approved).
"""
import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.spatial import QhullError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kallus_bounds import closed_form_bounds                         # noqa: E402
from BiasBounding.methods.old_scripts.experiments import run_grid, metric_values, plt                  # noqa: E402
from BiasBounding.methods.old_scripts.grid_limits import cmi_values, _lp_rate_range                    # noqa: E402

EDGES = [0, 25, 50, 75, 100, 150, 250, np.inf]                        # income_bin as in hmda_build.py
LP_METRICS = ["DP_diff", "TPR_diff", "FPR_diff", "PPV_diff", "NPV_diff", "ACC_diff",
              "DI_ratio", "F1_diff", "FDR_diff"]
HARD_METRICS = ["EOdds", "Sufficiency", "CMI", "TE_diff", "DP_gap", "TPR_gap"]


# =================================================================================== 1-3 data
def load(path):
    df = pd.read_csv(path)
    need = {"outcome", "race", "applicant_income_000s", "county_id"}
    if need - set(df.columns):
        raise SystemExit(f"{path} needs columns {sorted(need)}")
    return df.dropna(subset=list(need)).reset_index(drop=True)


def make_proxy(df, kind, bins=None, ref=None):
    """Proxy (common variable) as strings. kind: income_bin | income_q<k> | income | county |
    county_income (county x income_bin)."""
    inc = df.applicant_income_000s
    ib = pd.cut(inc, EDGES, right=False, labels=False).astype(int)
    if kind == "income_bin":
        return ib.map(lambda v: f"i{v:02d}")
    if kind.startswith("income_q"):
        k = int(kind[8:])
        #e = np.unique(np.quantile(inc, np.linspace(0, 1, k + 1)))
        src = (ref if ref is not None else df).applicant_income_000s
        e = np.unique(np.quantile(src, np.linspace(0, 1, k + 1)))
        return pd.cut(inc, e, include_lowest=True, labels=False).map(lambda v: f"q{int(v):03d}")
    if kind == "income":
        return inc.map(lambda v: f"{int(v):05d}")
    if kind == "county":
        return df.county_id.astype(int).map(lambda v: f"c{v:05d}")
    if kind == "county_income":
        return df.county_id.astype(int).map(lambda v: f"c{v:05d}") + "|" + ib.map(str)
    raise ValueError(kind)


def split(df, seed, internal_frac, train_frac, external_frac):
    """Disjoint internal / external rows; internal split into train / evaluation."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_int = int(internal_frac * len(df))
    internal, rest = idx[:n_int], idx[n_int:]
    external = rest[: int(external_frac * len(df))] if external_frac else rest
    n_tr = int(train_frac * len(internal))
    return dict(train=internal[:n_tr], eval=internal[n_tr:], external=external)


def fit_classifiers(df, tr, ev, names, seed):
    """Classifiers on (log income, county target encoding); predictions on the evaluation rows
    with a threshold matching the training approval rate."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    y = df.outcome.to_numpy().astype(int)
    li = np.log1p(df.applicant_income_000s.to_numpy(float))
    cty = df.county_id.to_numpy()
    s = pd.Series(y[tr]).groupby(cty[tr]).agg(["sum", "count"])
    prior, k = y[tr].mean(), 20.0                                      # smoothed target encoding
    te_map = (s["sum"] + k * prior) / (s["count"] + k)
    te = pd.Series(cty).map(te_map).fillna(prior).to_numpy()
    feats = {"income": li[:, None], "income+county": np.c_[li, te]}
    models = {
        "logit_income": (LogisticRegression(max_iter=1000), "income"),
        "logit_income_county": (LogisticRegression(max_iter=1000), "income+county"),
        "gbm_income_county": (HistGradientBoostingClassifier(max_iter=200, random_state=seed),
                              "income+county"),
    }
    out = {}
    for name in names:
        model, f = models[name]
        model.fit(feats[f][tr], y[tr])
        p_tr, p_ev = model.predict_proba(feats[f][tr])[:, 1], model.predict_proba(feats[f][ev])[:, 1]
        thr = np.quantile(p_tr, 1 - y[tr].mean())
        out[name] = (p_ev >= thr).astype(int)
        acc = (out[name] == y[ev]).mean()
        print(f"  {name:<22} eval accuracy {acc:.3f}, approval rate {out[name].mean():.3f}")
    return out


# =================================================================================== 4 marginals
def internal_values(k):
    """Internal values (prediction_1..prediction_k, label) as tuples; label last."""
    return list(itertools.product((0, 1), repeat=k + 1))


def marginals(z_ev, internal_cols, z_ext, race_ext, race_ev, group):
    """
    R [Z, I]: p(proxy, internal) from the evaluation rows; M [Z, 2]: P(proxy) * P(group | proxy)
    with P(group | proxy) from the external rows (pooled rate if a level is missing there);
    T [Z, I, 2]: the true joint (real race of the evaluation rows), for truth and oracle runs.
    Columns of M and T: (rest, group).
    """
    I = internal_values(internal_cols.shape[1] - 1)
    key = {v: k for k, v in enumerate(I)}
    ic = np.array([key[tuple(r)] for r in internal_cols])
    Zs = sorted(set(z_ev))
    zi = {z: k for k, z in enumerate(Zs)}
    zc = np.array([zi[z] for z in z_ev])
    n = len(z_ev)
    R = np.zeros((len(Zs), len(I)))
    np.add.at(R, (zc, ic), 1.0 / n)
    g_ev = (race_ev == group).astype(int)
    T = np.zeros((len(Zs), len(I), 2))
    np.add.at(T, (zc, ic, g_ev), 1.0 / n)
    ext = pd.Series((race_ext == group).astype(float)).groupby(np.asarray(z_ext)).mean()
    pg = pd.Series(Zs).map(ext).fillna((race_ext == group).mean()).to_numpy()
    P = R.sum(1)
    M = np.stack([P * (1 - pg), P * pg], 1)
    return R, M, T, I, Zs


# =================================================================================== 5 sets
def cell_vertices(r, mz, tol=1e-13):
    """Vertices of {q : 0 <= q <= r, sum q = mz}: group-1 mass in each internal cell at one proxy
    level (binary group). Every vertex has all cells but at most one at a bound."""
    I = len(r)
    if mz <= tol:
        return np.zeros((1, I))
    if mz >= r.sum() - tol:
        return r[None].copy()
    masks = np.array(list(itertools.product((0.0, 1.0), repeat=I - 1)))
    out = []
    for k in range(I):
        oth = [i for i in range(I) if i != k]
        q = np.zeros((len(masks), I))
        q[:, oth] = masks * r[oth]
        q[:, k] = mz - q[:, oth].sum(1)
        ok = (q[:, k] >= -tol) & (q[:, k] <= r[k] + tol)
        out.append(q[ok])
    return np.unique(np.round(np.clip(np.vstack(out), 0, None), 15), axis=0)


def _hull(P):
    """Vertices of conv(P) (handles lower-dimensional sets). Returns (vertices, hull info)."""
    P = np.unique(np.round(P, 14), axis=0)
    if len(P) <= 1:
        return P, None
    c = P.mean(0)
    _, s, Vt = np.linalg.svd(P - c, full_matrices=False)
    k = int((s > max(s[0], 1e-300) * 1e-9).sum())
    if k == 0:
        return P[:1], None
    X = (P - c) @ Vt[:k].T
    if k == 1:
        return P[[X[:, 0].argmin(), X[:, 0].argmax()]], dict(c=c, B=Vt[:1], k=1,
                                                            lo=X[:, 0].min(), hi=X[:, 0].max())
    try:
        h = ConvexHull(X)
    except QhullError:
        h = ConvexHull(X, qhull_options="QJ")
    return P[h.vertices], dict(c=c, B=Vt[:k], k=k, eq=h.equations)


def aggregate_vertices(R, M, verbose=False, max_vertices=None):
    """
    Vertices of the set of feasible aggregated group-1 tables  theta_1 = sum_z q_z  (a Minkowski
    sum of the per-level polytopes), built level by level with convex-hull pruning.
    Returns (vertices [V, I], hull info for interior search).
    """
    S, info = cell_vertices(R[0], M[0, 1]), None
    for z in range(1, len(R)):
        V = cell_vertices(R[z], M[z, 1])
        if max_vertices and len(S) * len(V) > 2 * max_vertices:   # hull cost explodes in high
            return None, None                                         # dimension with input size
        S, info = _hull((S[:, None, :] + V[None, :, :]).reshape(-1, R.shape[1]))
        if max_vertices and len(S) > max_vertices:
            return None, None
        if verbose and z % 200 == 0:
            print(f"    level {z}/{len(R)}: {len(S)} vertices", flush=True)
    if info is None:
        S, info = _hull(S)
    return S, info


def product_vertices(R, M, max_joints=5e6):
    """All combinations of per-level vertices: full joints [N, Z, I, 2] (small proxies only)."""
    Vs = [cell_vertices(R[z], M[z, 1]) for z in range(len(R))]
    if np.prod([float(len(v)) for v in Vs]) > max_joints:
        return None
    q = np.zeros((1, 0, R.shape[1]))
    for V in Vs:
        q = np.concatenate([np.repeat(q, len(V), 0), np.tile(V, (len(q), 1))[:, None]], 1)
    return np.stack([R[None] - q, q], -1)


def to_theta(theta1, Ragg):
    """Aggregated group-1 tables [N, I] -> full tables [N, I, 2] (rest, group)."""
    return np.stack([Ragg[None] - theta1, theta1], -1)


# =================================================================================== 6 metrics
def masks(I, k=0):
    """decision / label masks over internal values for classifier k (label is last)."""
    return (np.array([bool(v[k]) for v in I]), np.array([bool(v[-1]) for v in I]))


def all_metrics(theta, I, k=0):
    """Every metric for classifier k from aggregated tables theta [N, I, 2]. Cells below 1e-14
    (floating-point noise from subtracting tables) are set to 0, so a rate whose condition has
    no mass is undefined (NaN) instead of a huge number."""
    theta = np.where(theta < 1e-14, 0.0, theta)
    d, y = masks(I, k)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = metric_values(theta, d, y)
        acc = theta[:, d == y, :].sum(1) / theta.sum(1)
        v["ACC_diff"] = acc[:, 1] - acc[:, 0]
        fn, fp = theta[:, ~d & y, :].sum(1), theta[:, d & ~y, :].sum(1)
        te = fn / fp
        v["TE_diff"] = te[:, 1] - te[:, 0]
        v["CMI"] = cmi_values(theta, d, y)
        v["DP_gap"], v["TPR_gap"] = np.abs(v["DP_diff"]), np.abs(v["TPR_diff"])
    return v


def feasible_table(theta1, R, M, tol=1e-9):
    """Exact check (LP) that an aggregated group-1 table is attainable: some q[z, i] with
    0 <= q <= R, sum_i q[z, i] = M[z, 1] and sum_z q[z, i] = theta1[i]."""
    from scipy.optimize import linprog
    Z, nI = R.shape
    A = np.zeros((Z + nI, Z * nI))
    for z in range(Z):
        A[z, z * nI:(z + 1) * nI] = 1
    for i in range(nI):
        A[Z + i, i::nI] = 1
    b = np.r_[M[:, 1], theta1]
    res = linprog(np.zeros(Z * nI), A_eq=A, b_eq=b, bounds=[(0, R[z, i]) for z in range(Z) for i in range(nI)],
                  method="highs", options=dict(primal_feasibility_tolerance=tol))
    return res.status == 0


def interior_search(f, verts, info, Ragg, n_samples=50000, n_refine=6, seed=0, RM=None):
    """
    Min and max of f(theta [N, I, 2]) -> [N] over the polytope conv(verts): vertices, random
    convex combinations of 2-4 vertices, then SLSQP refinement inside the hull from the best
    points. Inner approximation: the minimum found is >= the true minimum.
    """
    rng = np.random.default_rng(seed)
    V = len(verts)
    pts = [verts]
    for kk in (2, 3, 4):
        if V < kk:
            continue
        idx = rng.integers(0, V, (n_samples // 3, kk))
        w = rng.dirichlet(np.ones(kk), n_samples // 3)
        pts.append((verts[idx] * w[:, :, None]).sum(1))
    P = np.vstack(pts)
    with np.errstate(invalid="ignore", divide="ignore"):
        vals = f(to_theta(P, Ragg))
    best = {}
    for sgn, name in ((1, "min"), (-1, "max")):
        sv = np.where(np.isfinite(vals), sgn * vals, np.inf)
        order = np.argsort(sv)[:n_refine]
        b = sv[order[0]] if len(order) else np.inf
        if info is not None and info.get("k", 0) >= 1 and np.isfinite(b):
            c, B = info["c"], info["B"]
            if info["k"] == 1:
                cons = [{"type": "ineq", "fun": lambda u: u[0] - info["lo"]},
                        {"type": "ineq", "fun": lambda u: info["hi"] - u[0]}]
            else:
                A, bb = info["eq"][:, :-1], info["eq"][:, -1]
                cons = [{"type": "ineq", "fun": lambda u: -(A @ u + bb)}]
            g = lambda u: (lambda x: sgn * x if np.isfinite(x) else 1e6)(
                float(f(to_theta((c + u @ B)[None], Ragg))[0]))
            for j in order:
                u0 = (P[j] - c) @ B.T
                with np.errstate(invalid="ignore", divide="ignore"):
                    res = minimize(g, u0, method="SLSQP", constraints=cons,
                                   options=dict(maxiter=200, ftol=1e-12))
                if not res.fun < b:
                    continue
                th1 = c + res.x @ B                    # verify exactly: the hull facets can be
                ok = RM is not None and feasible_table(th1, *RM)   # slightly off when degenerate
                if ok:
                    b = res.fun
        best[name] = sgn * b if np.isfinite(b) else np.nan
    return best["min"], best["max"]


# =================================================================================== stages
def stage_truth_and_bounds(ctx, a):
    """Bounds for every metric, classifier and group, from aggregated vertices (+ interior
    search for the hard metrics), with the true value and Kallus closed forms alongside."""
    rows, regions = [], {}
    for group in a.groups:
        for name in ctx["clf"]:
            ic = np.c_[ctx["pred"][name], ctx["y_ev"]]
            R, M, T, I, Zs = marginals(ctx["z_ev"], ic, ctx["z_ext"], ctx["race_ext"],
                                       ctx["race_ev"], group)
            Ragg = R.sum(0)
            t0 = time.perf_counter()
            verts, info = aggregate_vertices(R, M)
            dt = time.perf_counter() - t0
            th_v = to_theta(verts, Ragg)
            mv = all_metrics(th_v, I)
            truth = all_metrics(T.sum(0)[None], I)
            # oracle auxiliary data (true P(group | proxy) of the evaluation rows): truth must lie inside
            Mo = T.sum(1)
            vo, _ = aggregate_vertices(R, Mo)
            mo = all_metrics(to_theta(vo, Ragg), I)
            d, y = masks(I)
            cf_specs = {"DP_diff": (d, np.ones_like(d)), "TPR_diff": (d & y, y),
                        "FPR_diff": (d & ~y, ~y), "PPV_diff": (d & y, d), "NPV_diff": (~d & ~y, ~d)}
            for met in LP_METRICS + HARD_METRICS:
                lo, hi = np.nanmin(mv[met]), np.nanmax(mv[met])
                how = "vertices (exact)"
                if met in HARD_METRICS:
                    f = lambda th, met=met: all_metrics(th, I)[met]
                    slo, shi = interior_search(f, verts, info, Ragg, a.samples, seed=a.seed, RM=(R, M))
                    lo, hi = np.nanmin([lo, slo]), np.nanmax([hi, shi])
                    how = "vertices + interior search (lower: inner approx.)"
                cf = closed_form_bounds(R, M, *cf_specs[met], harmonize=False) if met in cf_specs else (np.nan, np.nan)
                tv = float(truth[met][0])
                rows.append(dict(group=group, classifier=name, metric=met, lower=lo, upper=hi,
                                 truth=tv, covered=bool(lo - 1e-9 <= tv <= hi + 1e-9),
                                 oracle_lower=np.nanmin(mo[met]), oracle_upper=np.nanmax(mo[met]),
                                 kallus_lower=cf[0], kallus_upper=cf[1], method=how,
                                 proxy_levels=len(Zs), vertices=len(verts), seconds=dt))
            regions[(group, name)] = (mv, truth, verts, info, Ragg, I)
            print(f"  {group:<6} {name:<22} {len(Zs)} proxy levels -> {len(verts)} vertices ({dt:.2f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "bounds.csv"), index=False)
    show = df[df.metric.isin(["DP_diff", "TPR_diff", "FPR_diff", "PPV_diff", "EOdds", "CMI"])]
    print(show[["group", "classifier", "metric", "lower", "upper", "truth", "covered",
                "kallus_lower", "kallus_upper"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    ctx["regions"] = regions
    if plt is not None:
        _plot_bounds(df, a)
        _plot_regions(regions, a)
    return df


def stage_validate(ctx, a):
    """On a coarse proxy: grid (sequential-all), per-level vertex combinations and aggregated
    vertices must give the same bounds for vertex-attained metrics, equal to Kallus."""
    rows = []
    z_c = make_proxy(ctx["df"].iloc[ctx["split"]["eval"]], a.validate_proxy).to_numpy()
    z_x = make_proxy(ctx["df"].iloc[ctx["split"]["external"]], a.validate_proxy).to_numpy()
    for group in a.groups:
        name = next(iter(ctx["clf"]))
        ic = np.c_[ctx["pred"][name], ctx["y_ev"]]
        R, M, T, I, Zs = marginals(z_c, ic, z_x, ctx["race_ext"], ctx["race_ev"], group)
        Ragg = R.sum(0)
        res = {}
        t0 = time.perf_counter(); verts, _ = aggregate_vertices(R, M)
        res["aggregate"] = (all_metrics(to_theta(verts, Ragg), I), time.perf_counter() - t0, len(verts))
        t0 = time.perf_counter(); J = product_vertices(R, M)
        if J is not None:
            res["vertex product"] = (all_metrics(J.sum(1), I), time.perf_counter() - t0, len(J))
        pp = {(I[i], Zs[z]): R[z, i] for z in range(len(Zs)) for i in range(len(I))}
        pa = {(Zs[z], g): M[z, k] for z in range(len(Zs)) for k, g in enumerate(["0_rest", "1_grp"])}
        t0 = time.perf_counter(); g = run_grid(pp, pa, 2, "sequential-all", 5e6)
        if g is not None:
            Jg, Ig, _, _ = g
            order = [Ig.index(v) for v in I]
            res["grid seq-all num=2"] = (all_metrics(Jg.sum(1)[:, order, :], I),
                                         time.perf_counter() - t0, len(Jg))
        for met in LP_METRICS + ["EOdds", "Sufficiency"]:
            for meth, (mv, sec, n) in res.items():
                rows.append(dict(group=group, proxy=a.validate_proxy, levels=len(Zs), metric=met,
                                 method=meth, n=n, seconds=sec, lower=np.nanmin(mv[met]),
                                 upper=np.nanmax(mv[met])))
    df = pd.DataFrame(rows)
    piv = df.groupby(["group", "metric"]).apply(
        lambda s: pd.Series(dict(max_spread_lower=s.lower.max() - s.lower.min(),
                                 max_spread_upper=s.upper.max() - s.upper.min())), include_groups=False)
    df.to_csv(os.path.join(a.out, "validation.csv"), index=False)
    print(f"\nVALIDATION ({a.validate_proxy}): largest difference between methods (upper bounds of "
          "max-type metrics and all LP-type bounds should agree; lower bounds of EOdds/Sufficiency "
          "may differ, they lie between vertices)")
    print(piv.to_string(float_format=lambda x: f"{x:.1e}"))
    print(df.groupby("method")[["n", "seconds"]].max().to_string())
    return df


def metric_terms(I, c, met):
    """Rate terms (target, cond) of a metric for classifier c (column c of the internal tuple)."""
    d, y = masks(I, c)
    ones = np.ones_like(d)
    return {"DP_diff": [(d, ones)], "TPR_diff": [(d & y, y)], "FPR_diff": [(d & ~y, ~y)],
            "PPV_diff": [(d & y, d)], "EOdds": [(d & y, y), (d & ~y, ~y)],
            "Sufficiency": [(d & y, d), (~d & ~y, ~d)]}[met]


def compare_lp(R, M, I, met, kind, n_grid=15, refine_rounds=30):
    """
    max over all couplings of  M_A - M_B (kind 'signed', single-rate metrics) or
    |M_A| - |M_B| (kind 'abs'), where |M_c| = max_k |rate disparity_k| for classifier c.
    For fixed group-1 masses of every condition the rate disparities are linear in the coupling
    q[z, i] = P(z, i, group); 'max of (s*L_A - max_j |L_Bj|)' is an LP with one epigraph variable.
    The condition masses are searched on a grid (refined by pattern search): exact for DP (no
    varying denominators), inner approximation otherwise (as in the paper's t-grid).
    Returns the maximum (call with A and B swapped and negate for the minimum).
    """
    from scipy.optimize import linprog
    Z, nI = R.shape
    P, m1 = R.sum(1), M[:, 1]
    tA, tB = metric_terms(I, 0, met), metric_terms(I, 1, met)
    conds = []
    for _, c in tA + tB:
        if not any(np.array_equal(c, cc) for cc in conds):
            conds.append(c)
    fixed = [bool(c.all()) for c in conds]
    K = [R[:, c].sum() for c in conds]
    rng_ = [(np.maximum(0, R[:, c].sum(1) + m1 - P).sum(), np.minimum(R[:, c].sum(1), m1).sum())
            for c in conds]
    nv = Z * nI
    A_eq = np.zeros((Z, nv + 1)); b_eq = m1.copy()
    for z in range(Z):
        A_eq[z, z * nI:(z + 1) * nI] = 1
    bounds = [(0, R[z, i]) for z in range(Z) for i in range(nI)] + [(None, None)]
    cidx = lambda c: next(k for k, cc in enumerate(conds) if np.array_equal(c, cc))

    def lin(term, x):                    # disparity = w @ q + const, for condition masses x
        tg, c = term
        k = cidx(c)
        x1 = m1.sum() if fixed[k] else x[k]
        x0 = (P.sum() - m1.sum()) if fixed[k] else K[k] - x[k]
        if x1 <= 1e-12 or x0 <= 1e-12:
            return None
        w = np.tile(tg.astype(float), Z) * (1 / x1 + 1 / x0)
        return w, -R[:, tg].sum() / x0

    def value(x):
        rows_eq = [A_eq]; beq = [b_eq]
        for k, c in enumerate(conds):    # fix the condition masses
            if not fixed[k]:
                r_ = np.zeros(nv + 1); r_[:nv] = np.tile(c.astype(float), Z)
                rows_eq.append(r_[None]); beq.append([x[k]])
        Aeq, beq = np.vstack(rows_eq), np.concatenate(beq)
        LB = [lin(t, x) for t in tB]
        LA = [lin(t, x) for t in tA]
        if any(v is None for v in LA + LB):
            return -np.inf
        best = -np.inf
        choices = ([(1, LA[0])] if kind == "signed" else [(s, L) for L in LA for s in (1, -1)])
        for s, (wa, ca) in choices:
            ub, bu = [], []
            if kind == "signed":
                wb, cb = LB[0]
                cons = [(-1, wb, cb)]            # w <= s*L_A - L_B
            else:
                cons = [(t, wb, cb) for wb, cb in LB for t in (1, -1)]   # w <= s*L_A - t*L_Bj
            for t, wb, cb in cons:
                r_ = np.zeros(nv + 1)
                r_[:nv] = -(s * wa) + t * wb if kind == "abs" else -(s * wa) + wb
                r_[nv] = 1
                ub.append(r_)
                bu.append((s * ca - t * cb) if kind == "abs" else (s * ca - cb))
            res = linprog(np.r_[np.zeros(nv), -1.0], A_ub=np.array(ub), b_ub=bu, A_eq=Aeq,
                          b_eq=beq, bounds=bounds, method="highs")
            if res.status == 0:
                best = max(best, -res.fun)
        return best

    free = [k for k in range(len(conds)) if not fixed[k]]
    if not free:
        return value(np.zeros(len(conds)))
    axes = [np.linspace(*rng_[k], n_grid) for k in free]
    best, arg = -np.inf, None
    for pt in itertools.product(*axes):
        x = np.zeros(len(conds)); x[free] = pt
        v = value(x)
        if v > best:
            best, arg = v, x
    if arg is None:
        return np.nan
    step = np.array([(rng_[k][1] - rng_[k][0]) / max(n_grid - 1, 1) for k in free])
    for _ in range(refine_rounds):                 # pattern search
        moved = False
        for j, k in enumerate(free):
            for sgn in (1, -1):
                x = arg.copy(); x[k] = np.clip(x[k] + sgn * step[j], *rng_[k])
                v = value(x)
                if v > best + 1e-13:
                    best, arg, moved = v, x, True
        if not moved:
            step /= 2
            if np.all(step < 1e-10):
                break
    return best


def stage_compare(ctx, a):
    """One joint over two classifiers' predictions. For each metric: the range of M_A - M_B and
    of |M_A| - |M_B| over ALL feasible joints (LP, any number of proxy levels). A is fairer for
    every feasible joint if max(|M_A| - |M_B|) < 0. Cross-checked by aggregate-vertex search on a
    coarse proxy, and contrasted with comparing the two separate intervals."""
    rows = []
    names = list(ctx["clf"])
    df, sp = ctx["df"], ctx["split"]
    for group in a.groups:
        for A, B in itertools.combinations(names, 2):
            ic = np.c_[ctx["pred"][A], ctx["pred"][B], ctx["y_ev"]]
            out = {}
            for proxy in (a.compare_proxy, a.compare_check_proxy):
                z_c = make_proxy(df.iloc[sp["eval"]], proxy).to_numpy()
                z_x = make_proxy(df.iloc[sp["external"]], proxy).to_numpy()
                out[proxy] = marginals(z_c, ic, z_x, ctx["race_ext"], ctx["race_ev"], group)
            R, M, T, I, Zs = out[a.compare_proxy]
            Rc, Mc, Tc, _, Zc = out[a.compare_check_proxy]
            t0 = time.perf_counter()
            vc, ic_info = aggregate_vertices(Rc, Mc, max_vertices=a.max_check_vertices)
            t_v = time.perf_counter() - t0
            for met in ["DP_diff", "TPR_diff", "FPR_diff", "PPV_diff", "EOdds"]:
                t0 = time.perf_counter()
                res = {}
                if met != "EOdds":
                    res["diff"] = (-_swap_lp(R, M, I, met, "signed"), compare_lp(R, M, I, met, "signed"))
                res["absdiff"] = (-_swap_lp(R, M, I, met, "abs"), compare_lp(R, M, I, met, "abs"))
                t_lp = time.perf_counter() - t0
                # separate intervals of |M_A| and |M_B| (what one would compare without a joint)
                sep = {}
                for c in (0, 1):
                    terms = metric_terms(I, c, met)
                    gaps = []
                    for tg, cd in terms:
                        lo, hi = closed_form_bounds(R, M, tg, cd, harmonize=False)
                        gaps.append((0.0 if lo <= 0 <= hi else min(abs(lo), abs(hi)), max(abs(lo), abs(hi))))
                    sep[c] = (max(g[0] for g in gaps), max(g[1] for g in gaps))
                # cross-check on the coarse proxy: aggregate vertices + interior search
                fA = lambda th: all_metrics(th, I, 0)[met]
                fB = lambda th: all_metrics(th, I, 1)[met]
                chk = (interior_search(lambda th: np.abs(fA(th)) - np.abs(fB(th)), vc, ic_info,
                                       Rc.sum(0), a.samples, seed=a.seed, RM=(Rc, Mc))
                       if vc is not None else (np.nan, np.nan))
                chk_lp = (-_swap_lp(Rc, Mc, I, met, "abs"), compare_lp(Rc, Mc, I, met, "abs"))
                alo, ahi = res["absdiff"]
                tt = T.sum(0)[None]
                true_abs = float((np.abs(fA(tt)) - np.abs(fB(tt)))[0])
                verdict = ("A fairer for every joint" if ahi < 0 else
                           "B fairer for every joint" if alo > 0 else "undetermined")
                sep_verdict = ("A fairer" if sep[0][1] < sep[1][0] else
                               "B fairer" if sep[1][1] < sep[0][0] else "undetermined")
                rows.append(dict(group=group, A=A, B=B, metric=met, proxy_levels=len(Zs),
                                 diff_lower=res.get("diff", (np.nan, np.nan))[0],
                                 diff_upper=res.get("diff", (np.nan, np.nan))[1],
                                 absdiff_lower=alo, absdiff_upper=ahi, true_absdiff=true_abs,
                                 covered=bool(alo - 1e-9 <= true_abs <= ahi + 1e-9),
                                 verdict_joint=verdict, verdict_separate_intervals=sep_verdict,
                                 sepA=f"[{sep[0][0]:.3f}, {sep[0][1]:.3f}]",
                                 sepB=f"[{sep[1][0]:.3f}, {sep[1][1]:.3f}]",
                                 lp_seconds=t_lp,
                                 check_proxy_levels=len(Zc),
                                 check_vertices=len(vc) if vc is not None else "too many",
                                 check_vertex_search=f"[{chk[0]:.4f}, {chk[1]:.4f}]",
                                 check_lp=f"[{chk_lp[0]:.4f}, {chk_lp[1]:.4f}]"))
            print(f"  {group:<6} {A} vs {B} done "
                  f"({len(vc) if vc is not None else 'skipped:'} check vertices, {t_v:.1f}s)", flush=True)
    df_ = pd.DataFrame(rows)
    df_.to_csv(os.path.join(a.out, "compare.csv"), index=False)
    print("\nCLASSIFIER COMPARISON over all feasible joints (LP). absdiff = |M_A| - |M_B|; "
          "< 0 everywhere means A is fairer under every joint consistent with the data")
    print(df_[["group", "A", "B", "metric", "absdiff_lower", "absdiff_upper", "true_absdiff",
               "verdict_joint", "verdict_separate_intervals"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\ncross-check on the coarse proxy (LP vs aggregate-vertex search):")
    print(df_[["group", "A", "B", "metric", "check_proxy_levels", "check_lp", "check_vertex_search"]].to_string(index=False))
    return df_


def _swap_lp(R, M, I, met, kind):
    """compare_lp with the two classifiers swapped (gives -min of M_A - M_B or |M_A| - |M_B|)."""
    J = [(v[1], v[0]) + tuple(v[2:]) for v in I]
    order = [J.index(v) for v in I]                  # reorder cells so classifier B comes first
    return compare_lp(R[:, order], M, I, met, kind)


def stage_inform(ctx, a):
    """What data are informative: proxy choice and resolution, external sample size, known group
    base rates, and which proxy levels drive the width (closed forms / LP: full scale)."""
    df, sp = ctx["df"], ctx["split"]
    name = a.inform_classifier if a.inform_classifier in ctx["pred"] else list(ctx["clf"])[-1]
    print(f"  classifier: {name} (a classifier whose inputs are the proxy itself makes DP "
          "point-identified at fine proxy resolutions)")
    ic = np.c_[ctx["pred"][name], ctx["y_ev"]]
    rows = []
    for group in a.groups:
        for proxy in a.inform_proxies:
            z_ev = make_proxy(df.iloc[sp["eval"]], proxy).to_numpy()
            z_ext = make_proxy(df.iloc[sp["external"]], proxy).to_numpy()
            R, M, T, I, Zs = marginals(z_ev, ic, z_ext, ctx["race_ext"], ctx["race_ev"], group)
            d, y = masks(I)
            for met, spec in {"DP": (d, np.ones_like(d)), "TPR": (d & y, y), "FPR": (d & ~y, ~y),
                              "PPV": (d & y, d)}.items():
                for aux, MM in (("estimated", M), ("oracle", T.sum(1))):
                    lo, hi = closed_form_bounds(R, MM, *spec, harmonize=False)
                    rows.append(dict(study="proxy", group=group, setting=proxy, levels=len(Zs),
                                     metric=met, aux=aux, lower=lo, upper=hi, width=hi - lo))
        # external sample size (income_bin proxy)
        z_ev = make_proxy(df.iloc[sp["eval"]], a.proxy).to_numpy()
        for frac in a.ext_fracs:
            n_ext = max(int(frac * len(df)), 50)
            sub = np.random.default_rng(a.seed).choice(sp["external"], min(n_ext, len(sp["external"])), replace=False)
            R, M, T, I, Zs = marginals(z_ev, ic, make_proxy(df.iloc[sub], a.proxy).to_numpy(),
                                       df.race.to_numpy()[sub], ctx["race_ev"], group)
            d, y = masks(I)
            lo, hi = closed_form_bounds(R, M, d & y, y, harmonize=False)
            olo, ohi = closed_form_bounds(R, T.sum(1), d & y, y, harmonize=False)
            rows.append(dict(study="external size", group=group, setting=f"{len(sub)} external rows",
                             classifier=name,
                             levels=len(Zs), metric="TPR", aux="estimated", lower=lo, upper=hi,
                             width=hi - lo, endpoint_error=max(abs(lo - olo), abs(hi - ohi))))
        # known group base rates P(label = 1 | group) to within eps (LP)
        R, M, T, I, Zs = marginals(z_ev, ic, ctx["z_ext"], ctx["race_ext"], ctx["race_ev"], group)
        d, y = masks(I)
        base = T.sum(0)[y].sum(0) / T.sum(0).sum(0)
        yw = lambda g_: (lambda lp: lp.masks(y, np.ones_like(y))[0][g_])
        for eps in a.base_eps:
            cons = None if not np.isfinite(eps) else [
                (yw(g_), (base[g_] - eps) * M[:, g_].sum(), (base[g_] + eps) * M[:, g_].sum()) for g_ in (0, 1)]
            lo, hi = _lp_rate_range(R, M, (d & y, y), cons)
            rows.append(dict(study="known base rates", group=group, setting=f"+-{eps}", levels=len(Zs),
                             metric="TPR", aux="estimated", lower=lo, upper=hi, width=hi - lo))
        # width decomposition: which proxy levels drive the DP width (county proxy, DP is additive)
        z_ev = make_proxy(df.iloc[sp["eval"]], "county").to_numpy()
        R, M, T, I, Zs = marginals(z_ev, ic, make_proxy(df.iloc[sp["external"]], "county").to_numpy(),
                                   ctx["race_ext"], ctx["race_ev"], group)
        h, P, mg = R[:, d].sum(1), R.sum(1), M[:, 1]
        k = 1 / mg.sum() + 1 / (1 - mg.sum())
        wz = (np.minimum(h, mg) - np.maximum(0, h + mg - P)) * k
        o = np.argsort(-wz)
        top = max(len(Zs) // 100, 1)
        mix = np.minimum(mg / P, 1 - mg / P)
        rows.append(dict(study="width decomposition", group=group, setting=f"top 1% of counties ({top})",
                         levels=len(Zs), metric="DP", aux="estimated",
                         share_of_width=wz[o[:top]].sum() / wz.sum(),
                         share_of_applicants=P[o[:top]].sum(),
                         corr_width_mixing=np.corrcoef(wz / P, mix)[0, 1]))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(a.out, "informativeness.csv"), index=False)
    print("\nINFORMATIVENESS")
    for st, s in out.groupby("study", sort=False):
        cols = [c for c in s.columns if s[c].notna().any() and c != "study"]
        print(f"-- {st}")
        print(s[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if plt is not None:
        s = out[(out.study == "proxy") & (out.aux == "estimated")]
        fig, ax = plt.subplots(figsize=(7, 4))
        for (g, met), ss in s.groupby(["group", "metric"]):
            ss = ss.sort_values("levels")
            ax.plot(ss.levels, ss.width, "o-", label=f"{g}: {met}")
        ax.set_xscale("log"); ax.set_xlabel("proxy levels"); ax.set_ylabel("width of identified interval")
        ax.legend(fontsize=7, ncol=2); fig.tight_layout()
        fig.savefig(os.path.join(a.out, "informativeness_proxy.png"), dpi=150)
    return out


# =================================================================================== plots
def _plot_bounds(df, a):
    mets = ["DP_diff", "TPR_diff", "FPR_diff", "PPV_diff", "NPV_diff", "EOdds"]
    for group, g in df.groupby("group"):
        g = g[g.metric.isin(mets)]
        fig, ax = plt.subplots(figsize=(8, 0.5 + 0.35 * len(g)))
        for k, r in enumerate(g.itertuples()):
            yy = len(g) - 1 - k
            ax.plot([r.lower, r.upper], [yy, yy], lw=4, color="#2a78d6", solid_capstyle="round",
                    label="identified set" if k == 0 else None)
            ax.plot([r.oracle_lower, r.oracle_upper], [yy - 0.25] * 2, lw=1.5, color="#eb6834",
                    label="with oracle P(race | proxy)" if k == 0 else None)
            ax.plot(r.truth, yy, "|", ms=14, mew=2, color="k", label="true value" if k == 0 else None)
        ax.set_yticks(range(len(g)), [f"{r.classifier}: {r.metric}" for r in g.itertuples()][::-1], fontsize=7)
        ax.axvline(0, color="0.8", ls="--", lw=1)
        ax.set_title(f"{group} vs rest, proxy {a.proxy}", loc="left")
        ax.legend(fontsize=7, loc="lower right"); fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"bounds_{group}.png"), dpi=150)


def _plot_regions(regions, a):
    pairs = [("TPR_diff", "FPR_diff"), ("DP_diff", "TPR_diff"), ("PPV_diff", "TPR_diff")]
    for (group, name), (mv, truth, verts, info, Ragg, I) in regions.items():
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(verts), (20000, 3))
        w = rng.dirichlet(np.ones(3), 20000)
        inner = all_metrics(to_theta((verts[idx] * w[:, :, None]).sum(1), Ragg), I)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, (xk, yk) in zip(axes, pairs):
            ax.scatter(inner[xk], inner[yk], s=1, c="#a9c7ef", label="feasible tables")
            ax.scatter(mv[xk], mv[yk], s=6, c="#2a78d6", label="vertices")
            ax.plot(truth[xk], truth[yk], "k*", ms=10, label="truth")
            ax.add_patch(plt.Rectangle((np.nanmin(mv[xk]), np.nanmin(mv[yk])),
                                       np.nanmax(mv[xk]) - np.nanmin(mv[xk]),
                                       np.nanmax(mv[yk]) - np.nanmin(mv[yk]),
                                       fill=False, ec="#eb6834", lw=1.2, label="separate intervals"))
            ax.set_xlabel(xk); ax.set_ylabel(yk)
        axes[0].legend(fontsize=7)
        fig.suptitle(f"{group} vs rest, {name}: joint identified regions")
        fig.tight_layout(); fig.savefig(os.path.join(a.out, f"regions_{group}_{name}.png"), dpi=130)
        plt.close(fig)


# =================================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.expanduser("~/kallus/HMDA/small_proxy_county_income.csv"))
    ap.add_argument("--out", default="results/hmda_pipeline")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--groups", nargs="+", default=["White", "Black", "API"])
    ap.add_argument("--classifiers", nargs="+",
                    default=["logit_income", "logit_income_county", "gbm_income_county"])
    ap.add_argument("--proxy", default="income_bin", help="proxy for the main bounds")
    ap.add_argument("--validate-proxy", default="income_q3")
    ap.add_argument("--compare-proxy", default="income_bin")
    ap.add_argument("--compare-check-proxy", default="income_q5",
                    help="coarse proxy for the vertex cross-check of the comparison LP")
    ap.add_argument("--max-check-vertices", type=int, default=2000)
    ap.add_argument("--inform-proxies", nargs="+",
                    default=["income_q2", "income_q4", "income_bin", "income_q16", "income_q64",
                             "income", "county", "county_income"])
    ap.add_argument("--inform-classifier", default="gbm_income_county")
    ap.add_argument("--internal-frac", type=float, default=0.5)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--external-frac", type=float, default=None,
                    help="fraction of all rows in the external data (default: all non-internal rows)")
    ap.add_argument("--ext-fracs", nargs="+", type=float, default=[0.001, 0.005, 0.02, 0.1, 0.5])
    ap.add_argument("--base-eps", nargs="+", type=float, default=[np.inf, 0.05, 0.01, 0.001])
    ap.add_argument("--samples", type=int, default=30000, help="interior search points")
    ap.add_argument("--stages", nargs="+", default=["bounds", "validate", "compare", "inform"],
                    choices=["bounds", "validate", "compare", "inform"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    print("1. data"); df = load(a.data)
    print(f"  {len(df):,} rows; race: {df.race.value_counts().to_dict()}")
    sp = split(df, a.seed, a.internal_frac, a.train_frac, a.external_frac)
    print(f"2. split: train {len(sp['train']):,} | evaluation {len(sp['eval']):,} | external {len(sp['external']):,}")
    print("3. classifiers"); pred = fit_classifiers(df, sp["train"], sp["eval"], a.classifiers, a.seed)
    ctx = dict(df=df, split=sp, clf=a.classifiers, pred=pred,
               y_ev=df.outcome.to_numpy()[sp["eval"]].astype(int),
               race_ev=df.race.to_numpy()[sp["eval"]], race_ext=df.race.to_numpy()[sp["external"]],
               z_ev=make_proxy(df.iloc[sp["eval"]], a.proxy).to_numpy(),
               z_ext=make_proxy(df.iloc[sp["external"]], a.proxy).to_numpy())
    if "bounds" in a.stages:
        print(f"\n4-6. bounds (proxy {a.proxy})"); stage_truth_and_bounds(ctx, a)
    if "validate" in a.stages:
        print("\n5. validation of the three enumerations"); stage_validate(ctx, a)
    if "compare" in a.stages:
        print(f"\n6. classifier comparison (proxy {a.compare_proxy})"); stage_compare(ctx, a)
    if "inform" in a.stages:
        print("\n6. informativeness"); stage_inform(ctx, a)


if __name__ == "__main__":
    main()