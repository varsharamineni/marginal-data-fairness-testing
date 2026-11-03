"""
Global optimisation baseline: certified bounds with SCIP (spatial branch-and-bound).

For each metric and side (lower / upper) we solve

    min / max  metric(q)   over joint tables q[z, i, a] >= 0
    s.t.       sum_a q[z, i, a] = p(z, i)      (primary data)
               sum_i q[z, i, a] = p(z, a)      (auxiliary data)

SCIP returns two numbers:
  value  the objective of the best feasible joint found (attainable: an inner bound, like the
         grid's)
  bound  SCIP's proven dual bound: no feasible joint does better (an outer bound, valid even
         when the time limit is reached)
When |value - bound| <= the gap tolerance the result is the exact sharp bound.

Metrics (same definitions as experiments/exp_synthetic.py; group 1 vs group 0)
  internal D       rates mu_g = P(D=1 | A=g) are linear in q (P(A=g) is fixed)
    DD 1-0            mu_1 - mu_0                                  linear
    DI 1/0            mu_1 / mu_0                                  t * mu_0 = mu_1
    worst-case DI     min_g mu_g / max_g mu_g                      lower: min over ordered
                                                                   pairs of mu_g / mu_h;
                                                                   upper: max t s.t.
                                                                   mu_g >= t mu_h for all g, h
    max DD gap        max over pairs |mu_g - mu_h|                 lower: min t s.t. |.| <= t;
                                                                   upper: max over ordered
                                                                   pairs of mu_g - mu_h
    I(D;A)            mutual information (bits)                    x log x terms
  internal (D, Y)  internal order TP, FP, FN, TN (as in exp_synthetic)
    TPRD 1-0, PPVD 1-0, BA diff 1-0, F1 diff 1-0   each rate r = num / den is a variable with
                                                   r * den = num (bilinear) and den >= eps
    I(D;A|Y)          conditional mutual information (bits)      x log x terms
Rates are only defined where their denominator is positive (den >= 1e-9), matching the
grid code, which drops joints where a metric is undefined. In x log x terms the probabilities
are bounded below by 1e-12, which changes mutual information by less than 1e-10.

Interface (shared with the other baselines)
    r : [Z, I]  p(z, internal value)   from the primary data
    m : [Z, G]  p(z, a)                from the auxiliary data
    global_bound(r, m, run, metric, side, time_limit, gap) -> dict(value, bound, status,
                                                                   seconds, q)
    global_bounds(r, m, run, metrics=None, ...) -> {metric: dict(lower=..., upper=...)}

Requires PySCIPOpt:  pip install pyscipopt
Self-test:          python global_opt.py
"""
import itertools
import math
import time

import numpy as np
from pyscipopt import Model, log, quicksum

from ci_baseline import harmonize

METRICS = {"D": ["DD 1-0", "DI 1/0", "worst-case DI", "max DD gap", "I(D;A)"],
           "DY": ["TPRD 1-0", "PPVD 1-0", "BA diff 1-0", "F1 diff 1-0", "I(D;A|Y)"]}
DEN_EPS, LOG_EPS, RATIO_UB = 1e-9, 1e-12, 1e4
LN2 = math.log(2)


# ------------------------------------------------------------------------- model building
def _base_model(r, m, time_limit, gap, feastol=1e-7):
    K, I = r.shape
    G = m.shape[1]
    model = Model()
    model.hideOutput()
    model.setParam("limits/time", time_limit)
    model.setParam("limits/gap", gap)
    if feastol is not None:
        model.setParam("numerics/feastol", feastol)  # margins are probabilities
    q = {(z, i, a): model.addVar(lb=0, ub=float(min(r[z, i], m[z, a])) + 1e-12)
         for z in range(K) for i in range(I) for a in range(G)}
    for z in range(K):
        for i in range(I):
            model.addCons(quicksum(q[z, i, a] for a in range(G)) == float(r[z, i]))
        for a in range(G):
            model.addCons(quicksum(q[z, i, a] for i in range(I)) == float(m[z, a]))
    th = {(i, a): quicksum(q[z, i, a] for z in range(K)) for i in range(I) for a in range(G)}
    return model, q, th


def _rate(model, num, den, ub=1.0):
    """Variable rho with rho * den = num and den >= DEN_EPS."""
    rho = model.addVar(lb=0, ub=ub)
    model.addCons(den >= DEN_EPS)
    model.addCons(rho * den == num)
    return rho


def _xlogx_sum(model, exprs):
    """Sum of x log x over linear expressions, each tied to a variable x >= LOG_EPS."""
    terms = []
    for e in exprs:
        x = model.addVar(lb=LOG_EPS, ub=1.0)
        model.addCons(x == e)
        terms.append(x * log(x))
    return quicksum(terms)


def _xlogx_const(p):
    p = np.asarray(p, float)
    p = p[p > 0]
    return float((p * np.log(p)).sum())


def _objective(model, th, r, m, run, metric, pair=None):
    """Return (objective expression or variable, needs_epigraph)."""
    G = m.shape[1]
    if run == "D":
        pa = m.sum(0)
        mu = [th[1, g] * (1.0 / pa[g]) for g in range(G)]
        if metric == "DD 1-0":
            return mu[1] - mu[0], False
        if metric == "DI 1/0" or (metric == "worst-case DI" and pair is not None):
            g, h = pair if pair is not None else (1, 0)
            return _rate(model, mu[g], mu[h], ub=RATIO_UB), False
        if metric == "worst-case DI":                       # upper: max t, mu_g >= t mu_h
            t = model.addVar(lb=0, ub=1.0)
            for g in range(G):
                for h in range(G):
                    if g != h:
                        model.addCons(mu[g] >= t * mu[h])
            return t, False
        if metric == "max DD gap" and pair is not None:     # upper part: mu_g - mu_h
            g, h = pair
            return mu[g] - mu[h], False
        if metric == "max DD gap":                          # lower: min t, |mu_g - mu_h| <= t
            t = model.addVar(lb=0, ub=1.0)
            for g, h in itertools.combinations(range(G), 2):
                model.addCons(mu[g] - mu[h] <= t)
                model.addCons(mu[h] - mu[g] <= t)
            return t, False
        if metric == "I(D;A)":
            I = r.shape[1]
            expr = _xlogx_sum(model, [th[i, a] for i in range(I) for a in range(G)])
            const = -_xlogx_const(r.sum(0)) - _xlogx_const(m.sum(0))
            return (expr + const) * (1 / LN2), True
    else:
        TP, FP, FN, TN = 0, 1, 2, 3
        tp = [th[TP, g] for g in range(G)]
        fp = [th[FP, g] for g in range(G)]
        fn = [th[FN, g] for g in range(G)]
        tn = [th[TN, g] for g in range(G)]
        if metric == "TPRD 1-0":
            v = [_rate(model, tp[g], tp[g] + fn[g]) for g in (0, 1)]
            return v[1] - v[0], False
        if metric == "PPVD 1-0":
            v = [_rate(model, tp[g], tp[g] + fp[g]) for g in (0, 1)]
            return v[1] - v[0], False
        if metric == "BA diff 1-0":
            ba = []
            for g in (0, 1):
                tpr = _rate(model, tp[g], tp[g] + fn[g])
                fpr = _rate(model, fp[g], fp[g] + tn[g])
                ba.append(0.5 * (tpr + 1 - fpr))
            return ba[1] - ba[0], False
        if metric == "F1 diff 1-0":
            v = [_rate(model, 2 * tp[g], 2 * tp[g] + fp[g] + fn[g]) for g in (0, 1)]
            return v[1] - v[0], False
        if metric == "I(D;A|Y)":
            cells = {(1, 1): TP, (1, 0): FP, (0, 1): FN, (0, 0): TN}       # (d, y) -> index
            expr = _xlogx_sum(model, [th[cells[d, y], a] for (d, y) in cells for a in range(G)])
            expr = expr - _xlogx_sum(model, [th[cells[1, y], a] + th[cells[0, y], a]
                                             for y in (0, 1) for a in range(G)])
            pdy = r.sum(0)                                                  # fixed p(d, y)
            py = [pdy[cells[1, y]] + pdy[cells[0, y]] for y in (0, 1)]
            const = -_xlogx_const(pdy) + _xlogx_const(py)
            return (expr + const) * (1 / LN2), True
    raise ValueError(f"unknown metric {metric} for internal {run}")


# ------------------------------------------------------------------------- solving
def _solve(r, m, run, metric, side, time_limit, gap, pair=None):
    """Solve once with a tight tolerance; on a numerical failure retry with SCIP's defaults."""
    last = None
    for feastol in (1e-7, None):
        try:
            return _solve_once(r, m, run, metric, side, time_limit, gap, pair, feastol)
        except Exception as e:                          # e.g. 'SCIP: error in LP solver!'
            last = e
    return dict(value=np.nan, bound=np.nan, status=f"error: {last}", seconds=np.nan, q=None)


def _solve_once(r, m, run, metric, side, time_limit, gap, pair, feastol):
    model, q, th = _base_model(r, m, time_limit, gap, feastol)
    obj, nonlinear = _objective(model, th, r, m, run, metric, pair)
    if nonlinear or not hasattr(obj, "getLPPos"):          # epigraph for expressions
        z = model.addVar(lb=-1e6, ub=1e6)
        model.addCons(z == obj)
        obj = z
    model.setObjective(obj, "minimize" if side == "lower" else "maximize")
    t0 = time.perf_counter()
    model.optimize()
    sec = time.perf_counter() - t0
    status = model.getStatus()
    value = model.getObjVal() if model.getNSols() > 0 else np.nan
    bound = model.getDualbound()
    qsol = None
    if model.getNSols() > 0:
        K, I = r.shape
        G = m.shape[1]
        qsol = np.array([[[model.getVal(q[z, i, a]) for a in range(G)] for i in range(I)]
                         for z in range(K)])
    return dict(value=value, bound=bound, status=status, seconds=sec, q=qsol)


def global_bound(r, m, run, metric, side, time_limit=60.0, gap=1e-6, harmonize_margins=True):
    """Certified bound of one metric on one side: dict(value, bound, status, seconds, q)."""
    if harmonize_margins:
        r, m = harmonize(r, m)
    r, m = np.asarray(r, float), np.asarray(m, float)
    G = m.shape[1]
    # sides that decompose over ordered pairs of groups
    pairs = [(g, h) for g in range(G) for h in range(G) if g != h]
    if (metric == "worst-case DI" and side == "lower") or (metric == "max DD gap"
                                                          and side == "upper"):
        res = [_solve(r, m, run, metric, side, time_limit, gap, pair=p) for p in pairs]
        pick = min if side == "lower" else max
        best = pick(res, key=lambda d: d["value"] if np.isfinite(d["value"]) else
                    (np.inf if side == "lower" else -np.inf))
        return dict(value=best["value"], bound=pick(d["bound"] for d in res),
                    status=",".join(sorted({d["status"] for d in res})),
                    seconds=sum(d["seconds"] for d in res), q=best["q"])
    return _solve(r, m, run, metric, side, time_limit, gap)


def global_bounds(r, m, run, metrics=None, time_limit=60.0, gap=1e-6, verbose=False):
    """{metric: {'lower': result, 'upper': result}} for all metrics of the run."""
    out = {}
    for metric in metrics or METRICS[run]:
        out[metric] = {}
        for side in ("lower", "upper"):
            res = global_bound(r, m, run, metric, side, time_limit, gap)
            out[metric][side] = res
            if verbose:
                print(f"  {metric:14s} {side}: value {res['value']:.6f}  bound "
                      f"{res['bound']:.6f}  [{res['status']}, {res['seconds']:.1f}s]")
    return out


# ------------------------------------------------------------------------- self-test
def _self_test():
    from vertex_sampling import vertex_bounds

    rng = np.random.default_rng(3)
    K, G = 2, 2
    pz = rng.dirichlet(np.ones(K) * 3)
    pa = rng.dirichlet(np.ones(G), size=K) * pz[:, None]
    # internal D: linear and vertex-attained bounds must equal the vertex enumeration
    r = rng.dirichlet(np.ones(2), size=K) * pz[:, None]

    def rates(Q):
        th = Q.sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            mu = th[:, 1, :] / th.sum(1)
        return {"DD 1-0": mu[:, 1] - mu[:, 0], "DI 1/0": mu[:, 1] / mu[:, 0]}

    vb, _ = vertex_bounds(r, pa, rates)
    gb = global_bounds(r, pa, "D", metrics=["DD 1-0", "DI 1/0"], time_limit=30)
    for k in ("DD 1-0", "DI 1/0"):
        for j, side in enumerate(("lower", "upper")):
            g = gb[k][side]
            assert abs(g["value"] - vb[k][j]) < 1e-5, (k, side, g, vb[k])
            assert (g["bound"] <= g["value"] + 1e-6) if side == "lower" else \
                   (g["bound"] >= g["value"] - 1e-6)
        print(f"{k}: SCIP [{gb[k]['lower']['value']:.5f}, {gb[k]['upper']['value']:.5f}] = "
              f"vertices [{vb[k][0]:.5f}, {vb[k][1]:.5f}]: OK")
    # I(D;A) minimum is 0 when independence is feasible (always true with one proxy level)
    r1, m1 = r.sum(0, keepdims=True), pa.sum(0, keepdims=True)
    res = global_bound(r1, m1, "D", "I(D;A)", "lower", time_limit=30)
    assert res["value"] < 1e-6, res
    print(f"I(D;A) minimum with one proxy level: {res['value']:.2e} (exact 0): OK")
    print("the returned joint reproduces the margins:",
          np.allclose(res["q"].sum(2), harmonize(r1, m1)[0], atol=1e-6))


if __name__ == "__main__":
    _self_test()