"""
Synthetic study - strengths and limits of the feasible set (grid) approach.

Exact synthetic populations p(z, a, y, d) (no sampling), so every gap is due to the method.
Starting from a base case, one factor is varied at a time:

  factor     values (default)            base
  strength   0, 0.5, 1, 2, 4             1      how informative Z is about A (log-odds scale)
  share      0.05, 0.15, 0.3, 0.5        0.3    total share of the minority groups
  groups     2, 3, 4                     2      number of protected groups (one majority)
  levels     1, 2, 3, 5, 8               3      number of proxy levels K

For each population two internal variables are used (as in exp1):
  D   internal = decision D                       metrics on group rates mu_g = P(D=1 | A=g)
  DY  internal = (D, Y), true positive cell first metrics on the confusion matrix per group

Metrics and the class of each bound (lower, upper); group 1 is compared with group 0 (majority)
  D    DD 1-0           linear, linear
       DI 1/0           vertex, vertex            (ratio of linear functions)
       worst-case DI    vertex, interior          min_g mu_g / max_g mu_g
       max DD gap       interior (convex), vertex max over pairs |mu_g - mu_h|
       I(D;A)           interior (convex), vertex mutual information, bits
  DY   TPRD 1-0         linear-fractional (both)
       PPVD 1-0         linear-fractional (both)
       BA diff 1-0      nonconvex (both)          balanced accuracy difference
       F1 diff 1-0      nonconvex (both)
       I(D;A|Y)         nonconvex (both)          separation, bits

Reference (true) bounds, best available per bound
  Kallus LP       linear and linear-fractional differences (sharp; rates up to --kallus-max-groups)
  vertex          exact enumeration of the vertices of each proxy level's feasible table and all
                  combinations: exact for 'linear' and 'vertex' bounds (and linear-fractional
                  with two groups); skipped above --vertex-cap combinations
  exact LP        minimum of the maximum DD gap (an LP); minimum of I(D;A) when a coupling
                  making D and A independent exists (then it is 0); maximum of worst-case DI
                  (bisection on the LP 'mu_g >= c mu_h for all g, h')
  optimisation    multistart SLSQP over the joint table with the margin constraints: best known
  The grid's own values are also attainable, so a 'best known' reference is never worse than
  the grid. Every reference is labelled exact or best known.

Grid: solve_joint_distribution for each method in --methods and num in --nums (runs above
--max-joints are recorded as infeasible).

Outputs (experiment_results/synthetic/): results.csv (one row per population, run, metric,
method, num), summary.csv, and figures
  fig1_gap_vs_num.png        base case: gap to the reference vs num, by bound class and method
  fig2_width_vs_strength.png reference and grid interval widths as the proxy gets stronger
  fig3_gap_vs_groups_share.png  gap at the largest feasible num vs groups and minority share
  fig4_feasibility.png       largest feasible num vs proxy levels and groups

Usage
  python exp_synthetic.py                    # full study (about 10-20 minutes)
  python exp_synthetic.py --quick            # small test run
  python exp_synthetic.py --methods sequential sweep sequential-all --seeds 5
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
from scipy.optimize import linprog, minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "methods"))
from grid_joint import solve_joint_distribution  # noqa: E402
from kallus_bounds import lp_bounds  # noqa: E402

EPS = 1e-15
BASE = dict(strength=1.0, share=0.3, groups=2, levels=3)
SWEEPS = {"strength": [0.0, 0.5, 1.0, 2.0, 4.0], "share": [0.05, 0.15, 0.3, 0.5],
          "groups": [2, 3, 4], "levels": [1, 2, 3, 5, 8]}
RUNS = {"D": [[(0, 0), (0, 1)], [(1, 0), (1, 1)]],              # index = D
        "DY": [[(1, 1)], [(1, 0)], [(0, 1)], [(0, 0)]]}         # TP, FP, FN, TN
CLASS = {
    "DD 1-0": ("linear", "linear"),
    "DI 1/0": ("vertex", "vertex"),
    "worst-case DI": ("vertex", "interior"),
    "max DD gap": ("interior (convex)", "vertex"),
    "I(D;A)": ("interior (convex)", "vertex"),
    "TPRD 1-0": ("linear-fractional", "linear-fractional"),
    "PPVD 1-0": ("linear-fractional", "linear-fractional"),
    "BA diff 1-0": ("nonconvex", "nonconvex"),
    "F1 diff 1-0": ("nonconvex", "nonconvex"),
    "I(D;A|Y)": ("nonconvex", "nonconvex"),
}
CLASS_ORDER = ["linear", "linear-fractional", "vertex", "interior (convex)", "interior",
               "nonconvex"]


# ------------------------------------------------------------------------- populations
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def make_population(groups, share, levels, strength, seed):
    """Exact p[z, a, y, d]. Group 0 is the majority; the minorities share `share` equally."""
    rng = np.random.default_rng(seed)
    K, G = levels, groups
    pz = rng.dirichlet(np.full(K, 3.0))
    pi = np.r_[1 - share, np.full(G - 1, share / (G - 1))]
    logits = np.log(pi)[None, :] + strength * rng.normal(size=(K, G))
    pa = np.exp(logits - logits.max(1, keepdims=True))
    pa /= pa.sum(1, keepdims=True)
    ey = np.r_[0.0, rng.normal(-0.4, 0.3, G - 1)]
    ed = np.r_[0.0, rng.normal(-0.6, 0.3, G - 1)]
    zeff = rng.normal(size=K)
    p = np.zeros((K, G, 2, 2))
    for z in range(K):
        for a in range(G):
            py1 = sigmoid(0.3 + ey[a] + 0.8 * zeff[z])
            for y in (0, 1):
                pd1 = sigmoid(-1.0 + 2.0 * y + ed[a] + 0.5 * zeff[z])
                for d in (0, 1):
                    p[z, a, y, d] = (pz[z] * pa[z, a] * (py1 if y else 1 - py1)
                                     * (pd1 if d else 1 - pd1))
    return p


def joint_table(p, run):
    """q[z, i, a] for the run's internal variable."""
    K, G = p.shape[:2]
    q = np.zeros((K, len(RUNS[run]), G))
    for i, cells in enumerate(RUNS[run]):
        for d, y in cells:
            q[:, i, :] += p[:, :, y, d]
    return q


# ------------------------------------------------------------------------- metrics
def mi(p, ax, ay, cond=()):
    """Mutual information (bits) between axes ax and ay of p [N, ...], given axes `cond`."""
    other = lambda keep: tuple(a for a in range(1, p.ndim) if a not in keep)
    pxz = p.sum(axis=other((ax,) + tuple(cond)), keepdims=True)
    pyz = p.sum(axis=other((ay,) + tuple(cond)), keepdims=True)
    pz = p.sum(axis=other(tuple(cond)), keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(p > EPS, p * np.log2(p * pz / (pxz * pyz)), 0.0)
    return t.reshape(len(p), -1).sum(1)


def metrics(Q, run):
    """Q [N, z, i, a] -> {metric: values [N]}."""
    th = Q.sum(1)                                                    # [N, i, a]
    G = th.shape[2]
    out = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        if run == "D":
            mu = th[:, 1, :] / th.sum(1)
            out["DD 1-0"] = mu[:, 1] - mu[:, 0]
            out["DI 1/0"] = mu[:, 1] / mu[:, 0]
            out["worst-case DI"] = mu.min(1) / mu.max(1)
            out["max DD gap"] = np.max(np.abs(np.stack(
                [mu[:, g] - mu[:, h] for g, h in itertools.combinations(range(G), 2)])), 0)
            out["I(D;A)"] = mi(th, 1, 2)
        else:
            tp, fp, fn, tn = th[:, 0], th[:, 1], th[:, 2], th[:, 3]
            tpr, ppv, fpr = tp / (tp + fn), tp / (tp + fp), fp / (fp + tn)
            ba, f1 = (tpr + 1 - fpr) / 2, 2 * tp / (2 * tp + fp + fn)
            out["TPRD 1-0"] = tpr[:, 1] - tpr[:, 0]
            out["PPVD 1-0"] = ppv[:, 1] - ppv[:, 0]
            out["BA diff 1-0"] = ba[:, 1] - ba[:, 0]
            out["F1 diff 1-0"] = f1[:, 1] - f1[:, 0]
            p4 = np.stack([tn, fn, fp, tp], 1).reshape(len(th), 2, 2, G)   # [N, d, y, a]
            out["I(D;A|Y)"] = mi(p4, 1, 3, cond=(2,))
    return out


# ------------------------------------------------------------------------- references
def margin_system(r, m):
    """Equality constraints A x = b on x = q[z, i, a].ravel()."""
    K, I = r.shape
    G = m.shape[1]
    idx = np.arange(K * I * G).reshape(K, I, G)
    rows, b = [], []
    for z in range(K):
        for i in range(I):
            row = np.zeros(K * I * G); row[idx[z, i, :]] = 1; rows.append(row); b.append(r[z, i])
        for a in range(G):
            row = np.zeros(K * I * G); row[idx[z, :, a]] = 1; rows.append(row); b.append(m[z, a])
    return np.array(rows), np.array(b)


def cell_vertices(rows, cols):
    """All vertices of {X >= 0 : X 1 = rows, 1' X = cols} by basis enumeration."""
    mm, n = len(rows), len(cols)
    A = np.zeros((mm + n, mm * n))
    for i in range(mm):
        A[i, i * n:(i + 1) * n] = 1
    for j in range(n):
        A[mm + j, j::n] = 1
    b = np.r_[rows, cols]
    V = []
    for S in itertools.combinations(range(mm * n), mm + n - 1):
        AS = A[:, S]
        x, *_ = np.linalg.lstsq(AS, b, rcond=None)
        if (x < -1e-12).any() or np.abs(AS @ x - b).max() > 1e-10:
            continue
        v = np.zeros(mm * n); v[list(S)] = np.maximum(x, 0)
        V.append(v.reshape(mm, n))
    return np.unique(np.round(np.array(V), 14), axis=0)


def vertex_refs(r, m, run, cap, chunk=200_000):
    """Exact min / max over all vertex combinations, and the extreme points, or None."""
    V = [cell_vertices(r[z], m[z]) for z in range(len(r))]
    sizes = [len(v) for v in V]
    total = int(np.prod(sizes))
    if total > cap:
        return None, V, total
    best = {}
    for start in range(0, total, chunk):
        idx = np.unravel_index(np.arange(start, min(total, start + chunk)), sizes)
        Q = np.stack([V[z][idx[z]] for z in range(len(V))], 1)       # [n, z, i, a]
        for k, v in metrics(Q, run).items():
            ok = ~np.isnan(v)
            if not ok.any():
                continue
            lo_i, hi_i = np.nanargmin(np.where(ok, v, np.inf)), np.nanargmax(np.where(ok, v, -np.inf))
            cur = best.get(k)
            if cur is None or v[lo_i] < cur[0]:
                cur = (v[lo_i], cur[1] if cur else -np.inf, Q[lo_i], cur[3] if cur else None)
            if v[hi_i] > cur[1]:
                cur = (cur[0], v[hi_i], cur[2], Q[hi_i])
            best[k] = cur
    return best, V, total


def kallus_refs(r, m, run, G, max_groups, n_grid):
    out = {}
    if run == "D":
        df = lp_bounds(r, m, np.array([False, True]), np.array([True, True]), ref=0,
                       n_grid=n_grid)
        row = df[df.group == 1].iloc[0]
        out["DD 1-0"] = (row.lower, row.upper)
    elif G <= max_groups:
        target = np.array([True, False, False, False])
        for name, cond in (("TPRD 1-0", [True, False, True, False]),
                           ("PPVD 1-0", [True, True, False, False])):
            df = lp_bounds(r, m, target, np.array(cond), ref=0, n_grid=n_grid)
            row = df[df.group == 1].iloc[0]
            out[name] = (row.lower, row.upper)
    return out


def optimise(r, m, run, metric, sense, starts, maxiter):
    """Best value of `metric` (sense +1 = minimise, -1 = maximise) from multistart SLSQP."""
    K, I = r.shape
    G = m.shape[1]
    A, b = margin_system(r, m)

    def f(x):
        v = metrics(np.maximum(x, 0).reshape(1, K, I, G), run)[metric][0]
        return sense * v if np.isfinite(v) else 1e6

    best = np.nan
    for x0 in starts:
        with np.errstate(all="ignore"):
            res = minimize(f, x0.ravel(), method="SLSQP", bounds=[(0, None)] * A.shape[1],
                           constraints=[{"type": "eq", "fun": lambda x: A @ x - b,
                                         "jac": lambda x: A}],
                           options=dict(maxiter=maxiter, ftol=1e-12))
        x = np.maximum(res.x, 0)
        if np.abs(A @ x - b).max() > 1e-8:
            continue
        v = metrics(x.reshape(1, K, I, G), run)[metric][0]
        if np.isfinite(v) and (np.isnan(best) or sense * v < sense * best):
            best = v
    return best


def exact_convex_minima(r, m):
    """
    Exact minima for run D:
      max DD gap   LP: min t s.t. |mu_g - mu_h| <= t, mu_g = sum_z q[z,1,g] / P(A=g)
      I(D;A)       0 if a coupling with p(d, a) = p(d) p(a) exists (LP feasibility), else None
    """
    K, I = r.shape
    G = m.shape[1]
    A, b = margin_system(r, m)
    n = A.shape[1]
    idx = np.arange(n).reshape(K, I, G)
    pa = m.sum(0)
    out = {}
    # max DD gap: variables x = (q, t)
    A_ub, b_ub = [], []
    for g, h in itertools.combinations(range(G), 2):
        row = np.zeros(n + 1)
        row[idx[:, 1, g]] = 1 / pa[g]; row[idx[:, 1, h]] = -1 / pa[h]
        for s_ in (1, -1):
            rr = s_ * row.copy(); rr[-1] = -1
            A_ub.append(rr); b_ub.append(0.0)
    c = np.zeros(n + 1); c[-1] = 1
    res = linprog(c, A_ub=np.array(A_ub), b_ub=b_ub, A_eq=np.c_[A, np.zeros(len(A))], b_eq=b,
                  bounds=[(0, None)] * (n + 1), method="highs")
    out["max DD gap"] = res.fun if res.status == 0 else np.nan
    # I(D;A) = 0 feasible?
    pd_ = r.sum(0)
    rows, rhs = [], []
    for i in range(I):
        for g in range(G):
            row = np.zeros(n); row[idx[:, i, g]] = 1; rows.append(row); rhs.append(pd_[i] * pa[g])
    res = linprog(np.zeros(n), A_eq=np.r_[A, np.array(rows)], b_eq=np.r_[b, rhs],
                  bounds=[(0, None)] * n, method="highs")
    out["I(D;A)"] = 0.0 if res.status == 0 else None
    return out


def exact_worst_case_di_max(r, m, iters=50):
    """max c such that mu_g >= c mu_h for all g, h is feasible (bisection on LP feasibility)."""
    K, I = r.shape
    G = m.shape[1]
    A, b = margin_system(r, m)
    n = A.shape[1]
    idx = np.arange(n).reshape(K, I, G)
    pa = m.sum(0)

    def feasible(c):
        A_ub = []
        for g in range(G):
            for h in range(G):
                if g != h:                         # c mu_h - mu_g <= 0
                    row = np.zeros(n)
                    row[idx[:, 1, h]] += c / pa[h]; row[idx[:, 1, g]] -= 1 / pa[g]
                    A_ub.append(row)
        res = linprog(np.zeros(n), A_ub=np.array(A_ub), b_ub=np.zeros(len(A_ub)), A_eq=A,
                      b_eq=b, bounds=[(0, None)] * n, method="highs")
        return res.status == 0

    lo, hi = 0.0, 1.0
    if feasible(1.0):
        return 1.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if feasible(mid) else (lo, mid)
    return lo


def random_starts(V, n, rng):
    """Feasible starting tables: random convex combinations of each level's vertices."""
    out = []
    for _ in range(n):
        out.append(np.stack([(rng.dirichlet(np.ones(len(v)))[:, None, None] * v).sum(0)
                             for v in V]))
    return out


# ------------------------------------------------------------------------- one population
def run_population(cfg, seed, a):
    p = make_population(cfg["groups"], cfg["share"], cfg["levels"], cfg["strength"], seed)
    rng = np.random.default_rng(10_000 + seed)
    rows = []
    for run in ("D", "DY"):
        q = joint_table(p, run)
        r, m = q.sum(2), q.sum(1)
        K, I, G = q.shape
        truth = {k: float(v[0]) for k, v in metrics(q[None], run).items()}

        # references
        t0 = time.perf_counter()
        vert, V, n_vert = vertex_refs(r, m, run, a.vertex_cap)
        kal = kallus_refs(r, m, run, G, a.kallus_max_groups, a.lp_grid)
        convex = exact_convex_minima(r, m) if run == "D" else {}
        wcdi_max = exact_worst_case_di_max(r, m) if run == "D" else None
        ref = {}
        for metric, (cl_lo, cl_hi) in CLASS.items():
            if metric not in truth:
                continue
            for side, cl in (("lower", cl_lo), ("upper", cl_hi)):
                sense = 1 if side == "lower" else -1
                cands, exact = [], False
                if metric in kal:
                    cands.append(kal[metric][0 if side == "lower" else 1]); exact = True
                if vert is not None and metric in vert:
                    cands.append(vert[metric][0 if side == "lower" else 1])
                    exact |= cl in ("linear", "vertex") or (cl == "linear-fractional" and G == 2)
                if side == "lower" and convex.get(metric) is not None:
                    ref[(metric, side)] = (convex[metric], True)          # exact LP
                    continue
                if metric == "worst-case DI" and side == "upper" and wcdi_max is not None:
                    ref[(metric, side)] = (wcdi_max, True)                # exact LP bisection
                    continue
                need_opt = cl in ("interior (convex)", "interior", "nonconvex") or (
                    cl == "linear-fractional" and metric not in kal)
                if need_opt:
                    starts = random_starts(V, a.starts, rng)
                    if vert is not None and metric in vert:
                        starts.append(vert[metric][2 if side == "lower" else 3])
                    v = optimise(r, m, run, metric, sense, starts, a.maxiter)
                    if np.isfinite(v):
                        cands.append(v)
                cands = [c for c in cands if np.isfinite(c)]
                ref[(metric, side)] = ((min(cands) if side == "lower" else max(cands))
                                       if cands else np.nan, exact)
        t_ref = time.perf_counter() - t0

        # grid
        grid = []
        for method in a.methods:
            for num in a.nums:
                p_primary = {(i, z): r[z, i] for z in range(K) for i in range(I)}
                p_aux = {(z, g): m[z, g] for z in range(K) for g in range(G)}
                t0 = time.perf_counter()
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        J, _ = solve_joint_distribution(p_primary, p_aux, num=num, method=method,
                                                        max_joints=a.max_joints, as_dicts=False)
                    vals = metrics(J, run)
                    grid.append((method, num, len(J), time.perf_counter() - t0, vals))
                except SystemExit:
                    grid.append((method, num, np.nan, time.perf_counter() - t0, None))

        # best-known references also include every grid value (all are attainable)
        for (metric, side), (val, exact) in list(ref.items()):
            if exact:
                continue
            gv = [np.nanmin(v[metric]) if side == "lower" else np.nanmax(v[metric])
                  for *_, v in grid if v is not None and np.isfinite(v[metric]).any()]
            if gv:
                val = np.nanmin([val] + gv) if side == "lower" else np.nanmax([val] + gv)
            ref[(metric, side)] = (val, exact)

        for method, num, nj, sec, vals in grid:
            for metric in truth:
                lo = hi = np.nan
                if vals is not None:
                    v = vals[metric][~np.isnan(vals[metric])]
                    if v.size:
                        lo, hi = v.min(), v.max()
                (rl, el), (ru, eu) = ref[(metric, "lower")], ref[(metric, "upper")]
                rows.append(dict(**cfg, seed=seed, run=run, metric=metric, method=method,
                                 num=num, feasible=vals is not None, n_joints=nj, seconds=sec,
                                 truth=truth[metric], lower=lo, upper=hi,
                                 ref_lower=rl, ref_upper=ru, ref_lower_exact=el,
                                 ref_upper_exact=eu, class_lower=CLASS[metric][0],
                                 class_upper=CLASS[metric][1], n_vertex_combos=n_vert,
                                 ref_seconds=t_ref))
    return rows


# ------------------------------------------------------------------------- summaries and plots
def long_form(res):
    """One row per bound side, with gap = how far the grid is inside the reference."""
    out = []
    for side in ("lower", "upper"):
        d = res.copy()
        d["side"] = side
        d["class"] = d[f"class_{side}"]
        d["grid"] = d[side]
        d["ref"] = d[f"ref_{side}"]
        d["ref_exact"] = d[f"ref_{side}_exact"]
        d["gap"] = (d.grid - d.ref) if side == "lower" else (d.ref - d.grid)
        out.append(d)
    out = pd.concat(out, ignore_index=True)
    inf = ~np.isfinite(out.ref) & ~np.isfinite(out.grid) & (out.ref == out.grid)
    out.loc[inf, "gap"] = 0.0                                          # both infinite
    return out


def at_best_num(L):
    """Rows at the largest feasible num per (population, run, metric, method)."""
    keys = ["factor", "value", "seed", "run", "metric", "method"]
    f = L[L.feasible]
    return f.loc[f.groupby(keys + ["side"]).num.idxmax()]


def plot_all(res, L, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = a.out
    colors = dict(zip(CLASS_ORDER, ["tab:blue", "tab:cyan", "tab:green", "tab:orange",
                                    "tab:red", "tab:purple"]))

    # fig 1: base case, gap vs num by class
    base = L[(L.factor == "strength") & (L.value == BASE["strength"]) & L.feasible]
    methods = list(dict.fromkeys(base.method))
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 3.8), squeeze=False,
                             sharey=True)
    for ax, method in zip(axes[0], methods):
        d = base[base.method == method]
        for cl in CLASS_ORDER:
            dc = d[d["class"] == cl]
            g = dc.groupby("num").gap.mean()
            if len(g):
                exact = dc.ref_exact.all()
                ax.plot(g.index, np.maximum(g.values, 1e-7), "o-" if exact else "o--", ms=4,
                        color=colors[cl], label=cl + ("" if exact else " (best-known ref.)"))
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(f"{method} (base case)")
        ax.set_xlabel("num"); ax.set_xticks(a.nums); ax.set_xticklabels(a.nums)
        ax.minorticks_off()
    axes[0][0].set_ylabel("mean gap to reference (floor 1e-7)")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig1_gap_vs_num.png"), dpi=150)
    plt.close(fig)

    # fig 2: widths vs strength (reference vs grid at the largest feasible num)
    S = res[(res.factor == "strength") & res.feasible & (res.method == a.methods[0])]
    S = S.loc[S.groupby(["value", "seed", "run", "metric"]).num.idxmax()]
    S = S.assign(ref_width=S.ref_upper - S.ref_lower, grid_width=S.upper - S.lower)
    mets = list(CLASS)
    fig, axes = plt.subplots(2, 5, figsize=(17, 6.4))
    for ax, metric in zip(axes.ravel(), mets):
        g = S[S.metric == metric].replace([np.inf, -np.inf], np.nan).groupby("value")[
            ["ref_width", "grid_width"]].mean()
        ax.plot(g.index, g.ref_width, "k-o", ms=4, label="reference")
        ax.plot(g.index, g.grid_width, "--s", ms=4, color="tab:orange",
                label=f"grid ({a.methods[0]})")
        ax.set_title(metric, fontsize=9); ax.set_xlabel("proxy strength")
    axes[0][0].set_ylabel("interval width"); axes[1][0].set_ylabel("interval width")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_width_vs_strength.png"), dpi=150)
    plt.close(fig)

    # fig 3: gap at the largest feasible num vs groups and share
    B = at_best_num(L)
    B = B[B.method == a.methods[0]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, factor in zip(axes, ("groups", "share")):
        d = B[B.factor == factor]
        for cl in CLASS_ORDER:
            g = d[d["class"] == cl].groupby("value").gap.mean()
            if len(g):
                ax.plot(g.index, np.maximum(g.values, 1e-7), "o-", ms=4, color=colors[cl],
                        label=cl)
        ax.set_yscale("log"); ax.set_xlabel(factor)
        ax.set_title(f"gap at largest feasible num ({a.methods[0]})", fontsize=9)
    axes[0].set_ylabel("mean gap (floor 1e-7)"); axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig3_gap_vs_groups_share.png"), dpi=150)
    plt.close(fig)

    # fig 4: feasibility
    F = res[res.feasible].groupby(["factor", "value", "run", "method"]).num.max().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, factor in zip(axes, ("levels", "groups")):
        d = F[F.factor == factor]
        for (run, method), g in d.groupby(["run", "method"]):
            ax.plot(g.value, g.num, "o-", ms=4, label=f"internal {run}, {method}")
        ax.set_yscale("log"); ax.set_xlabel(factor); ax.set_yticks(a.nums)
        ax.set_yticklabels(a.nums); ax.minorticks_off()
        ax.set_title(f"largest feasible num (max {a.max_joints:.0e} joints)", fontsize=9)
    axes[0].set_ylabel("num"); axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig4_feasibility.png"), dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--methods", nargs="+", default=["sequential", "sweep"])
    ap.add_argument("--nums", type=int, nargs="+", default=[2, 3, 5, 9, 17, 33])
    ap.add_argument("--max-joints", type=int, default=2_000_000)
    ap.add_argument("--vertex-cap", type=int, default=2_000_000)
    ap.add_argument("--kallus-max-groups", type=int, default=3)
    ap.add_argument("--lp-grid", type=int, default=21)
    ap.add_argument("--starts", type=int, default=6, help="random starts for optimisation")
    ap.add_argument("--maxiter", type=int, default=200)
    ap.add_argument("--factors", nargs="+", default=list(SWEEPS), choices=list(SWEEPS))
    ap.add_argument("--quick", action="store_true", help="1 seed, few values, small nums")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent
                                         / "experiment_results" / "synthetic"))
    a = ap.parse_args()
    sweeps = {f: SWEEPS[f] for f in a.factors}
    if "strength" not in sweeps:
        sweeps = {"strength": [BASE["strength"]], **sweeps}            # base case for fig 1
    if a.quick:
        a.seeds, a.nums, a.starts = 1, [2, 3, 5, 9], 3
        sweeps = {"strength": [0.0, 1.0, 4.0], "share": [0.05, 0.3], "groups": [2, 3],
                  "levels": [1, 3, 5]}
    os.makedirs(a.out, exist_ok=True)

    cache, rows = {}, []
    t_start = time.perf_counter()
    for factor, values in sweeps.items():
        for value in values:
            cfg = dict(BASE, **{factor: value})
            for seed in range(a.seeds):
                key = (tuple(sorted(cfg.items())), seed)
                if key not in cache:
                    t0 = time.perf_counter()
                    cache[key] = run_population(cfg, seed, a)
                    print(f"  {factor}={value} seed={seed}: {time.perf_counter() - t0:.1f}s "
                          f"(total {time.perf_counter() - t_start:.0f}s)", flush=True)
                rows += [dict(r, factor=factor, value=value) for r in cache[key]]
    res = pd.DataFrame(rows)
    res["covers_truth"] = (res.lower <= res.truth + 1e-9) & (res.truth <= res.upper + 1e-9)
    res.to_csv(os.path.join(a.out, "results.csv"), index=False)

    L = long_form(res)
    B = at_best_num(L)
    summ = (B.groupby(["factor", "value", "run", "class", "method"])
             .agg(bounds=("gap", "size"), mean_gap=("gap", "mean"), max_gap=("gap", "max"),
                  share_exact_ref=("ref_exact", "mean"), largest_num=("num", "max"))
             .reset_index())
    cov = (res[res.feasible].loc[lambda d: d.groupby(["factor", "value", "seed", "run",
                                                      "metric", "method"]).num.transform("max")
                                  == d.num]
           .groupby(["factor", "value", "run", "method"]).covers_truth.mean()
           .rename("coverage").reset_index())
    summ = summ.merge(cov, on=["factor", "value", "run", "method"], how="left")
    summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    plot_all(res, L, a)

    neg = L[L.feasible & (L.gap < -1e-6) & L.ref_exact]
    print(f"\ngrid bounds outside an exact reference (should be none): {len(neg)}")
    with pd.option_context("display.width", 200, "display.max_rows", 300):
        base = B[(B.factor == "strength") & (B.value == BASE["strength"])]
        print("\nBase case, gap at the largest feasible num, by bound class:")
        print(base.groupby(["class", "method"]).agg(mean_gap=("gap", "mean"),
                                                    max_gap=("gap", "max"),
                                                    exact_ref=("ref_exact", "mean"),
                                                    num=("num", "max"))
              .round(4).to_string())
    print(f"\nsaved to {a.out}/ ({time.perf_counter() - t_start:.0f}s)")


if __name__ == "__main__":
    main()