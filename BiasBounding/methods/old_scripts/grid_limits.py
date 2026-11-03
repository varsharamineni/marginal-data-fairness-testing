"""
Limits of the grid (joint-enumeration) method: for which metrics it is exact, when it is
feasible, and what the enumerated joints reveal beyond per-metric bounds.

  exactness    per metric, grid bounds vs an exact (or best-known) reference, as num grows
  feasibility  number of joints vs proxy levels, internal size, groups and num, counted
               without materialising them; plus the paper's real proxies (HMDA, warfarin)
  groups       three protected groups: is 'sequential-all' still exact? (vs the Kallus LP)
  extras       what the joints add: joint regions of metric pairs, conditional bounds
               ("range of TPRD if |DP| <= eps"), external constraints (known group base
               rates), and why grid frequencies are not probabilities. Each against an LP.

Usage (repo root):  python methods/grid_limits.py all [--quick] [--kallus-repo ~/kallus]
Results: results/grid_limits/ (CSV, PNG).
"""
import argparse
import itertools
import math
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog, minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid_joint import cell_tables                                       # noqa: E402
from kallus_bounds import CouplingLP, lp_bounds                           # noqa: E402
from BiasBounding.methods.old_scripts.experiments import (run_grid, kallus_inputs, kallus_bounds_all, dy_masks,     # noqa: E402
                         metric_values, max_gap_lp_lower, random_feasible_joints,
                         rate_specs, plt)


# ============================================================================ problems
def problem(Z, n_internal=4, G=2, seed=0, strength=1.0):
    """Population with Z common values, n_internal internal values (2 = decision,
    4 = (decision, label), 8 = (decision, label, extra binary)) and G groups."""
    rng = np.random.default_rng(seed)
    bits = int(round(math.log2(n_internal)))
    I = [tuple(t) for t in itertools.product((0, 1), repeat=bits)] if bits > 1 else [0, 1]
    C = [f"z{k:03d}" for k in range(Z)]
    E = [f"g{g}" for g in range(G)]
    pc = rng.dirichlet(np.ones(Z) * 3)
    pi = rng.dirichlet(np.ones(len(I)) * 2, Z)
    logits = rng.normal(0, 1.5 * strength, (Z, 1, G)) + rng.normal(0, 1, (1, len(I), G))
    pg = np.exp(logits) / np.exp(logits).sum(2, keepdims=True)
    joint = pc[:, None, None] * pi[:, :, None] * pg                      # [Z, I, G]
    pp = {(i, c): joint[k, a].sum() for k, c in enumerate(C) for a, i in enumerate(I)}
    pa = {(c, e): joint[k, :, b].sum() for k, c in enumerate(C) for b, e in enumerate(E)}
    return pp, pa


def _marg(pp, pa):
    """Per-common-value row and column sums exactly as grid_joint builds them."""
    srt = lambda v: sorted(v, key=str)
    I, C, E = srt({k[0] for k in pp}), srt({k[0] for k in pa}), srt({k[1] for k in pa})
    Paux = np.array([[pa.get((c, e), 0) for e in E] for c in C], float)
    Pprim = np.array([[pp.get((i, c), 0) for i in I] for c in C], float)
    keep = Pprim.sum(1) > 0
    Paux, Pprim = Paux[keep] / Paux[keep].sum(), Pprim[keep]
    rows = Pprim / Pprim.sum(1, keepdims=True) * Paux.sum(1, keepdims=True)
    return rows, Paux


def cell_log_counts(pp, pa, num, method, max_cells=None):
    """log10 of the number of tables per common value, for each fill order (one order unless
    'sequential-all'). Only the first max_cells common values are counted if given."""
    rows, cols = _marg(pp, pa)
    m, n = rows.shape[1], cols.shape[1]
    if num ** ((m - 1) * (n - 1)) > 1e5:
        return None
    ks = range(len(rows) if max_cells is None else min(max_cells, len(rows)))
    orders = ([[i for i in range(m) if i != last] + [last] for last in range(m)]
              if method == "sequential-all" else [list(range(m))])
    meth = "sequential" if method == "sequential-all" else method
    return [np.array([math.log10(max(len(cell_tables(rows[k, o], cols[k], num, meth)), 1))
                      for k in ks]) for o in orders]


def _combine(per_order_logsum):
    """log10 of sum over orders of 10**logsum (orders are pooled, not deduplicated)."""
    mx = max(per_order_logsum)
    return mx + math.log10(sum(10 ** (v - mx) for v in per_order_logsum))


def log10_joints(pp, pa, num, method):
    """Exact log10 of the number of joints grid_joint would build, without building them."""
    lc = cell_log_counts(pp, pa, num, method)
    return None if lc is None else _combine([c.sum() for c in lc])


# ============================================================================ references
def mi_closed_form(r, m, d):
    """
    Exact bounds on I(decision; A) for binary decision and group: with both marginals fixed,
    the (decision, A) table is determined by t = P(decision=1, A=1), whose feasible range is the
    Frechet sum (the same quantity as the DP numerator). MI is convex in t with its minimum at
    independence, so min = MI(clamp(P(d)P(A))), max = max of MI at the two ends.
    """
    P, ma = r.sum(1), m[:, 1]
    h = r[:, d].sum(1)
    lo, hi = np.maximum(0, h + ma - P).sum(), np.minimum(h, ma).sum()
    pd_, pa_ = h.sum(), ma.sum()

    def mi(t):
        tab = np.array([[1 - pd_ - pa_ + t, pa_ - t], [pd_ - t, t]])
        mar = np.outer([1 - pd_, pd_], [1 - pa_, pa_])
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(np.nansum(np.where(tab > 0, tab * np.log(tab / mar), 0)))
    return mi(np.clip(pd_ * pa_, lo, hi)), max(mi(lo), mi(hi))


def cmi_values(theta, d, y):
    """I(decision; A | label) per joint, theta [N, I, G]."""
    out = 0
    for yv in (False, True):
        sel = (y == yv)
        tab = np.stack([theta[:, sel & ~d, :].sum(1), theta[:, sel & d, :].sum(1)], 1)  # [N,2,G]
        py = tab.sum((1, 2), keepdims=True)
        pdy = tab.sum(2, keepdims=True); pgy = tab.sum(1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = tab * np.log(tab * py / (pdy * pgy))
        out = out + np.nansum(t, (1, 2))
    return out


def cmi_optimised(r, m, d, y, starts=12, seed=0):
    """Best-known min / max of I(decision; A | label): SLSQP over q = P(c, i, A=1) from
    random feasible starts (no certificate: CMI is neither convex nor concave in the joint)."""
    Z, I = r.shape
    A_eq = np.zeros((Z, Z * I))
    for k in range(Z):
        A_eq[k, k * I:(k + 1) * I] = 1
    b_eq = m[:, 1]
    bounds = [(0, r[k, i]) for k in range(Z) for i in range(I)]

    def f(qflat, s):
        q = qflat.reshape(Z, I)
        th = np.stack([r - q, q], -1).sum(0)[None].clip(1e-15)
        return s * cmi_values(th, d, y)[0]
    starts_q = random_feasible_joints(r, m, starts, seed=seed)[:, :, :, 1].reshape(starts, -1)
    best = {}
    for s, name in [(1, "min"), (-1, "max")]:
        vals = []
        for q0 in starts_q:
            res = minimize(f, q0, args=(s,), method="SLSQP", bounds=bounds,
                           constraints=[{"type": "eq", "fun": lambda q: A_eq @ q - b_eq}],
                           options=dict(maxiter=300, ftol=1e-12))
            q = np.clip(res.x, [b[0] for b in bounds], [b[1] for b in bounds])
            if np.abs(A_eq @ q - b_eq).max() < 1e-7:
                vals.append(f(q, 1))
        best[name] = (min if s == 1 else max)(vals) if vals else np.nan
    return best["min"], best["max"]


def gap_from_diff(lo, hi):
    return (0.0 if lo <= 0 <= hi else min(abs(lo), abs(hi)), max(abs(lo), abs(hi)))


# ============================================================================ study 1
def study_exactness(a):
    rows = []
    for s in range(a.reps):
        pp, pa = problem(2, 4, 2, seed=100 + s, strength=a.strength)
        for num in a.nums:
            g = run_grid(pp, pa, num, "sequential-all", 2e6)
            if g is None:
                continue
            J, I, C, E = g
            r, m = kallus_inputs(pp, pa, I, C, E)
            d, y = dy_masks(I)
            th = J.sum(1)
            v = metric_values(th, d, y)
            v["DP_gap"], v["TPR_gap"] = abs(v["DP_diff"]), abs(v["TPR_diff"])
            v["CMI_sep"] = cmi_values(th, d, y)
            kb = kallus_bounds_all(r, m, I)
            ref = {k: (kb[k], "closed form (Kallus)") for k in
                   ["DP_diff", "TPR_diff", "FPR_diff", "PPV_diff", "NPV_diff"]}
            ref.update({k: (kb[k], "closed-form corollary") for k in ["F1_diff", "FDR_diff", "DI_ratio"]})
            ref["DP_gap"] = (gap_from_diff(*kb["DP_diff"]), "from diff interval")
            ref["TPR_gap"] = (gap_from_diff(*kb["TPR_diff"]), "from diff interval")
            ref["MI"] = (mi_closed_form(r, m, d), "closed form (1-d in P(d=1,A=1))")
            if num == a.nums[0]:
                eo = max_gap_lp_lower(r, m, [(d & y, y), (d & ~y, ~y)], a.lp_grid)
                su = max_gap_lp_lower(r, m, [(d & y, d), (~d & ~y, ~d)], a.lp_grid)
                cmi = cmi_optimised(r, m, d, y, seed=s)
                cache = (eo, su, cmi)
            eo, su, cmi = cache
            ref["EOdds"] = ((eo, kb["EOdds"][1]), "LP (lower) / max of gaps (upper)")
            ref["Sufficiency"] = ((su, kb["Sufficiency"][1]), "LP (lower) / max of gaps (upper)")
            ref["CMI_sep"] = (cmi, "best of 12 SLSQP starts (no certificate)")
            for k, ((rl, ru), how) in ref.items():
                gl, gu = np.nanmin(v[k]), np.nanmax(v[k])
                rows.append(dict(problem=s, num=num, metric=k, reference=how,
                                 lower_err=gl - rl, upper_err=ru - gu))
        print(f"exactness: problem {s} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "exactness.csv"), index=False)
    # positive err = grid interval inside the reference (too narrow); for lower bounds that
    # sit between corners this means the grid OVERSTATES the lower bound
    summ = df.groupby(["metric", "reference", "num"]).agg(
        worst_lower=("lower_err", lambda x: x.abs().max()),
        worst_upper=("upper_err", lambda x: x.abs().max())).reset_index()
    wide = summ.pivot_table(index=["metric", "reference"], columns="num",
                            values=["worst_lower", "worst_upper"])
    print("\nEXACTNESS (sequential-all): worst |grid - reference| over problems")
    print(wide.to_string(float_format=lambda x: f"{x:.1e}"))
    return df


# ============================================================================ study 2
def study_feasibility(a):
    rows = []
    sample = 12                      # common values counted; larger Z is extrapolated
    for n_int in (2, 4, 8):
        for G in (2, 3):
            pp, pa = problem(max(a.Zs), n_int, G, seed=7)
            for num in (2, 3, 5):
                for method in ("sweep", "sequential-all"):
                    lc = cell_log_counts(pp, pa, num, method, max_cells=sample)
                    for Z in a.Zs:
                        if lc is None:
                            lg = np.nan
                        else:            # exact for Z <= sample, else mean per value x Z
                            lg = _combine([c[:Z].sum() if Z <= len(c) else c.mean() * Z
                                           for c in lc])
                        rows.append(dict(internal_values=n_int, groups=G, num=num, Z=Z,
                                         free_params=Z * (n_int - 1) * (G - 1), method=method,
                                         log10_joints=lg, estimated=Z > sample))
        print(f"feasibility: internal={n_int} done", flush=True)
    df = pd.DataFrame(rows)
    # calibrate memory: grid_joint stores float64 [N, Z, I, G] and copies while combining (~3x)
    df["est_GB"] = 10 ** df.log10_joints * df.Z * df.internal_values * df.groups * 8 * 3 / 1e9
    df.to_csv(os.path.join(a.out, "feasibility.csv"), index=False)
    lim = np.log10(a.max_joints)
    front = (df[df.log10_joints <= lim].groupby(["method", "internal_values", "groups", "num"])
             .Z.max().unstack("num"))
    print(f"\nFEASIBILITY: largest number of common values Z with <= {a.max_joints:,.0f} joints")
    print(front.to_string())

    # real proxies (counts only): HMDA county / income, warfarin medication proxies
    real = []
    if os.path.isdir(a.kallus_repo):
        try:
            h = pd.read_csv(os.path.join(a.kallus_repo, "HMDA", "small_proxy_county.csv"),
                            usecols=["county_id", "outcome", "race"])
            for prox in ["county_id"]:
                ppr = h.groupby(["outcome", prox]).size().to_dict()
                par = h.assign(g=np.where(h.race == "White", "1_W", "0_r")).groupby([prox, "g"]).size().to_dict()
                real.append(dict(data="HMDA", proxy=f"{prox} ({h[prox].nunique()} levels)",
                                 internal="decision", num=2,
                                 log10_joints=log10_joints(ppr, par, 2, "sequential-all")))
            inc = pd.read_csv(os.path.join(a.kallus_repo, "HMDA", "small_proxy_income.csv"),
                              usecols=["applicant_income_000s", "outcome", "race"])
            for b in (5, 10, 20, 50, 930):
                if b < 930:
                    e = np.unique(np.quantile(inc.applicant_income_000s, np.linspace(0, 1, b + 1)))
                    zc = pd.cut(inc.applicant_income_000s, e, include_lowest=True, labels=False)
                else:
                    zc = inc.applicant_income_000s
                df_ = inc.assign(z=zc, g=np.where(inc.race == "White", "1_W", "0_r"))
                real.append(dict(data="HMDA", proxy=f"income ({df_.z.nunique()} levels)",
                                 internal="decision", num=2,
                                 log10_joints=log10_joints(df_.groupby(["outcome", "z"]).size().to_dict(),
                                                           df_.groupby(["z", "g"]).size().to_dict(), 2,
                                                           "sequential-all")))
            w = pd.read_csv(os.path.join(a.kallus_repo, "Warfrin", "medicine_as_proxy.csv"))
            meds = [c for c in w.columns if c.endswith(".") and c != "race."]
            w = w.assign(i=list(zip(w.Yhat.astype(int), w.Y.astype(int))),
                         g=np.where(w.race == "White.", "1_W", "0_r"))
            for k in (1, 2, 3, 4, len(meds)):
                z = w[meds[:k]].fillna("NA").astype(str).agg("|".join, axis=1)
                dfw = w.assign(z=z)
                real.append(dict(data="warfarin", proxy=f"{k} medication column(s) ({z.nunique()} levels)",
                                 internal="(decision, label)", num=2,
                                 log10_joints=log10_joints(dfw.groupby(["i", "z"]).size().to_dict(),
                                                           dfw.groupby(["z", "g"]).size().to_dict(), 2,
                                                           "sequential-all")))
        except Exception as ex:                       # real data are optional
            print("real-data feasibility skipped:", ex)
    if real:
        rd = pd.DataFrame(real)
        rd["feasible"] = rd.log10_joints <= lim
        rd.to_csv(os.path.join(a.out, "feasibility_real.csv"), index=False)
        print("\nREAL PROXIES (sequential-all, num=2, exact for binary-group metrics)")
        print(rd.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    if plt is not None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, G in zip(axes, (2, 3)):
            sub = df[(df.groups == G) & (df.method == "sequential-all") & (df.num == 2)]
            for n_int, s in sub.groupby("internal_values"):
                ax.plot(s.Z, s.log10_joints, "o-", label=f"{n_int} internal values")
            ax.axhline(lim, color="k", ls="--", lw=1, label=f"{a.max_joints:,.0f} joints")
            ax.set_xlabel("common values Z"); ax.set_title(f"{G} groups, sequential-all, num=2")
            ax.legend(fontsize=8)
        axes[0].set_ylabel("log10 number of joints")
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "feasibility.png"), dpi=150)
    return df


# ============================================================================ study 3
def study_groups(a):
    """Three groups: sequential-all permutes only the internal (row) order; with three
    columns the last column is also derived. Compare with permuting rows and columns."""
    rows = []
    for s in range(a.reps):
        for Z in (1, 2):
            pp, pa = problem(Z, 4, 3, seed=300 + s, strength=a.strength)
            g = run_grid(pp, pa, 2, "sequential-all", 2e6)
            J, I, C, E = g
            # also permute group (column) order
            Js = [J]
            for perm in itertools.permutations(range(3)):
                lab = {e: f"{perm.index(k)}|{e}" for k, e in enumerate(E)}
                g2 = run_grid(pp, {(c, lab[e]): v for (c, e), v in pa.items()}, 2,
                              "sequential-all", 2e6)
                J2, _, _, E2 = g2
                Js.append(J2[:, :, :, [E2.index(lab[e]) for e in E]])
            Jall = np.concatenate(Js)
            r, m = kallus_inputs(pp, pa, I, C, E)
            d, y = dy_masks(I)
            for name, (tg, cd) in [("DP", (d, np.ones_like(d))), ("TPR", (d & y, y))]:
                lpb = lp_bounds(r, m, tg, cd, harmonize=False, n_grid=41)
                for label, JJ in [("rows only", J), ("rows x columns", Jall)]:
                    th = JJ.sum(1)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        rate = th[:, tg, :].sum(1) / th[:, cd, :].sum(1)
                    for gi in (1, 2):
                        dv = rate[:, gi] - rate[:, 0]
                        ref = lpb[lpb.group == gi].iloc[0]
                        rows.append(dict(problem=s, Z=Z, metric=name, permutations=label,
                                         contrast=f"g{gi}-g0", n_joints=len(JJ),
                                         lower_err=np.nanmin(dv) - ref.lower,
                                         upper_err=ref.upper - np.nanmax(dv)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "three_groups.csv"), index=False)
    summ = df.groupby(["metric", "Z", "permutations"]).agg(
        joints=("n_joints", "max"), worst_lower=("lower_err", "max"),
        worst_upper=("upper_err", "max")).reset_index()
    print("\nTHREE GROUPS (num=2) vs Kallus LP; err > 0 = grid interval inside the LP interval")
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.1e}"))
    return df


# ============================================================================ study 4
def _lp_rate_range(r, m, rate, extra=None, n_grid=41):
    """
    Min and max of a binary rate disparity (target, cond) over couplings that also satisfy
    linear constraints lo <= w @ q <= hi, given as (w_fn, lo, hi) with w_fn(lp) -> w.
    Group 1's condition mass x1 is searched over the range the constraints allow (found by LP),
    on a grid followed by bounded refinement; for fixed x1 the problem is an LP.
    """
    from scipy.optimize import minimize_scalar
    lp = CouplingLP(r, m, harmonize=False)
    T, X = lp.masks(*rate)
    W = [(w_fn(lp), lo, hi) for w_fn, lo, hi in (extra or [])]
    A_ub = np.array([w for w, _, _ in W] + [-w for w, _, _ in W]) if W else None
    b_ub = np.array([hi for _, _, hi in W] + [-lo for _, lo, _ in W]) if W else None

    def solve(c, x=None):
        A_eq, b_eq = lp.A_eq, lp.b_eq
        if x is not None:
            A_eq = sparse.vstack([A_eq, sparse.csr_matrix(X)]); b_eq = np.concatenate([b_eq, x])
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, None),
                      method="highs")
        return res.fun if res.status == 0 else np.inf

    K = -solve(-X.sum(0))
    lo_x, hi_x = solve(X[1]), -solve(-X[1])          # range of x1 under the constraints
    if not (np.isfinite(lo_x) and np.isfinite(hi_x)):
        return np.nan, np.nan
    out = []
    for s in (1, -1):                                 # s = 1: minimum, s = -1: maximum
        def val(x1):
            x0 = K - x1
            if x1 <= 1e-12 or x0 <= 1e-12:
                return np.inf
            return solve(s * (T[1] / x1 - T[0] / x0), np.array([x0, x1]))
        xs = np.linspace(lo_x, hi_x, n_grid) if hi_x > lo_x else np.array([lo_x])
        vs = np.array([val(x) for x in xs])
        i = int(np.argmin(vs)); best = vs[i]
        if len(xs) > 2:
            res = minimize_scalar(val, bounds=(xs[max(i - 1, 0)], xs[min(i + 1, len(xs) - 1)]),
                                  method="bounded", options=dict(xatol=1e-12))
            best = min(best, res.fun)
        out.append(s * best if np.isfinite(best) else np.nan)
    return out[0], out[1]


def study_extras(a):
    out = {}
    pp, pa = problem(2, 4, 2, seed=a.extras_seed, strength=a.strength)
    J, I, C, E = run_grid(pp, pa, a.extras_num, "sequential-all", 5e6)
    r, m = kallus_inputs(pp, pa, I, C, E)
    d, y = dy_masks(I)
    th = J.sum(1)
    v = metric_values(th, d, y)
    kb = kallus_bounds_all(r, m, I)
    Q = random_feasible_joints(r, m, 4000, seed=1, iters=1500)
    qv = metric_values(Q.sum(1), d, y)
    print(f"\nEXTRAS: one problem, {len(J):,} grid joints (sequential-all, num={a.extras_num})")

    # (a) joint regions of metric pairs: share of the Kallus rectangle no joint reaches
    pairs = [("TPR_diff", "FPR_diff"), ("DP_diff", "TPR_diff"), ("PPV_diff", "TPR_diff")]
    reg = []
    for xk, yk in pairs:
        pts = np.vstack([np.c_[v[xk], v[yk]], np.c_[qv[xk], qv[yk]]])
        (x0, x1), (y0, y1) = kb[xk], kb[yk]
        H, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=40, range=[[x0, x1], [y0, y1]])
        reg.append(dict(pair=f"{xk} x {yk}", rectangle_cells_reached=(H > 0).mean()))
    reg = pd.DataFrame(reg)
    print("\n(a) share of the Kallus rectangle (40x40 cells) reached by any grid/random joint")
    print(reg.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    out["regions"] = reg

    # (b) conditional bounds: range of TPRD among joints with |DP| <= eps, grid vs exact LP
    dp_w = lambda lp: (lambda T, X: T[1] / m[:, 1].sum() - T[0] / m[:, 0].sum())(*lp.masks(d, np.ones_like(d)))
    rows = []
    for eps in (np.inf, 0.2, 0.1, 0.05, 0.02, 0.01, 0.001):
        sel = np.abs(v["DP_diff"]) <= eps
        g = (np.nanmin(v["TPR_diff"][sel]), np.nanmax(v["TPR_diff"][sel])) if sel.any() else (np.nan, np.nan)
        lpb = _lp_rate_range(r, m, (d & y, y), [(dp_w, -eps, eps)] if np.isfinite(eps) else None,
                             n_grid=a.lp_grid)
        rows.append(dict(constraint=f"|DP| <= {eps}", grid_joints_kept=int(sel.sum()),
                         grid_TPRD=f"[{g[0]:.3f}, {g[1]:.3f}]", exact_TPRD=f"[{lpb[0]:.3f}, {lpb[1]:.3f}]"))
    cond = pd.DataFrame(rows)
    print("\n(b) conditional bounds: TPR disparity given demographic parity (nearly) holds")
    print(cond.to_string(index=False))
    out["conditional"] = cond

    # (c) external information: group base rates P(Y=1 | A=g) known to within +-eps
    true_base = None
    pr = (th[:, y, :].sum(1) / th.sum(1))                         # [N, G] base rate per joint
    # a 'true' joint to take the base rates from: the midpoint random joint
    qb = (Q.sum(1)[:, y, :].sum(1) / Q.sum(1).sum(1))
    true_base = qb[0]
    yw = lambda g_: (lambda lp: (lambda T, X: T[g_])(*lp.masks(y, np.ones_like(y))))
    rows = []
    for eps in (np.inf, 0.1, 0.05, 0.02, 0.01, 0.001):
        sel = np.all(np.abs(pr - true_base) <= eps, axis=1)
        g = (np.nanmin(v["TPR_diff"][sel]), np.nanmax(v["TPR_diff"][sel])) if sel.any() else (np.nan, np.nan)
        cons = None
        if np.isfinite(eps):
            cons = [(yw(gi), (true_base[gi] - eps) * m[:, gi].sum(), (true_base[gi] + eps) * m[:, gi].sum())
                    for gi in (0, 1)]
        lpb = _lp_rate_range(r, m, (d & y, y), cons, n_grid=a.lp_grid)
        rows.append(dict(known_base_rates=f"+-{eps}", grid_joints_kept=int(sel.sum()),
                         grid_TPRD=f"[{g[0]:.3f}, {g[1]:.3f}]", exact_TPRD=f"[{lpb[0]:.3f}, {lpb[1]:.3f}]"))
    ext = pd.DataFrame(rows)
    print(f"\n(c) external information: P(Y=1 | group) known to within eps (values {np.round(true_base, 3)})")
    print(ext.to_string(index=False))
    out["external"] = ext

    # (d) grid frequencies are not probabilities: distribution of DP over the same identified set
    J_sw = run_grid(pp, pa, a.extras_num, "sweep", 5e6)[0]
    dists = {"sequential-all": v["DP_diff"],
             "sweep": metric_values(J_sw.sum(1), d, y)["DP_diff"],
             "random (IPF)": qv["DP_diff"]}
    qs = pd.DataFrame({k: np.nanquantile(x, [0.1, 0.5, 0.9]) for k, x in dists.items()},
                      index=["10%", "median", "90%"]).T
    print("\n(d) quantiles of DP disparity across 'joints' of the same identified set")
    print(qs.to_string(float_format=lambda x: f"{x:.3f}"))
    out["density"] = qs

    if plt is not None:
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
        for ax, (xk, yk) in zip(axes[:3], pairs):
            ax.scatter(qv[xk], qv[yk], s=1, c="0.75", label="random feasible joints")
            ax.scatter(v[xk], v[yk], s=1, c="C0", label="grid joints")
            (x0, x1), (y0, y1) = kb[xk], kb[yk]
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="C3", lw=1.5,
                                       label="Kallus intervals"))
            ax.set_xlabel(xk); ax.set_ylabel(yk); ax.legend(fontsize=7, loc="upper left")
        for k, x in dists.items():
            axes[3].hist(x[np.isfinite(x)], bins=60, density=True, histtype="step", label=k)
        axes[3].set_xlabel("DP disparity"); axes[3].set_title("same set, different 'distributions'")
        axes[3].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "extras.png"), dpi=150)
    for k, df in out.items():
        df.to_csv(os.path.join(a.out, f"extras_{k}.csv"))
    return out


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("study", choices=["exactness", "feasibility", "groups", "extras", "all"])
    ap.add_argument("--out", default="results/grid_limits")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--kallus-repo", default=os.path.expanduser("~/kallus"))
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--strength", type=float, default=2.0, help="proxy strength of synthetic problems")
    ap.add_argument("--max-joints", type=float, default=5e6)
    ap.add_argument("--lp-grid", type=int, default=41)
    ap.add_argument("--extras-seed", type=int, default=5)
    ap.add_argument("--extras-num", type=int, default=9)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    q = a.quick
    if a.study in ("exactness", "all"):
        a.reps = a.reps or (2 if q else 8)
        a.nums = [2, 5] if q else [2, 3, 5, 9]
        study_exactness(a)
    if a.study in ("feasibility", "all"):
        a.Zs = [1, 2, 4, 8] if q else [1, 2, 3, 4, 6, 8, 10, 15, 20, 30, 50]
        study_feasibility(a)
    if a.study in ("groups", "all"):
        a.reps = a.reps or (2 if q else 5)
        study_groups(a)
    if a.study in ("extras", "all"):
        a.extras_num = 5 if q else a.extras_num
        study_extras(a)


if __name__ == "__main__":
    main()