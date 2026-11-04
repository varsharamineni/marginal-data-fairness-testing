"""
Synthetic experiments for the feasible-set approach. Populations are specified exactly
(probabilities, no sampling), so every true value is known and the reference bounds are exact.

Population: a common variable Z with K equally likely values; a binary protected attribute A
with P(A = 1 | Z = z) = 0.5 + 0.45 s l_z (l_z evenly spaced in [-1, 1]); s in [0, 1] is the
PROXY STRENGTH (s = 0: Z says nothing about A). Decisions follow logistic models in which
gamma is the DIRECT EFFECT of A beyond Z (gamma = 0: conditional independence holds).

Experiments (--exp)
  compare      Classifier comparison. Rule A: P(D = 1 | z, a) = sigmoid(-0.2 + 0.8 l_z + gamma a).
               Rule B: A, plus approving a fraction --frac of A's denials in the upper half of
               the bins (where P(A = 1 | z) is highest once s > 0). Bounds on DD of A, of B, and of DD(A) - DD(B),
               against proxy strength.
  joint        Joint estimation. Internal variable (D, Y): two metrics (DD and the TPR
               difference) bounded jointly. The feasible (DD, TPRD) pairs fill only part of the
               box formed by the two separate intervals, and knowing one metric (here DD <= 0)
               tightens the bounds on the other.
  sensitivity  Conditional independence. Against the direct effect gamma: the true DD, the CI
               estimate, the full bounds, the true dependence I(D; A | Z), and the breakdown
               point (least dependence at which the bounds reach parity). Both the grid and the
               exact values are reported: bounds under I <= delta and the breakdown point are
               convex problems (linear objective or constraint, convex I), solved with SLSQP.
  cost         Limits and computational cost: number of joints and time against the grid size
               num, the number of common values K, internal values and groups; and the accuracy
               of an interior bound (min |DD|) against num.

Usage
  python synthetic.py                      # all experiments
  python synthetic.py --exp compare cost
Outputs: <out>/synthetic_<exp>.csv and <out>/figures/synthetic_<exp>.png/.pdf
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from fs_common import (C_A, C_B, C_CI, C_DIFF, C_EXACT, C_GRID, GRIDC, INK, REF, STYLE, TRUTH,
                       ci_joint, cond_mi, enumerate_joints, lp_bounds)

sig = lambda x: 1 / (1 + np.exp(-x))


def p_a(K, s):
    return np.clip(0.5 + 0.45 * s * np.linspace(-1, 1, K), 0.01, 0.99)


def dd_from_theta(theta, pos):
    """DD (A=1 minus A=0) of the decision 'internal value in pos' from theta[N, i, a]."""
    pa = theta.sum(1)
    rate = theta[:, pos, :].sum(1) / pa
    return rate[:, 1] - rate[:, 0]


def dd_weights(r, m, pos):
    p = m.sum(0)
    w = np.zeros((r.shape[0], r.shape[1], 2))
    w[:, pos, 1] = 1 / p[1]
    w[:, pos, 0] = -1 / p[0]
    return w


def interval(ax, x, lo, hi, col, lw=1.3, cap=3):
    mid = (lo + hi) / 2
    ax.errorbar(x, mid, yerr=[[mid - lo], [hi - mid]], fmt="none", ecolor=col, elinewidth=lw,
                capsize=cap, capthick=lw, zorder=3)


def star(ax, x, y):
    ax.plot(x, y, "*", ms=8, color=TRUTH, mec=INK, mew=0.5, zorder=6)


def setup(ax):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRIDC, lw=0.6)


def save(fig, out, name):
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, "figures", f"{name}.{ext}"), dpi=300, bbox_inches="tight",
                    pad_inches=0.02)


# ============================================================ 1. classifier comparison
def pop_compare(K, s, gamma, frac):
    pz, pa1 = np.full(K, 1 / K), p_a(K, s)
    l = np.linspace(-1, 1, K)
    high = l > 0
    q = np.zeros((K, 4, 2))                 # internal (D_A, D_B): index 2 d_A + d_B
    for z in range(K):
        for a in (0, 1):
            w = pz[z] * (pa1[z] if a else 1 - pa1[z])
            d1 = sig(-0.2 + 0.8 * l[z] + gamma * a)
            f = frac if high[z] else 0.0
            q[z, 3, a] = w * d1                     # A approves (so does B)
            q[z, 1, a] = w * (1 - d1) * f           # A denies, B approves
            q[z, 0, a] = w * (1 - d1) * (1 - f)     # both deny
    return q


def exp_compare(a, plt):
    rows = []
    for s in a.strengths:
        q = pop_compare(a.K, s, a.gamma, a.frac)
        r, m = q.sum(2), q.sum(1)
        J, num, sec = enumerate_joints(r, m, a.nums, a.max_joints)
        th = J.sum(1)
        A_, B_ = dd_from_theta(th, [2, 3]), dd_from_theta(th, [1, 3])
        tA = dd_from_theta(q.sum(0)[None], [2, 3])[0]
        tB = dd_from_theta(q.sum(0)[None], [1, 3])[0]
        wA, wB = dd_weights(r, m, [2, 3]), dd_weights(r, m, [1, 3])
        eA, eB, eD = lp_bounds(r, m, wA), lp_bounds(r, m, wB), lp_bounds(r, m, wA - wB)
        D = A_ - B_
        rows.append(dict(strength=s, num=num, n_joints=len(J), seconds=sec,
                         A_lower=A_.min(), A_upper=A_.max(), truth_A=tA,
                         B_lower=B_.min(), B_upper=B_.max(), truth_B=tB,
                         diff_lower=D.min(), diff_upper=D.max(), truth_diff=tA - tB,
                         exact_A_lower=eA[0], exact_A_upper=eA[1], exact_B_lower=eB[0],
                         exact_B_upper=eB[1], exact_diff_lower=eD[0], exact_diff_upper=eD[1],
                         separate_overlap=bool(A_.max() >= B_.min() and B_.max() >= A_.min()),
                         diff_excludes_0=bool(D.max() < 0 or D.min() > 0)))
    res = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.3), layout="constrained",
                             gridspec_kw=dict(width_ratios=[1.3, 1]))
    x = np.arange(len(res))
    ax = axes[0]; setup(ax); ax.axhline(0, color=REF, lw=0.8)
    for _, row in res.iterrows():
        k = row.name
        interval(ax, k - 0.12, row.A_lower, row.A_upper, C_A)
        interval(ax, k + 0.12, row.B_lower, row.B_upper, C_B)
        star(ax, k - 0.12, row.truth_A); star(ax, k + 0.12, row.truth_B)
    ax.set_title("(a) Separate bounds on DD", loc="left", pad=4)
    ax = axes[1]; setup(ax); ax.axhline(0, color=INK, lw=0.8, ls=":")
    for _, row in res.iterrows():
        interval(ax, row.name, row.diff_lower, row.diff_upper, C_DIFF, lw=1.8, cap=4)
        star(ax, row.name, row.truth_diff)
    ax.set_title("(b) Bounds on the difference", loc="left", pad=4)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels([f"{s:g}" for s in res.strength])
        ax.set_xlabel("Proxy strength $s$")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=C_A, lw=1.3, label="Rule A"),
         Line2D([], [], color=C_B, lw=1.3, label="Rule B"),
         Line2D([], [], color=C_DIFF, lw=1.8, label="A $-$ B"),
         Line2D([], [], ls="", marker="*", ms=8, color=TRUTH, mec=INK, mew=0.5,
                label="True value")]
    fig.legend(handles=h, loc="outside upper center", ncol=4, frameon=False)
    return res, fig


# ============================================================ 2. joint estimation
def pop_joint(K, s, gamma):
    pz, pa1 = np.full(K, 1 / K), p_a(K, s)
    l = np.linspace(-1, 1, K)
    q = np.zeros((K, 4, 2))                 # internal (D, Y): index 2 d + y
    for z in range(K):
        for a in (0, 1):
            w = pz[z] * (pa1[z] if a else 1 - pa1[z])
            py = sig(0.2 + 0.6 * l[z] - 0.3 * a)
            for y, wy in ((1, py), (0, 1 - py)):
                pd1 = sig(-0.3 + 1.2 * y + 0.5 * l[z] + gamma * a)
                q[z, 2 + y, a] = w * wy * pd1
                q[z, y, a] = w * wy * (1 - pd1)
    return q


def tpr_diff(theta):
    pos_y = theta[:, [1, 3], :].sum(1)
    tpr = theta[:, 3, :] / pos_y
    return tpr[:, 1] - tpr[:, 0]


def exp_joint(a, plt):
    from scipy.spatial import ConvexHull
    rows, keep = [], None
    for s in a.strengths:
        q = pop_joint(a.K_joint, s, a.gamma)
        r, m = q.sum(2), q.sum(1)
        J, num, sec = enumerate_joints(r, m, a.nums, a.max_joints)
        th = J.sum(1)
        dd, tp = dd_from_theta(th, [2, 3]), tpr_diff(th)
        ok = np.isfinite(dd) & np.isfinite(tp)
        dd, tp = dd[ok], tp[ok]
        box = (dd.max() - dd.min()) * (tp.max() - tp.min())
        hull = ConvexHull(np.column_stack([dd, tp])).volume if box > 0 else 0.0
        # how much knowing the sign of DD tightens the TPR difference
        neg = dd <= 0
        rows.append(dict(strength=s, num=num, n_joints=int(ok.sum()), seconds=sec,
                         dd_lower=dd.min(), dd_upper=dd.max(), tpr_lower=tp.min(),
                         tpr_upper=tp.max(), truth_dd=dd_from_theta(q.sum(0)[None], [2, 3])[0],
                         truth_tpr=tpr_diff(q.sum(0)[None])[0], hull_share_of_box=hull / box,
                         tpr_lower_given_dd_le0=tp[neg].min() if neg.any() else np.nan,
                         tpr_upper_given_dd_le0=tp[neg].max() if neg.any() else np.nan,
                         corr=float(np.corrcoef(dd, tp)[0, 1])))
        if abs(s - a.show_strength) < 1e-9:
            keep = (dd, tp, rows[-1])
    res = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.4), layout="constrained",
                             gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax = axes[0]
    if keep is not None:
        dd, tp, row = keep
        idx = np.random.default_rng(0).choice(len(dd), min(len(dd), 40_000), replace=False)
        ax.add_patch(plt.Rectangle((dd.min(), tp.min()), dd.max() - dd.min(),
                                   tp.max() - tp.min(), fill=False, ec=INK, lw=0.8, ls="--"))
        ax.scatter(dd[idx], tp[idx], s=1.2, color=C_GRID, alpha=0.25, lw=0, rasterized=True)
        star(ax, row["truth_dd"], row["truth_tpr"])
        ax.set_xlabel("Demographic disparity")
        ax.set_ylabel("TPR difference")
        ax.set_title(f"(a) Feasible pairs ($s$ = {a.show_strength:g})", loc="left", pad=4)
    ax = axes[1]; setup(ax); ax.axhline(0, color=REF, lw=0.8)
    for k, row in res.iterrows():
        interval(ax, k - 0.12, row.tpr_lower, row.tpr_upper, C_GRID)
        if np.isfinite(row.tpr_lower_given_dd_le0):
            interval(ax, k + 0.12, row.tpr_lower_given_dd_le0, row.tpr_upper_given_dd_le0,
                     C_EXACT)
        star(ax, k - 0.12, row.truth_tpr)
    ax.set_xticks(range(len(res)))
    ax.set_xticklabels([f"{s:g}" for s in res.strength])
    ax.set_xlabel("Proxy strength $s$")
    ax.set_ylabel("TPR difference")
    ax.set_title("(b) TPR difference, alone and given DD $\\leq$ 0", loc="left", pad=4)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=C_GRID, lw=1.3, label="Separate bounds"),
         Line2D([], [], color=C_EXACT, lw=1.3, label="Given DD $\\leq$ 0"),
         Line2D([], [], ls="", marker="*", ms=8, color=TRUTH, mec=INK, mew=0.5,
                label="True value")]
    fig.legend(handles=h, loc="outside upper center", ncol=3, frameon=False)
    return res, fig


# ============================================================ 3. sensitivity to CI
def pop_d(K, s, gamma):
    pz, pa1 = np.full(K, 1 / K), p_a(K, s)
    l = np.linspace(-1, 1, K)
    q = np.zeros((K, 2, 2))
    for z in range(K):
        for a in (0, 1):
            w = pz[z] * (pa1[z] if a else 1 - pa1[z])
            d1 = sig(-0.2 + 0.8 * l[z] + gamma * a)
            q[z, 1, a], q[z, 0, a] = w * d1, w * (1 - d1)
    return q


def _cmi_single(r, m):
    """Smooth I(D; A | Z) of one flattened joint, for the convex solver."""
    from fs_common import ci_joint
    ex = ci_joint(r, m).ravel()
    eps = 1e-12
    return lambda x: float(np.sum(np.where(ex > 0, x * np.log2((x + eps) / (ex + eps)), 0.0)))


def _solve(r, m, obj, ineq, x0s):
    """min obj(x) s.t. margins, x >= 0, ineq(x) >= 0 (all convex): best SLSQP solution."""
    from scipy.optimize import minimize
    from fs_common import _constraints
    A, b, _ = _constraints(r, m)
    best = np.inf
    for x0 in x0s:
        res = minimize(obj, x0, method="SLSQP", bounds=[(0, 1)] * len(x0),
                       constraints=[{"type": "eq", "fun": lambda x: A @ x - b},
                                    {"type": "ineq", "fun": ineq}],
                       options=dict(maxiter=1000, ftol=1e-13))
        if np.abs(A @ res.x - b).max() < 1e-7 and ineq(res.x) > -1e-7:
            best = min(best, float(obj(res.x)))
    return best


def exact_sensitivity(r, m, w, delta, truth_q):
    """Exact bounds of the linear metric w . q under I <= delta (both convex)."""
    cmi = _cmi_single(r, m)
    from fs_common import ci_joint
    x0s = [ci_joint(r, m).ravel(), truth_q.ravel()]
    wv = w.ravel()
    lo = _solve(r, m, lambda x: wv @ x, lambda x: delta - cmi(x), x0s)
    hi = -_solve(r, m, lambda x: -(wv @ x), lambda x: delta - cmi(x), x0s)
    return lo, hi


def exact_breakdown(r, m, w, target, side, x0s):
    """Least I(D; A | Z) at which the metric w . q reaches target from the given side."""
    cmi = _cmi_single(r, m)
    wv = w.ravel()
    return _solve(r, m, cmi, lambda x: side * (target - wv @ x), x0s)


def exp_sensitivity(a, plt):
    rows = []
    for g in a.gammas:
        q = pop_d(a.K, a.strength, g)
        r, m = q.sum(2), q.sum(1)
        J, num, sec = enumerate_joints(r, m, a.nums_fine, a.max_joints)
        J = np.concatenate([J, ci_joint(r, m)[None]])
        dep = cond_mi(J, r, m)
        dd = dd_from_theta(J.sum(1), [1])
        ci = dd[-1]
        truth = dd_from_theta(q.sum(0)[None], [1])[0]
        true_dep = float(cond_mi(q[None], r, m)[0])
        side = np.sign(ci) if abs(ci) > 1e-12 else 1.0
        reach = np.sign(dd) != side
        brk = float(dep[reach].min()) if reach.any() else np.nan
        within = dep <= true_dep + 1e-12
        w = dd_weights(r, m, [1])
        ex = lp_bounds(r, m, w)
        ex_lo_t, ex_hi_t = exact_sensitivity(r, m, w, true_dep, q)
        # exact breakdown: least dependence with DD on the other side of 0 from the CI value
        x_far = J[np.argmin(dd) if side > 0 else np.argmax(dd)]
        ex_brk = exact_breakdown(r, m, w, 0.0, side, [x_far.ravel(), q.ravel()])
        rows.append(dict(gamma=g, num=num, n_joints=len(J), truth=truth, ci=ci,
                         ci_error=ci - truth, lower=dd.min(), upper=dd.max(), exact_lower=ex[0],
                         exact_upper=ex[1], true_dep=true_dep, breakdown=brk,
                         max_dep=float(dep.max()),
                         lower_at_true_dep=dd[within].min(), upper_at_true_dep=dd[within].max(),
                         exact_lower_at_true_dep=ex_lo_t, exact_upper_at_true_dep=ex_hi_t,
                         exact_breakdown=ex_brk,
                         truth_in_exact_bounds_at_true_dep=bool(ex_lo_t - 1e-6 <= truth
                                                                <= ex_hi_t + 1e-6)))
    res = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.3), layout="constrained")
    ax = axes[0]; setup(ax); ax.axhline(0, color=REF, lw=0.8)
    ax.fill_between(res.gamma, res.lower, res.upper, color=C_GRID, alpha=0.2, lw=0,
                    label="Bounds")
    ax.fill_between(res.gamma, res.exact_lower_at_true_dep, res.exact_upper_at_true_dep,
                    color=C_GRID, alpha=0.55, lw=0, label="Bounds at true dependence")
    ax.plot(res.gamma, res.truth, "*-", color=TRUTH, mec=INK, mew=0.4, ms=7, lw=1,
            label="True value")
    ax.plot(res.gamma, res.ci, "o--", color=C_CI, mfc="white", ms=3.5, lw=0.9,
            label="Cond. independence")
    ax.set_xlabel(r"Direct effect of $A$, $\gamma$")
    ax.set_ylabel("Demographic disparity")
    ax.set_title("(a) Estimate vs. truth", loc="left", pad=4)
    ax = axes[1]; setup(ax)
    ax.plot(res.gamma, res.true_dep, "*-", color=TRUTH, mec=INK, mew=0.4, ms=7, lw=1,
            label="True dependence")
    ax.plot(res.gamma, res.exact_breakdown, "s-", color=C_EXACT, ms=3.5, lw=1,
            label="Breakdown point")
    ax.set_yscale("log")
    ax.set_xlabel(r"Direct effect of $A$, $\gamma$")
    ax.set_ylabel("Bits")
    ax.set_title("(b) Dependence needed vs. present", loc="left", pad=4)
    h, l = [], []
    for ax_ in axes:
        hh, ll = ax_.get_legend_handles_labels()
        h += hh; l += ll
    fig.legend(h, l, loc="outside upper center", ncol=3, frameon=False)
    return res, fig


# ============================================================ 4. limits and cost
def pop_general(K, B, G, s=0.5, seed=0):
    """K common values, B internal values, G groups; a smooth random population."""
    rng = np.random.default_rng(seed)
    pz = np.full(K, 1 / K)
    l = np.linspace(-1, 1, K)
    ga = np.stack([np.exp(s * l * (g - (G - 1) / 2)) for g in range(G)], 1)
    pa = ga / ga.sum(1, keepdims=True)
    q = np.zeros((K, B, G))
    for z in range(K):
        for g in range(G):
            logits = rng.normal(0, 0.5, B) + 0.6 * l[z] * np.arange(B) / max(B - 1, 1) \
                - 0.4 * g * np.arange(B) / max(B - 1, 1)
            pi = np.exp(logits) / np.exp(logits).sum()
            q[z, :, g] = pz[z] * pa[z, g] * pi
    return q


def exp_cost(a, plt):
    rows = []
    configs = ([("K", K, 2, 2) for K in (2, 3, 4, 6, 8)]
               + [("internal", 2, B, 2) for B in (3, 4, 5)]
               + [("groups", 2, 2, G) for G in (3, 4)])
    for kind, K, B, G in configs:
        q = pop_general(K, B, G)
        r, m = q.sum(2), q.sum(1)
        F = K * (B - 1) * (G - 1)
        for num in a.cost_nums:
            row = dict(kind=kind, K=K, internal=B, groups=G, free=F, num=num,
                       grid_points=float(num) ** F)
            try:
                J, _, sec = enumerate_joints(r, m, [num], a.max_joints)
                row.update(n_joints=len(J), seconds=sec, feasible=True)
                if G == 2:                      # accuracy of an interior bound: min |DD|
                    dd = dd_from_theta(J.sum(1), list(range(1, B)))
                    lo, hi = lp_bounds(r, m, dd_weights(r, m, list(range(1, B))))
                    exact = 0.0 if lo <= 0 <= hi else min(abs(lo), abs(hi))
                    row.update(min_abs_dd=float(np.abs(dd).min()), exact_min_abs_dd=exact,
                               shortfall=float(np.abs(dd).min()) - exact)
            except SystemExit:
                row.update(n_joints=np.nan, seconds=np.nan, feasible=False)
            rows.append(row)
            if not row["feasible"]:
                break                           # larger num is infeasible too
    res = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), layout="constrained")
    cols = {2: "#0072B2", 3: "#009E73", 4: "#E69F00", 6: "#CC79A7", 8: "#D55E00"}
    ax = axes[0]; setup(ax)
    for K, gdf in res[res.kind == "K"].groupby("K"):
        f = gdf[gdf.feasible]
        ax.plot(f.num, f.n_joints, "o-", ms=3, lw=1, color=cols[K], label=f"K = {K}")
    ax.axhline(a.max_joints, color=INK, lw=0.8, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Grid points per parameter (num)")
    ax.set_ylabel("Joints enumerated")
    ax.set_title("(a) Size of the feasible set", loc="left", pad=4)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax = axes[1]; setup(ax)
    f = res[res.feasible]
    for kind, mk, col in (("K", "o", C_EXACT), ("internal", "s", C_DIFF), ("groups", "^", C_B)):
        g = f[f.kind == kind]
        ax.scatter(g.n_joints, g.seconds, s=10, marker=mk, color=col,
                   label={"K": "common values", "internal": "internal values",
                          "groups": "groups"}[kind])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Joints enumerated")
    ax.set_ylabel("Seconds")
    ax.set_title("(b) Time", loc="left", pad=4)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax = axes[2]; setup(ax)
    for K, gdf in res[(res.kind == "K") & res.feasible].groupby("K"):
        ax.plot(gdf.num, gdf.shortfall.clip(lower=1e-6), "o-", ms=3, lw=1, color=cols[K],
                label=f"K = {K}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Grid points per parameter (num)")
    ax.set_ylabel(r"Shortfall in min $|$DD$|$")
    ax.set_title("(c) Accuracy of an interior bound", loc="left", pad=4)
    return res, fig


EXPS = {"compare": exp_compare, "joint": exp_joint, "sensitivity": exp_sensitivity,
        "cost": exp_cost}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", nargs="+", default=list(EXPS), choices=list(EXPS))
    ap.add_argument("--K", type=int, default=3, help="common values (compare, sensitivity)")
    ap.add_argument("--K-joint", type=int, default=2, help="common values (joint)")
    ap.add_argument("--gamma", type=float, default=-0.6, help="direct effect (compare, joint)")
    ap.add_argument("--strength", type=float, default=0.6, help="proxy strength (sensitivity)")
    ap.add_argument("--strengths", type=float, nargs="+", default=[0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--show-strength", type=float, default=0.5)
    ap.add_argument("--gammas", type=float, nargs="+",
                    default=[0, -0.1, -0.25, -0.5, -0.75, -1.0, -1.5])
    ap.add_argument("--frac", type=float, default=0.8)
    ap.add_argument("--nums", type=int, nargs="+", default=[2, 3, 4, 5, 10, 20])
    ap.add_argument("--nums-fine", type=int, nargs="+", default=[20, 50, 100])
    ap.add_argument("--cost-nums", type=int, nargs="+", default=[2, 3, 5, 10, 20, 50, 100])
    ap.add_argument("--max-joints", type=int, default=2_000_000)
    ap.add_argument("--out", default="results/synthetic")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(STYLE)
    os.makedirs(a.out, exist_ok=True)
    for e in a.exp:
        t0 = time.perf_counter()
        res, fig = EXPS[e](a, plt)
        res.to_csv(os.path.join(a.out, f"synthetic_{e}.csv"), index=False)
        save(fig, a.out, f"synthetic_{e}")
        plt.close(fig)
        with pd.option_context("display.width", 250, "display.max_columns", 40):
            print(f"\n=== {e} ({time.perf_counter() - t0:.1f}s)")
            print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()