"""
Partial identification of fairness disparities with an unobserved protected attribute,
following Kallus, Mao & Zhou (2022), "Assessing algorithmic fairness with unobserved
protected class using data combination", Management Science 68(3); arXiv:1906.00285.
Replication code of the original: github.com/CausalML/FairnessWithUnobservedProtectedClass
(Python 2 + Gurobi + R). This module is a Python 3 / scipy (HiGHS) re-implementation for
discrete proxies, with optimisation variables per proxy level rather than per individual.

Three parts, mirroring the paper:

1. closed_form_bounds   Props 7-8 (binary A, no extra assumptions). Exact, no optimisation.
2. lp_bounds,           Sections 6 and 7.2. LP over couplings (Prop 10) for demographic
   support_function,    disparity; for rate disparities the group condition masses are fixed
   support_set          on a grid and an LP solved at each point (the t-grid of Prop 11 /
                        eq. 28, Theorem 4). Supports more than two groups (support function,
                        Algorithm 2) and the Lipschitz smoothness assumption (eqs. 16-17).
3. debiased_bounds      Section 7.1 / Appendix A.2 / A.5. Cross-fitted debiased estimators of
                        the binary bounds with Wald confidence intervals (Theorems 1-2).

Population-level inputs (parts 1 and 2) use the interface
    r : [Z, I] p(internal = i, z) from the primary data (internal = e.g. (decision, label))
    m : [Z, G] p(z, a = g) from the auxiliary data
    target, cond : boolean masks over the I internal values, target a subset of cond
The rate of group g is mu_g = P(target | cond, A = g); disparities are mu_g - mu_ref
(with two groups and ref = 0 this is "last group minus first").

    demographic disparity (DD)   target = {d=1}         cond = everything
    TPR disparity (TPRD)         target = {d=1, y=1}    cond = {y=1}
    FPR disparity (= -TNRD)      target = {d=1, y=0}    cond = {y=0}
    PPV disparity (PPVD)         target = {d=1, y=1}    cond = {d=1}
    NPV disparity (NPVD)         target = {d=0, y=0}    cond = {d=0}

As in the paper (footnote 4) only P(A | Z) is taken from the auxiliary data; P(Z) comes from
the primary data (m is rescaled to the primary p(z) unless harmonize=False).
"""
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog, minimize_scalar
from scipy.stats import norm


# ----------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------
def _harmonize(r, m, harmonize=True):
    """Drop proxy levels with no primary mass; take P(Z) from r and P(A|Z) from m."""
    r, m = np.asarray(r, float), np.asarray(m, float)
    pz = r.sum(1)
    keep = pz > 0
    if harmonize:
        mz = m.sum(1)
        if np.any(keep & (mz <= 0)):
            raise ValueError("auxiliary data has no mass at a proxy level present in the primary data")
        m = np.where(mz[:, None] > 0, m / np.where(mz > 0, mz, 1)[:, None], 0) * pz[:, None]
    return r[keep], m[keep], keep


def _check_masks(target, cond):
    target, cond = np.asarray(target, bool), np.asarray(cond, bool)
    if np.any(target & ~cond):
        raise ValueError("target must be a subset of cond")
    return target, cond


# ----------------------------------------------------------------------------------------
# 1. closed forms (Props 7-8), binary protected attribute
# ----------------------------------------------------------------------------------------
def closed_form_bounds(r, m, target, cond, ratio=False, harmonize=True):
    """
    Sharp bounds on mu_1 - mu_0 (m columns = (group 0, group 1)), or on mu_1 / mu_0 if ratio.

    The lower bound puts group 1's mass on the target cells at its Frechet lower bound and on the
    remaining cond cells at its upper bound; this simultaneously minimises mu_1 and maximises
    mu_0, and is feasible (Props 7-8). The upper bound is the mirror image. Because both
    extremes are attained by one coupling, any metric increasing in mu_1 and decreasing in mu_0
    (e.g. the ratio) is bounded by the same two couplings.
    """
    target, cond = _check_masks(target, cond)
    r, m, _ = _harmonize(r, m, harmonize)
    if m.shape[1] != 2:
        raise ValueError("closed forms need exactly two groups; use lp_bounds or one_vs_rest")
    P, ma = r.sum(1), m[:, 1]
    h = r[:, target].sum(1)                      # P(target, z)
    g = r[:, cond & ~target].sum(1)              # P(cond but not target, z)
    H, G = h.sum(), g.sum()
    lo = lambda c: np.maximum(0.0, c + ma - P).sum()     # Frechet lower on P(cell, A=1)
    up = lambda c: np.minimum(c, ma).sum()               # Frechet upper

    def rates(T, S):
        X1, X0 = T + S, H + G - T - S
        if X1 <= 0 or X0 <= 0:
            return np.nan, np.nan
        return T / X1, (H - T) / X0

    mu1_lo, mu0_hi = rates(lo(h), up(g))
    mu1_hi, mu0_lo = rates(up(h), lo(g))
    if ratio:
        div = lambda a, b: np.inf if b == 0 else a / b
        return div(mu1_lo, mu0_hi), div(mu1_hi, mu0_lo)
    return mu1_lo - mu0_hi, mu1_hi - mu0_lo


rate_disparity_bounds = closed_form_bounds          # backward-compatible name


def one_vs_rest(m, g):
    """Collapse the auxiliary p(z, a) to (rest, group g), as in the paper's Figs 4 and 7."""
    m = np.asarray(m, float)
    return np.stack([m.sum(1) - m[:, g], m[:, g]], 1)


# ----------------------------------------------------------------------------------------
# 2. optimisation-based bounds (Sections 6, 7.2): LP over couplings, grid over group masses
# ----------------------------------------------------------------------------------------
class CouplingLP:
    """
    Feasible set of couplings q[g, c, z] = P(A=g, C=c, Z=z) of p(c, z) (primary, cells C a
    coarsening of the internal values) and p(z, a) (auxiliary), optionally with the Lipschitz
    assumption |P(A=g | C=c, Z=z) - P(A=g | C=c, Z=z')| <= L d(z, z') (paper eqs. 16-17).

    cells     : length-I integer array mapping internal values to cells (default: identity).
                The Lipschitz constraint is imposed on P(A | cell, Z); without it the cells do
                not matter (Frechet couplings aggregate). The paper uses cells = decision for DD
                and cells = (decision, label) for TPRD / TNRD.
    lipschitz : L (None = no smoothness assumption).
    z_values  : 1-d proxy values; with L, constraints between consecutive sorted levels with
                d = |z - z'| (as in the replication code's smoothing1d).
    z_dist    : alternatively a [Z, Z] distance matrix; constraints between all pairs.
    """

    def __init__(self, r, m, cells=None, lipschitz=None, z_values=None, z_dist=None,
                 harmonize=True):
        r, m, keep = _harmonize(r, m, harmonize)
        Z, I = r.shape
        G = m.shape[1]
        cells = np.arange(I) if cells is None else np.asarray(cells, int)
        C = cells.max() + 1
        rc = np.zeros((Z, C))
        for i in range(I):
            rc[:, cells[i]] += r[:, i]
        self.r, self.m, self.rc, self.cells, self.G = r, m, rc, cells, G
        self.cz = [(c, z) for z in range(Z) for c in range(C) if rc[z, c] > 0]
        npair = len(self.cz)
        self.npair, self.nvar = npair, G * npair
        col = lambda g, p: g * npair + p

        rows, cols, vals, b = [], [], [], []
        k = 0
        for p, (c, z) in enumerate(self.cz):                  # sum_g q[g,c,z] = p(c,z)
            for g in range(G):
                rows.append(k); cols.append(col(g, p)); vals.append(1.0)
            b.append(rc[z, c]); k += 1
        by_z = {}
        for p, (c, z) in enumerate(self.cz):
            by_z.setdefault(z, []).append(p)
        for z, ps in by_z.items():                            # sum_c q[g,c,z] = p(z, a=g)
            for g in range(G):
                for p in ps:
                    rows.append(k); cols.append(col(g, p)); vals.append(1.0)
                b.append(m[z, g]); k += 1
        self.A_eq = sparse.csr_matrix((vals, (rows, cols)), shape=(k, self.nvar))
        self.b_eq = np.array(b)

        self.A_ub, self.b_ub = None, None
        if lipschitz is not None:
            if z_dist is not None:
                D = np.asarray(z_dist, float)[np.ix_(keep, keep)]
                pairs = [(z1, z2, D[z1, z2]) for z1 in range(Z) for z2 in range(z1 + 1, Z)]
            elif z_values is not None:
                zv = np.asarray(z_values, float)[keep]
                o = np.argsort(zv, kind="stable")
                pairs = [(o[j], o[j + 1], zv[o[j + 1]] - zv[o[j]]) for j in range(Z - 1)]
            else:
                raise ValueError("lipschitz needs z_values or z_dist")
            pos = {cz: p for p, cz in enumerate(self.cz)}
            rows, cols, vals, b = [], [], [], []
            k = 0
            for z1, z2, d in pairs:
                for c in range(C):
                    if (c, z1) not in pos or (c, z2) not in pos:
                        continue
                    p1, p2 = pos[(c, z1)], pos[(c, z2)]
                    for g in range(G):
                        for s in (1.0, -1.0):
                            rows += [k, k]; cols += [col(g, p1), col(g, p2)]
                            vals += [s / rc[z1, c], -s / rc[z2, c]]
                            b.append(lipschitz * d); k += 1
            if k:
                self.A_ub = sparse.csr_matrix((vals, (rows, cols)), shape=(k, self.nvar))
                self.b_ub = np.array(b)

    def masks(self, target, cond):
        """Vectors picking out T_g = P(target, A=g) and X_g = P(cond, A=g) for each g."""
        target, cond = _check_masks(target, cond)
        C = self.rc.shape[1]
        cell_t = np.zeros(C, bool); cell_c = np.zeros(C, bool)
        for c in range(C):
            members = self.cells == c
            if members.any():
                if len(set(target[members])) > 1 or len(set(cond[members])) > 1:
                    raise ValueError("cells must not split target or cond")
                cell_t[c], cell_c[c] = target[members][0], cond[members][0]
        T = np.zeros((self.G, self.nvar)); X = np.zeros((self.G, self.nvar))
        for p, (c, z) in enumerate(self.cz):
            for g in range(self.G):
                T[g, g * self.npair + p] = cell_t[c]
                X[g, g * self.npair + p] = cell_c[c]
        return T, X

    def solve(self, c, extra_eq=None, extra_b=None):
        """max c @ q over the feasible set (plus optional equalities). Returns (value, q)."""
        A_eq, b_eq = self.A_eq, self.b_eq
        if extra_eq is not None:
            A_eq = sparse.vstack([A_eq, sparse.csr_matrix(extra_eq)])
            b_eq = np.concatenate([b_eq, extra_b])
        res = linprog(-np.asarray(c), A_ub=self.A_ub, b_ub=self.b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=(0, None), method="highs",
                      options=dict(primal_feasibility_tolerance=1e-10,
                                   dual_feasibility_tolerance=1e-10))
        if res.status != 0:
            return -np.inf, None
        return -res.fun, res.x


def _support(lp, T, X, rho, ref, n_grid, refine):
    """
    h(rho) = max over couplings of sum_{g != ref} rho_g (mu_g - mu_ref).
    Returns (value, rates mu at the maximiser, X masses at the maximiser).
    """
    G = lp.G
    others = [g for g in range(G) if g != ref]
    rho_full = np.zeros(G)
    rho_full[others] = rho
    rho_full[ref] = -np.sum(rho)
    cond_all = np.allclose(X.sum(0), 1.0)               # every cell is in cond: X_g fixed

    def value_at(x):
        if np.any(x <= 1e-15):
            return -np.inf, None
        c = (rho_full / x) @ T
        if cond_all:
            return lp.solve(c)
        return lp.solve(c, extra_eq=X, extra_b=x)

    if cond_all:
        x = np.array([lp.m[:, g].sum() for g in range(G)])
        v, q = value_at(x)
        return v, (T @ q) / x if q is not None else None, x

    # feasible range of each group's condition mass (includes any Lipschitz constraint)
    rng = np.array([[-lp.solve(-X[g])[0], lp.solve(X[g])[0]] for g in range(G)])
    Ktot = lp.solve(X.sum(0))[0]                       # P(cond), identified

    def xvec(free):
        x = np.empty(G)
        x[others] = free
        x[ref] = Ktot - np.sum(free)
        return x

    if G == 2:
        g1 = others[0]
        a, b = rng[g1]
        grid = np.linspace(a, b, n_grid) if b > a else np.array([a])
        vals = np.array([value_at(xvec([t]))[0] for t in grid])
        i = int(np.argmax(vals)); best_t, best_v = grid[i], vals[i]
        if refine and len(grid) > 2 and np.isfinite(best_v):
            lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
            res = minimize_scalar(lambda t: -value_at(xvec([t]))[0], bounds=(lo, hi),
                                  method="bounded", options=dict(xatol=1e-13 * max(Ktot, 1e-12)))
            if np.isfinite(res.fun) and -res.fun > best_v:
                best_t, best_v = res.x, -res.fun
        x = xvec([best_t])
    else:
        axes = [np.linspace(*rng[g], n_grid) for g in others]
        mesh = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, len(others))
        best_v, x = -np.inf, None
        for free in mesh:
            xx = xvec(free)
            if not (rng[ref, 0] - 1e-12 <= xx[ref] <= rng[ref, 1] + 1e-12):
                continue
            v = value_at(xx)[0]
            if v > best_v:
                best_v, x = v, xx
        if refine and x is not None:                    # pattern search around the grid optimum
            step = (rng[others, 1] - rng[others, 0]) / max(n_grid - 1, 1)
            free = x[others].copy()
            for _ in range(60):
                improved = False
                for j in range(len(others)):
                    for s in (1, -1):
                        cand = free.copy(); cand[j] += s * step[j]
                        v = value_at(xvec(cand))[0]
                        if v > best_v + 1e-15:
                            best_v, free, improved = v, cand, True
                if not improved:
                    step = step / 2
                    if np.all(step < 1e-12):
                        break
            x = xvec(free)
    if x is None or not np.isfinite(best_v):
        return np.nan, None, None
    v, q = value_at(x)
    return v, (T @ q) / x, x


def support_function(r, m, target, cond, rho, ref=0, n_grid=201, refine=True, **lp_kw):
    """Support function h(rho) of the partial identification set of (mu_g - mu_ref)_{g != ref}."""
    lp = CouplingLP(r, m, **lp_kw)
    T, X = lp.masks(target, cond)
    v, _, _ = _support(lp, T, X, np.atleast_1d(np.asarray(rho, float)), ref, n_grid, refine)
    return v


def lp_bounds(r, m, target, cond, ref=0, n_grid=None, refine=True, **lp_kw):
    """
    Bounds on each mu_g - mu_ref (the projections h(e_g) and -h(-e_g) of the support function).
    Accepts CouplingLP keywords: cells, lipschitz, z_values, z_dist, harmonize.
    Returns a DataFrame with columns group, lower, upper.
    """
    lp = CouplingLP(r, m, **lp_kw)
    T, X = lp.masks(target, cond)
    G = lp.G
    if n_grid is None:
        n_grid = 201 if G == 2 else 41
    rows = []
    for j, g in enumerate([g for g in range(G) if g != ref]):
        e = np.zeros(G - 1); e[j] = 1.0
        up = _support(lp, T, X, e, ref, n_grid, refine)[0]
        lo = -_support(lp, T, X, -e, ref, n_grid, refine)[0]
        if not (np.isfinite(lo) and np.isfinite(up)):
            if lp.A_ub is not None:
                warnings.warn("LP infeasible: the Lipschitz constant is refuted by the data "
                              "(see min_lipschitz)")
            lo = up = np.nan
        rows.append(dict(group=g, lower=lo, upper=up))
    return pd.DataFrame(rows)


def min_lipschitz(r, m, cells=None, z_values=None, z_dist=None, hi=None, tol=1e-6,
                  harmonize=True):
    """
    Smallest Lipschitz constant compatible with the data (the paper's Fig. 5 uses this value;
    any smaller constant makes the feasible set empty). Found by bisection on feasibility.
    """
    feasible = lambda L: _feasible(r, m, cells, L, z_values, z_dist, harmonize)
    lo = 0.0
    if hi is None:
        hi = 1.0
        while not feasible(hi):
            hi *= 2
            if hi > 1e12:
                raise ValueError("no feasible Lipschitz constant found")
    while hi - lo > tol * max(hi, 1e-12):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if feasible(mid) else (mid, hi)
    return hi


def _feasible(r, m, cells, L, z_values, z_dist, harmonize):
    lp = CouplingLP(r, m, cells=cells, lipschitz=L, z_values=z_values, z_dist=z_dist,
                    harmonize=harmonize)
    return np.isfinite(lp.solve(np.zeros(lp.nvar))[0])


def support_set(r, m, target, cond, ref=0, n_dirs=64, n_grid=41, refine=True, seed=0, **lp_kw):
    """
    Algorithm 2: sample directions on the unit sphere, evaluate the support function, and
    return (directions [n, G-1], h values [n], maximising disparity vectors [n, G-1]).
    Outer approximation: {delta : directions @ delta <= h}; inner: convex hull of the points.
    """
    lp = CouplingLP(r, m, **lp_kw)
    T, X = lp.masks(target, cond)
    d = lp.G - 1
    rng = np.random.default_rng(seed)
    if d == 1:
        dirs = np.array([[1.0], [-1.0]])
    else:
        dirs = rng.normal(size=(n_dirs, d))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    others = [g for g in range(lp.G) if g != ref]
    hs, pts = [], []
    for rho in dirs:
        v, mu, _ = _support(lp, T, X, rho, ref, n_grid, refine)
        hs.append(v)
        pts.append(mu[others] - mu[ref] if mu is not None else np.full(d, np.nan))
    return dirs, np.array(hs), np.array(pts)


# ----------------------------------------------------------------------------------------
# wrapper with the metric names used in fairness_metrics.py
# ----------------------------------------------------------------------------------------
def _eq(v, ref):
    try:
        return float(v) == float(ref)
    except (TypeError, ValueError):
        return str(v) == str(ref)


def _masks(I, internal, decision, label, positive=1, label_positive=1):
    t = lambda i: i if isinstance(i, tuple) else (i,)
    d = internal.index(decision)
    dec = np.array([_eq(t(i)[d], positive) for i in I])
    if label is None:
        return dec, None
    y = internal.index(label)
    return dec, np.array([_eq(t(i)[y], label_positive) for i in I])


# metric -> (target, cond, Lipschitz cells as a function of (decision, label) masks)
PAPER_RATES = {             # covered by Kallus et al. (DD, TPRD, FPR = -TNRD, PPVD, NPVD)
    "demographic_parity": lambda d, y: (d, np.ones_like(d), d.astype(int)),
    "equal_opportunity": lambda d, y: (d & y, y, 2 * d + y),
    "predictive_equality": lambda d, y: (d & ~y, ~y, 2 * d + y),
    "predictive_parity": lambda d, y: (d & y, d, 2 * d + y),
    "negative_predictive_parity": lambda d, y: (~d & ~y, ~d, 2 * d + y),
}
COROLLARY_RATES = {         # not stated in the paper, but follow from the same closed forms
    "accuracy_parity": lambda d, y: (d == y, np.ones_like(d), 2 * d + y),
}


def kallus_bounds(r, m, I, E, internal, decision, label, positive=1, label_positive=1,
                  method="closed_form", ref=0, extended=False, lipschitz=None, z_values=None,
                  z_dist=None, n_grid=None, harmonize=True):
    """
    Bounds for '<rate>_diff' (mu_g - mu_ref; with two groups, last minus first) and '<rate>_gap'
    (|diff|, derived from the diff interval). method = 'closed_form' (binary, no assumptions)
    or 'lp' (any number of groups, optional Lipschitz). With extended=True also accuracy parity
    and, for closed_form, the disparate impact ratio mu_1 / mu_0 (corollaries, not in the paper).
    Returns a DataFrame with columns metric, group, lower, upper.
    """
    dmask, ymask = _masks(I, internal, decision, label, positive, label_positive)
    rates = dict(PAPER_RATES, **(COROLLARY_RATES if extended else {}))
    if ymask is None:
        rates = {"demographic_parity": rates["demographic_parity"]}
        ymask = np.zeros_like(dmask)
    if method == "closed_form" and len(E) != 2:
        raise ValueError(f"closed forms need 2 groups, got {len(E)}; use method='lp' or one_vs_rest")
    if method == "closed_form" and lipschitz is not None:
        raise ValueError("the Lipschitz assumption needs method='lp'")

    rows = []
    for name, fn in rates.items():
        target, cond, cells = fn(dmask, ymask)
        if method == "closed_form":
            res = [(1, *closed_form_bounds(r, m, target, cond, harmonize=harmonize))]
        else:
            df = lp_bounds(r, m, target, cond, ref=ref, n_grid=n_grid, cells=cells,
                           lipschitz=lipschitz, z_values=z_values, z_dist=z_dist,
                           harmonize=harmonize)
            res = list(df[["group", "lower", "upper"]].itertuples(index=False, name=None))
        for g, lo, hi in res:
            gap_lo = 0.0 if lo <= 0 <= hi else min(abs(lo), abs(hi))
            rows += [dict(metric=f"{name}_diff", group=E[g], lower=lo, upper=hi),
                     dict(metric=f"{name}_gap", group=E[g], lower=gap_lo,
                          upper=max(abs(lo), abs(hi)))]
    if extended and method == "closed_form":
        lo, hi = closed_form_bounds(r, m, dmask, np.ones_like(dmask), ratio=True,
                                    harmonize=harmonize)
        rows.append(dict(metric="disparate_impact_ratio", group=E[1], lower=lo, upper=hi))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
# 3. debiased estimation and confidence intervals (Section 7.1, Appendix A.2, A.5)
# ----------------------------------------------------------------------------------------
def _freq_model(z_train, y_train, z_eval):
    """Default nuisance: empirical P(y=1 | z) per discrete proxy level, pooled mean if unseen."""
    s = pd.Series(np.asarray(y_train, float)).groupby(np.asarray(z_train)).mean()
    out = pd.Series(np.asarray(z_eval)).map(s).to_numpy(dtype=float)
    return np.where(np.isnan(out), np.mean(y_train), out)


def _components(eta_S, eta_A, S_pri, A_aux, side):
    """
    Per-unit pieces of the debiased estimate of the extreme P(S, A=alpha) (Lemma 1 / eqs 35-36):
    W = E[lambda(Z)] + E_aux[xi(A, Z)] + E_pri[gamma(S, Z)].
    eta_S, eta_A: dict with arrays at 'all' (lambda units), 'pri', 'aux' units.
    """
    def ind(e_s, e_a):
        return (e_s + e_a - 1 >= 0) if side == "L" else (e_s - e_a <= 0)
    i_all = ind(eta_S["all"], eta_A["all"])
    if side == "L":
        lam = i_all * (eta_S["all"] + eta_A["all"] - 1)
    else:
        lam = i_all * (eta_S["all"] - eta_A["all"]) + eta_A["all"]
    i_pri = ind(eta_S["pri"], eta_A["pri"])
    gam = i_pri * (S_pri - eta_S["pri"])
    if A_aux is None:
        xi = np.zeros(0)
    else:
        i_aux = ind(eta_S["aux"], eta_A["aux"])
        w = i_aux if side == "L" else (1 - i_aux)
        xi = w * (A_aux - eta_A["aux"])
    return lam, gam, xi


def debiased_bounds(z_pri, target_pri, cond_pri, z_aux=None, a_aux=None, *, n_folds=5,
                    nuisance=None, alpha=0.05, seed=0, eta_target_pri=None, eta_other_pri=None,
                    eta_a_pri=None, p_a=None):
    """
    Debiased (cross-fitted) estimates of the binary partial identification interval of
    mu_1 - mu_0 and a (1 - alpha) confidence interval that covers it (Theorems 1-2).

    Primary data: proxy z_pri and boolean target_pri, cond_pri per unit.
    Auxiliary data: proxy z_aux and a_aux in {0, 1} (1 = group 1, e.g. one group vs the rest).

    Demographic disparity (cond all True) uses Algorithm 1: mu(alpha, w) = W / P(A = alpha).
    Other rates use the ratio form of Appendix A.2: mu = W_target / (W_target + W_other).

    nuisance(z_train, y_train, z_eval) -> P(y=1 | z_eval); default: per-level frequencies.
    Cross-fitting: both datasets are split into n_folds folds; nuisances for a fold are fit
    on the other folds.

    Population-auxiliary variant (Appendix A.5, used for HMDA in the paper): pass z_aux=None,
    eta_a_pri = known P(A=1 | Z) at the primary units and p_a = P(A=1). Then the xi terms
    vanish and lambda is averaged over the primary data.
    To reproduce the replication code exactly, pre-computed primary nuisances can be passed as
    eta_target_pri / eta_other_pri (then no cross-fitting is done for them).

    Returns a dict with lower, upper, se_lower, se_upper, ci_lower, ci_upper.
    """
    rng = np.random.default_rng(seed)
    nuisance = nuisance or _freq_model
    z_pri = np.asarray(z_pri)
    T_pri = np.asarray(target_pri, bool)
    C_pri = np.asarray(cond_pri, bool)
    if np.any(T_pri & ~C_pri):
        raise ValueError("target must be a subset of cond")
    O_pri = C_pri & ~T_pri
    dd = bool(np.all(C_pri))
    n_pri = len(z_pri)
    aux_known = z_aux is None

    # ---- nuisances at the 'all' (lambda), 'pri' and 'aux' units, cross-fitted ----
    if aux_known:
        if eta_a_pri is None or p_a is None:
            raise ValueError("without auxiliary data pass eta_a_pri and p_a")
        z_all = z_pri
        eta_a = {"all": np.asarray(eta_a_pri, float), "pri": np.asarray(eta_a_pri, float)}
        A_aux = None
        fold_pri = rng.integers(0, n_folds, n_pri)
        fold_all_pri = fold_pri
    else:
        z_aux = np.asarray(z_aux); A_aux = np.asarray(a_aux, float)
        n_aux = len(z_aux)
        z_all = np.concatenate([z_pri, z_aux])
        fold_pri = rng.permutation(np.arange(n_pri) % n_folds)
        fold_aux = rng.permutation(np.arange(n_aux) % n_folds)
        fold_all = np.concatenate([fold_pri, fold_aux])
        eta_a = {k: np.zeros(n) for k, n in [("all", len(z_all)), ("pri", n_pri), ("aux", n_aux)]}
        for k in range(n_folds):
            tr = fold_aux != k
            eta_a["all"][fold_all == k] = nuisance(z_aux[tr], A_aux[tr], z_all[fold_all == k])
            eta_a["pri"][fold_pri == k] = nuisance(z_aux[tr], A_aux[tr], z_pri[fold_pri == k])
            eta_a["aux"][fold_aux == k] = nuisance(z_aux[tr], A_aux[tr], z_aux[fold_aux == k])
        fold_all_pri = fold_all

    def primary_eta(S, given):
        keys = ["all", "pri"] + ([] if aux_known else ["aux"])
        if given is not None:
            if not aux_known:
                raise ValueError("pre-computed primary nuisances are only supported with z_aux=None")
            g = np.asarray(given, float)
            return {"all": g, "pri": g}
        out = {k: np.zeros(len(z_all) if k == "all" else (n_pri if k == "pri" else len(z_aux)))
               for k in keys}
        for k in range(n_folds):
            tr = fold_pri != k
            out["all"][fold_all_pri == k] = nuisance(z_pri[tr], S[tr], z_all[fold_all_pri == k])
            out["pri"][fold_pri == k] = nuisance(z_pri[tr], S[tr], z_pri[fold_pri == k])
            if not aux_known:
                out["aux"][fold_aux == k] = nuisance(z_pri[tr], S[tr], z_aux[fold_aux == k])
        return out

    eta_T = primary_eta(T_pri.astype(float), eta_target_pri)
    eta_O = primary_eta(O_pri.astype(float), eta_other_pri) if not dd else None
    flip = lambda e: {k: 1 - v for k, v in e.items()}
    eta_b = flip(eta_a)
    A_b = None if A_aux is None else 1 - A_aux

    def W(eta_S, S, eta_A, A, side):
        lam, gam, xi = _components(eta_S, eta_A, S, A, side)
        val = lam.mean() + gam.mean() + (xi.mean() if len(xi) else 0.0)
        return val, (lam, gam, xi)

    if dd:
        pa = float(p_a) if aux_known else A_aux.mean()
        pb = 1 - pa
        specs = {"lower": [(eta_T, T_pri, eta_a, A_aux, "L", 1 / pa),
                           (eta_T, T_pri, eta_b, A_b, "U", -1 / pb)],
                 "upper": [(eta_T, T_pri, eta_a, A_aux, "U", 1 / pa),
                           (eta_T, T_pri, eta_b, A_b, "L", -1 / pb)]}
    else:
        specs = None

    out = {}
    z_crit = norm.ppf(1 - alpha / 2)
    r_n = n_pri / len(z_all) if not aux_known else 1.0
    for which in ("lower", "upper"):
        if dd:
            terms = [(W(e, S, ea, A, sd), coef) for e, S, ea, A, sd, coef in specs[which]]
            est = sum(coef * w[0] for w, coef in terms)
            grads = [coef for _, coef in terms]
            comps = [w[1] for w, _ in terms]
        else:
            sL, sU = ("L", "U") if which == "lower" else ("U", "L")
            w1 = W(eta_T, T_pri, eta_a, A_aux, sL)        # group 1 on target
            w2 = W(eta_O, O_pri, eta_a, A_aux, sU)        # group 1 on rest of cond
            w3 = W(eta_T, T_pri, eta_b, A_b, sU)          # group 0 on target
            w4 = W(eta_O, O_pri, eta_b, A_b, sL)          # group 0 on rest of cond
            v1, v2, v3, v4 = [np.clip(w[0], 0, 1) for w in (w1, w2, w3, w4)]
            s12, s34 = v1 + v2, v3 + v4
            est = v1 / s12 - v3 / s34
            grads = [v2 / s12 ** 2, -v1 / s12 ** 2, -v4 / s34 ** 2, v3 / s34 ** 2]
            comps = [w[1] for w in (w1, w2, w3, w4)]
        phi_lam = sum(g * c[0] for g, c in zip(grads, comps))
        phi_gam = sum(g * c[1] for g, c in zip(grads, comps))
        if aux_known:
            V = np.mean((phi_lam + phi_gam - np.mean(phi_lam + phi_gam)) ** 2)
        else:
            phi_xi = sum(g * c[2] for g, c in zip(grads, comps))
            V = (r_n * np.mean((phi_lam - phi_lam.mean()) ** 2) + np.mean(phi_gam ** 2)
                 + r_n / (1 - r_n) * np.mean(phi_xi ** 2))
        se = np.sqrt(V / n_pri)
        out[which], out["se_" + which] = est, se
    out["ci_lower"] = out["lower"] - z_crit * out["se_lower"]
    out["ci_upper"] = out["upper"] + z_crit * out["se_upper"]
    return out