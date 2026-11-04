"""
Shared helpers for the COMPAS ranking and synthetic experiments.

Tables are numpy arrays indexed [common c, internal i, external e]:
  r[c, i] = p(common, internal)   (primary / internal data)
  m[c, e] = p(common, external)   (auxiliary / external data)
Joints from the grid are J[n, c, i, e] (as returned by grid_joint.solve_joint_distribution
with as_dicts=False). Margins are harmonised as in grid_joint: P(c) from the auxiliary table,
p(i | c) from the primary one.
"""
import contextlib
import io
import time

import numpy as np
from scipy.optimize import linprog, minimize

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "methods"))
from grid_joint import solve_joint_distribution  

STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
}
INK, TRUTH, REF, GRIDC = "#222222", "#E69F00", "#BDBDBD", "#ECECEC"
C_GRID, C_EXACT, C_CI = "darkorange", "#0072B2", "#222222"
C_A, C_B, C_DIFF = "#0072B2", "#CC79A7", "#009E73"


def code(k):
    """Zero-padded string codes so grid_joint's string sort keeps numeric order."""
    return f"{k:03d}"


def harmonize(r, m):
    pc = m.sum(1)
    rs = r.sum(1, keepdims=True)
    return np.where(rs > 0, r / np.where(rs > 0, rs, 1), 0) * pc[:, None], m / m.sum()


def enumerate_joints(r, m, nums, max_joints=2_000_000, method="sequential"):
    """Finest num (largest first) whose enumeration stays under max_joints.
    Returns (J [n, c, i, e], num, seconds); c rows with no primary mass are dropped, so the
    caller should pass tables without empty rows."""
    C, I = r.shape
    E = m.shape[1]
    pp = {(code(i), code(c)): r[c, i] for c in range(C) for i in range(I)}
    pa = {(code(c), code(e)): m[c, e] for c in range(C) for e in range(E)}
    for num in sorted(nums, reverse=True):
        t0 = time.perf_counter()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                J, _ = solve_joint_distribution(pp, pa, num=num, method=method,
                                                max_joints=max_joints, as_dicts=False)
            return J, num, time.perf_counter() - t0
        except SystemExit:
            continue
    raise SystemExit(f"no num in {nums} stays under {max_joints:,} joints")


def ci_joint(r, m):
    """Conditional-independence joint q[c, i, e] = p(i | c) p(e | c) P(c)."""
    r, m = harmonize(r, m)
    pc = m.sum(1)
    return np.where(pc[:, None, None] > 0,
                    r[:, :, None] * m[:, None, :] / np.where(pc > 0, pc, 1)[:, None, None], 0)


def cond_mi(Q, r, m):
    """I(internal; external | common) in bits for each joint Q[n, c, i, e]."""
    expected = ci_joint(r, m)[None]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where((Q > 0) & (expected > 0), Q * np.log2(Q / expected), 0.0)
    return np.maximum(t.sum(axis=(1, 2, 3)), 0.0)


# --------------------------------------------------------------- exact reference bounds
def _constraints(r, m):
    """Equality constraints A q = b for q[c, i, e] >= 0 with the harmonised margins."""
    r, m = harmonize(r, m)
    C, I = r.shape
    E = m.shape[1]
    idx = np.arange(C * I * E).reshape(C, I, E)
    rows, b = [], []
    for c in range(C):
        for i in range(I):
            a = np.zeros(C * I * E); a[idx[c, i, :]] = 1; rows.append(a); b.append(r[c, i])
        for e in range(E - 1):                       # one column per c is implied
            a = np.zeros(C * I * E); a[idx[c, :, e]] = 1; rows.append(a); b.append(m[c, e])
    return np.array(rows), np.array(b), (C, I, E)


def lp_bounds(r, m, num_w, den_w=None):
    """Exact bounds of (num_w . q) / (den_w . q) over the feasible set (Charnes-Cooper);
    den_w=None means a linear objective. Weights are arrays shaped [c, i, e]."""
    A, b, shape = _constraints(r, m)
    nv = A.shape[1]
    out = []
    for sign in (1, -1):
        if den_w is None:
            res = linprog(sign * num_w.ravel(), A_eq=A, b_eq=b, bounds=[(0, None)] * nv,
                          method="highs")
            out.append(sign * res.fun)
        else:                       # variables (y, t): y = t q, den . y = 1, A y = t b
            c = np.concatenate([sign * num_w.ravel(), [0]])
            Aeq = np.vstack([np.hstack([A, -b[:, None]]),
                             np.concatenate([den_w.ravel(), [0]])[None]])
            beq = np.concatenate([np.zeros(len(b)), [1]])
            res = linprog(c, A_eq=Aeq, b_eq=beq, bounds=[(0, None)] * (nv + 1), method="highs")
            out.append(sign * res.fun)
    return out[0], out[1]


def convex_min(r, m, f, starts=None):
    """Minimum of a convex function f(q[c, i, e]) over the feasible set (SLSQP from the CI
    joint and any extra starts; convexity makes the local minimum global)."""
    A, b, shape = _constraints(r, m)
    x0s = [ci_joint(r, m).ravel()] + [s.ravel() for s in (starts or [])]
    best = np.inf
    for x0 in x0s:
        res = minimize(lambda x: f(x.reshape(shape)), x0, method="SLSQP",
                       bounds=[(0, 1)] * len(x0),
                       constraints=[{"type": "eq", "fun": lambda x: A @ x - b}],
                       options=dict(maxiter=500, ftol=1e-12))
        if res.success or np.abs(A @ res.x - b).max() < 1e-7:
            best = min(best, float(f(res.x.reshape(shape))))
    return best