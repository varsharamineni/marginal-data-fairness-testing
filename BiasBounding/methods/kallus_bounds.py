"""
Sharp bounds on rate disparities with an unobserved binary protected attribute, in the data
combination setting of Kallus, Mao & Zhou (2022), "Assessing algorithmic fairness with
unobserved protected class using data combination", Management Science 68(3).

Setting: primary data give p(internal, z) with internal = (decision, label); auxiliary data
give p(z, a) for a discrete proxy z (the common variables) and binary a. No further
assumptions. For a rate  mu_a = P(target | cond, A = a)  (target an event within cond):

    demographic disparity   target = {decision=1}             cond = everything
    TPR disparity           target = {decision=1, label=1}    cond = {label=1}
    FPR disparity           target = {decision=1, label=0}    cond = {label=0}
    (PPV disparity          target = {decision=1, label=1}    cond = {decision=1})

the disparity  mu_a - mu_b = T / X - (H - T) / (K - X)  depends on the joint only through
X = P(cond, A=a) and T = P(target, A=a). Within each z the feasible (x_z, t_z) are
    max(0, k_z + m_za - P_z) <= x_z <= min(k_z, m_za)
    max(0, x_z - (k_z - h_z)) <= t_z <= min(x_z, h_z)
(h_z = P(target, z), k_z = P(cond, z), m_za = P(z, A=a)), which is sharp: any such split is
attainable by some joint. For fixed X the disparity increases in T, and the largest and
smallest T given X have a closed form (greedy fill of a separable concave / convex function).
The bounds are then a one-dimensional optimisation over X, done on a dense grid with local
refinement. This is the fractional-optimisation structure of the paper; it is our
implementation for discrete proxies, without the paper's smoothness assumptions or its
sampling-uncertainty tools (estimates are plug-in).
"""
import numpy as np
from scipy.optimize import minimize_scalar


def rate_disparity_bounds(r, m, target, cond, n_grid=20001):
    """
    r: [Z, I] p(internal, z); m: [Z, 2] p(z, a), columns (group b, group a);
    target, cond: boolean masks over internal values. Returns (lower, upper) of mu_a - mu_b.
    """
    P = m.sum(1)
    h, k, ma = r[:, target].sum(1), r[:, cond].sum(1), m[:, 1]
    H, K = h.sum(), k.sum()
    if K <= 0:
        return np.nan, np.nan
    lo, hi = np.maximum(0, k + ma - P), np.minimum(k, ma)
    X0, X1 = lo.sum(), hi.sum()
    base1, cap1 = np.minimum(lo, h).sum(), np.maximum(0, np.minimum(hi, h) - lo).sum()
    base0 = np.maximum(0, lo - (k - h)).sum()
    cap0 = np.maximum(0, np.minimum(hi, k - h) - lo).sum()
    t_max = lambda X: base1 + np.minimum(X - X0, cap1)            # concave in X
    t_min = lambda X: base0 + np.maximum(0, X - X0 - cap0)        # convex in X
    f = lambda X, T: T / X - (H - T) / (K - X)

    eps = 1e-12 * max(K, 1e-300)
    a, b = max(X0, eps), min(X1, K - eps)
    if a > b + 1e-12:
        return np.nan, np.nan
    b = max(a, b)
    Xs = np.linspace(a, b, n_grid) if b > a else np.array([a])

    def extreme(T, sign):
        vals = sign * f(Xs, T(Xs))
        i = int(np.argmax(vals))
        best = vals[i]
        if len(Xs) > 2:
            l, u = Xs[max(i - 1, 0)], Xs[min(i + 1, len(Xs) - 1)]
            res = minimize_scalar(lambda X: -sign * f(X, T(X)), bounds=(l, u), method="bounded",
                                  options=dict(xatol=1e-14))
            best = max(best, -res.fun)
        return sign * best

    return extreme(t_min, -1), extreme(t_max, +1)


def _masks(I, internal, decision, label, positive="1", label_positive="1"):
    d, y = internal.index(decision), internal.index(label)
    dec = np.array([str(i[d]) == str(positive) for i in I])
    lab = np.array([str(i[y]) == str(label_positive) for i in I])
    return dec, lab


# metric name -> (target, cond) as functions of (decision mask, label mask)
KALLUS_RATES = {
    "demographic_parity": lambda d, y: (d, np.ones_like(d)),
    "equal_opportunity": lambda d, y: (d & y, y),
    "predictive_equality": lambda d, y: (d & ~y, ~y),
}
EXTENDED_RATES = {
    "predictive_parity": lambda d, y: (d & y, d),
}


def kallus_bounds(r, m, I, E, internal, decision, label, positive="1", label_positive="1",
                  extended=False):
    """
    Bounds for '<rate>_diff' (last group minus first, matching fairness_metrics.py) and
    '<rate>_gap' (absolute difference) for the rates in KALLUS_RATES (plus EXTENDED_RATES if
    extended). Returns a list of dicts metric, lower, upper. Binary protected attribute only.
    """
    import pandas as pd
    if len(E) != 2:
        print(f"kallus: needs a binary protected attribute, got {len(E)} groups; skipped")
        return pd.DataFrame(columns=["metric", "lower", "upper"])
    dmask, ymask = _masks(I, internal, decision, label, positive, label_positive)
    rates = dict(KALLUS_RATES, **(EXTENDED_RATES if extended else {}))
    rows = []
    for name, fn in rates.items():
        target, cond = fn(dmask, ymask)
        lo, hi = rate_disparity_bounds(r, m, target, cond)
        gap_lo = 0.0 if lo <= 0 <= hi else min(abs(lo), abs(hi))
        rows += [dict(metric=f"{name}_diff", lower=lo, upper=hi),
                 dict(metric=f"{name}_gap", lower=gap_lo, upper=max(abs(lo), abs(hi)))]
    return pd.DataFrame(rows)