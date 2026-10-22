"""Step 3: estimate or bound DD, TPRD and TNRD with A unobserved in the primary data.

Notation follows Kallus, Mao & Zhou (Fairness Using Data Combination), Section 5.
All expectations E[.] are sample means over the primary data (Yhat, Y, Z).
P(A = alpha | Z = z) is estimated from the auxiliary data, and
P(A = alpha) = E[P(A = alpha | Z)] (law of total probability, P(Z) from primary data).

Metrics (group a minus group b):
    DD   = P(Yhat=1 | A=a)      - P(Yhat=1 | A=b)
    TPRD = P(Yhat=1 | Y=1, A=a) - P(Yhat=1 | Y=1, A=b)
    TNRD = P(Yhat=0 | Y=0, A=a) - P(Yhat=0 | Y=0, A=b)

Methods:
    ci_estimate    point estimate under conditional independence of A and the
                   observed outcomes given Z, i.e. w_alpha = P(A = alpha | Z).
    kallus_bounds  closed-form partial identification sets, Propositions 7 and 8.
    feasible_joints  joint distributions P(A, Z, Yhat, Y) consistent with both marginals;
                   any function of a joint (e.g. joint_disparity) can then be evaluated on each.
"""
import itertools

import numpy as np
import pandas as pd

METRICS = ("DD", "TPRD", "TNRD")
_CELL = {"TPRD": (1, 1), "TNRD": (0, 0)}   # (yhat, y) for mu_{yhat y}


# ------------------------------------------------------------------ inputs
def p_a_given_z(auxiliary, a_levels):
    """P(A = alpha | Z = z) from auxiliary data: DataFrame indexed by z, columns a_levels."""
    return (auxiliary.groupby("z")["a"].value_counts(normalize=True)
            .unstack(fill_value=0.0).reindex(columns=list(a_levels), fill_value=0.0))


def _pa_rows(primary, pa_table, alpha):
    """P(A = alpha | Z = Z_i) for every primary row i."""
    return primary["z"].map(pa_table[alpha]).to_numpy(dtype=float)


def _cell_prob_rows(primary, cols):
    """P(cols = observed values | Z = Z_i) for every primary row i, e.g. P(Yhat=Yhat_i | Z_i)."""
    n_cell = primary.groupby(["z"] + cols)["z"].transform("size")
    n_z = primary.groupby("z")["z"].transform("size")
    return (n_cell / n_z).to_numpy(dtype=float)


# ------------------------------------------------------------------ paper quantities
def frechet_weights(pa, p):
    """w^L, w^U of Propositions 7/8, evaluated per row.

    w^L = max{0, 1 + (P(A=alpha|Z) - 1) / p},   w^U = min{1, P(A=alpha|Z) / p},
    with p = P(Yhat=yhat | Z=z) (Prop. 7) or P(Yhat=yhat, Y=y | Z=z) (Prop. 8).
    """
    return np.maximum(0.0, 1.0 + (pa - 1.0) / p), np.minimum(1.0, pa / p)


def mu(w, yhat, p_alpha):
    """Eq. (7): mu(alpha; w) = E[w_alpha(Yhat, Z) Yhat] / P(A = alpha)."""
    return np.mean(w * yhat) / p_alpha


def mu_prime(w, w_prime, yhat, y, yhat_val, y_val):
    """Prop. 8: mu'_{yhat y}(alpha; w, w') =
        E[w I(Y=y) I(Yhat=yhat)] / (E[w I(Y=y) I(Yhat=yhat)] + E[w' I(Y=y) I(Yhat!=yhat)]).
    With w' = w this is Eq. (8), mu_{yhat y}(alpha; w)."""
    num = np.mean(w * (y == y_val) * (yhat == yhat_val))
    return num / (num + np.mean(w_prime * (y == y_val) * (yhat != yhat_val)))


def _inputs(primary, auxiliary, a_levels):
    a, b = a_levels
    table = p_a_given_z(auxiliary, a_levels)
    pa_a, pa_b = _pa_rows(primary, table, a), _pa_rows(primary, table, b)
    return primary["yhat"].to_numpy(), primary["y"].to_numpy(), pa_a, pa_b


# ------------------------------------------------------------------ methods
def ci_estimate(primary, auxiliary, a_levels, metric):
    """Point estimate assuming A is independent of Yhat (DD) or of (Yhat, Y) (TPRD, TNRD)
    given Z, so that w*_alpha = P(A = alpha | Z)."""
    yhat, y, pa_a, pa_b = _inputs(primary, auxiliary, a_levels)
    if metric == "DD":
        return mu(pa_a, yhat, pa_a.mean()) - mu(pa_b, yhat, pa_b.mean())
    k, c = _CELL[metric]
    return mu_prime(pa_a, pa_a, yhat, y, k, c) - mu_prime(pa_b, pa_b, yhat, y, k, c)


def kallus_bounds(primary, auxiliary, a_levels, metric):
    """(lower, upper) of the partial identification set: Eq. (11), (14), (15)."""
    yhat, y, pa_a, pa_b = _inputs(primary, auxiliary, a_levels)

    if metric == "DD":                                     # Proposition 7, Eq. (11)
        p = _cell_prob_rows(primary, ["yhat"])
        wL_a, wU_a = frechet_weights(pa_a, p)
        wL_b, wU_b = frechet_weights(pa_b, p)
        P_a, P_b = pa_a.mean(), pa_b.mean()
        lower = mu(wL_a, yhat, P_a) - mu(wU_b, yhat, P_b)
        upper = mu(wU_a, yhat, P_a) - mu(wL_b, yhat, P_b)
        return lower, upper

    k, c = _CELL[metric]                                   # Proposition 8, Eq. (14)/(15)
    p = _cell_prob_rows(primary, ["yhat", "y"])
    wL_a, wU_a = frechet_weights(pa_a, p)
    wL_b, wU_b = frechet_weights(pa_b, p)
    lower = mu_prime(wL_a, wU_a, yhat, y, k, c) - mu_prime(wU_b, wL_b, yhat, y, k, c)
    upper = mu_prime(wU_a, wL_a, yhat, y, k, c) - mu_prime(wL_b, wU_b, yhat, y, k, c)
    return lower, upper


# ------------------------------------------------------------------ feasible joints
def _grid_splits(s, u, num):
    """Grid of q with 0 <= q <= u and sum(q) = s; the last entry is determined by the others."""
    if len(u) == 1:
        return [[s]]
    lo, hi = max(0.0, s - u[1:].sum()), min(u[0], s)
    return [[q] + rest for q in np.unique(np.linspace(lo, hi, num))
            for rest in _grid_splits(s - q, u[1:], num)]


def _random_split(s, u, rng):
    """One random q with 0 <= q <= u and sum(q) = s (cells filled in random order)."""
    q, rem, left = np.zeros(len(u)), s, u.sum()
    for i in rng.permutation(len(u)):
        left -= u[i]
        hi = min(u[i], rem)
        q[i] = rng.uniform(min(max(0.0, rem - left), hi), hi)          # min() guards round-off
        rem -= q[i]
    return q


def _hit_and_run(s, u, n, rng, burn_in, thin):
    """n draws, approximately uniform on {q : 0 <= q <= u, sum(q) = s}, by hit-and-run:
    from the current q pick a random direction d with sum(d) = 0, find the chord
    {q + t d} inside the box, and move to a uniform point on it."""
    free = u > 1e-12                                                    # cells with u = 0 stay at 0
    q = np.zeros(len(u))
    q[free] = s * u[free] / u[free].sum() if u.sum() > 0 else 0.0       # interior start
    if free.sum() < 2 or s <= 1e-12 or s >= u.sum() - 1e-12:            # polytope is one point
        return np.tile(q, (n, 1))
    x, ub, out = q[free], u[free], []
    for it in range(burn_in + n * thin):
        d = rng.standard_normal(len(x))
        d -= d.mean()
        d /= np.linalg.norm(d)
        with np.errstate(divide="ignore"):
            t0, t1 = -x / d, (ub - x) / d                               # where each cell hits 0 / u
        lo, hi = np.minimum(t0, t1).max(), np.maximum(t0, t1).min()
        x = np.clip(x + rng.uniform(lo, hi) * d, 0.0, ub)
        if it >= burn_in and (it - burn_in) % thin == thin - 1:
            out.append(x.copy())
    q = np.tile(q, (n, 1))
    q[:, free] = out
    return q


def _joint_marginals(primary, auxiliary, a_levels, outcome):
    """Z levels, outcome cells, P(Z), P(O | Z) (K x m) and P(A=a_levels[0] | Z)."""
    z_levels = np.sort(primary["z"].unique())
    cells = list(itertools.product([0, 1], repeat=len(outcome)))
    vals, z = primary[list(outcome)].to_numpy(), primary["z"].to_numpy()
    counts = np.array([[((z == zv) & (vals == c).all(axis=1)).sum() for c in cells] for zv in z_levels])
    p_z = counts.sum(axis=1) / counts.sum()
    p_o = counts / counts.sum(axis=1, keepdims=True)
    p_a = p_a_given_z(auxiliary, a_levels).loc[z_levels, a_levels[0]].to_numpy()
    return z_levels, cells, p_z, p_o, p_a


def grid_size(primary, auxiliary, a_levels, outcome=("yhat", "y"), num=5):
    """Number of joints feasible_joints(method="grid") would build, without building them."""
    _, _, _, p_o, p_a = _joint_marginals(primary, auxiliary, a_levels, outcome)
    return int(np.prod([len(_grid_splits(s, u, num)) for s, u in zip(p_a, p_o)], dtype=float))


def feasible_joints(primary, auxiliary, a_levels, outcome=("yhat", "y"), method="grid",
                    num=5, n_samples=1000, seed=0, max_joints=100_000,
                    min_value=0.0, max_value=1.0, burn_in=1000, thin=10):
    """Joint distributions P(A, Z, O), O = outcome columns, that reproduce both marginals:
    P(Z, O) from the primary data and P(A | Z) from the auxiliary data.

    For each z the free part is q(o) = P(A=a, O=o | Z=z), subject to
    0 <= q(o) <= P(O=o | Z=z) and sum_o q(o) = P(A=a | Z=z); then
    P(A=b, O=o | Z=z) = P(O=o | Z=z) - q(o).

    method  "grid":   `num` values per free parameter, all combinations across Z levels
                      (raises if more than `max_joints`).
            "random": `n_samples` random feasible joints.
            "hit_and_run": `n_samples` joints approximately uniform over the feasible set,
                      from one hit-and-run chain per Z level (`burn_in` steps discarded,
                      then every `thin`-th step kept).
    min_value, max_value: keep only joints whose cell probabilities all lie in this range.

    Returns a list of DataFrames with columns a, z, *outcome, p (p sums to 1).
    """
    outcome = list(outcome)
    z_levels, cells, p_z, p_o, p_a = _joint_marginals(primary, auxiliary, a_levels, outcome)

    if method == "grid":
        options = [np.array(_grid_splits(s, u, num)) for s, u in zip(p_a, p_o)]
        n = int(np.prod([len(o) for o in options], dtype=float))
        if n > max_joints:
            raise ValueError(f"grid has {n} joints (> max_joints={max_joints}); "
                             "lower num or use method='random'")
        qs = (np.stack(combo) for combo in itertools.product(*options))
    elif method == "random":
        rng = np.random.default_rng(seed)
        qs = (np.stack([_random_split(s, u, rng) for s, u in zip(p_a, p_o)]) for _ in range(n_samples))
    elif method == "hit_and_run":
        rng = np.random.default_rng(seed)
        chains = np.stack([_hit_and_run(s, u, n_samples, rng, burn_in, thin)
                           for s, u in zip(p_a, p_o)], axis=1)              # n_samples x K x m
        qs = iter(chains)
    else:
        raise ValueError("method must be 'grid', 'random' or 'hit_and_run'")

    template = pd.DataFrame([(g, zv, *c) for g in a_levels for zv in z_levels for c in cells],
                            columns=["a", "z"] + outcome)
    joints = []
    for q in qs:                                                        # q: K x m
        p = np.stack([p_z[:, None] * q, p_z[:, None] * (p_o - q)]).ravel()
        p = np.clip(p, 0.0, None)                                       # float round-off
        if p.min() >= min_value - 1e-12 and p.max() <= max_value + 1e-12:
            joints.append(template.assign(p=p))
    return joints


def joint_disparity(joint, a_levels, metric):
    """DD, TPRD or TNRD (a_levels[0] minus a_levels[1]) computed from one joint distribution."""
    rate = []
    for g in a_levels:
        j = joint[joint["a"] == g]
        if metric != "DD":
            k, c = _CELL[metric]
            j = j[j["y"] == c]
        hit = j["yhat"] == (1 if metric == "DD" else k)
        rate.append(j.loc[hit, "p"].sum() / j["p"].sum())
    return rate[0] - rate[1]


def joint_metrics(joints, func):
    """func(joint) for every feasible joint, e.g.
    joint_metrics(joints, lambda j: joint_disparity(j, a_levels, "TPRD"))."""
    return np.array([func(j) for j in joints])


# ------------------------------------------------------------------ ground truth
def true_disparity(yhat, y, a, a_levels, metric):
    """The metric computed with the true A (only available for synthetic data)."""
    yhat, y, a = np.asarray(yhat), np.asarray(y), np.asarray(a)
    rate = []
    for g in a_levels:
        if metric == "DD":
            rate.append(yhat[a == g].mean())
        else:
            k, c = _CELL[metric]
            rate.append((yhat[(a == g) & (y == c)] == k).mean())
    return rate[0] - rate[1]
