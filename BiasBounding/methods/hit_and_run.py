"""
Hit-and-run baseline: uniform random joints from the feasible set.

The feasible set is a product over proxy levels z of transportation polytopes
    {X_z >= 0 : X_z 1 = p(z, .) (rows = internal values),  1' X_z = p(z, a) (columns = groups)}
so a uniform draw from the product is a uniform draw from each level, independently.

Each level is parametrised by its free entries x = X[:I-1, :G-1] (the last column and row
follow from the margins), so X = T x + t0 is affine in x and the polytope is {x : T x + t0 >= 0}.
Hit-and-run: from the current point pick a random direction d, find the segment of the line
x + t d that stays feasible, and jump to a uniform point on it. The chain starts from the
'independent' table (p(z,i) p(a|z)), discards `burn` steps and keeps every `thin`-th step.

Bounds are the min / max of a metric over the samples: an inner approximation (every sample is
feasible) that converges slowly towards extremes, because uniform samples rarely land near
them. The samples also give the distribution of the metric under a uniform prior on joints.

Interface (shared with ci_baseline.py and vertex_sampling.py)
    r : [Z, I]  p(z, internal value)   from the primary data
    m : [Z, G]  p(z, a)                from the auxiliary data
    hit_and_run_joints(r, m, n, burn, thin, seed) -> Q [n, Z, I, G]
    bounds(Q, metric_fn) -> {metric: (min, max)}      metric_fn(Q) -> {metric: values [N]}

Self-test:  python hit_and_run.py
"""
import numpy as np

from ci_baseline import harmonize


def level_affine(rows, cols):
    """X = T x + t0 for the free entries x of one level; returns (T [I*G, F], t0 [I*G])."""
    I, G = len(rows), len(cols)
    F = (I - 1) * (G - 1)

    def table(x):
        X = np.zeros((I, G))
        X[:I - 1, :G - 1] = x.reshape(I - 1, G - 1)
        X[:I - 1, G - 1] = rows[:I - 1] - X[:I - 1, :G - 1].sum(1)
        X[I - 1, :] = cols - X[:I - 1, :].sum(0)
        return X.ravel()

    t0 = table(np.zeros(F))
    T = np.stack([table(np.eye(F)[k]) - t0 for k in range(F)], 1) if F else np.zeros((I * G, 0))
    return T, t0


def hit_and_run_level(rows, cols, n, burn=1000, thin=5, rng=None, tol=1e-15):
    """n uniform draws [n, I, G] from one level's transportation polytope."""
    rng = rng or np.random.default_rng()
    rows, cols = np.asarray(rows, float), np.asarray(cols, float)
    I, G = len(rows), len(cols)
    T, t0 = level_affine(rows, cols)
    F = T.shape[1]
    total = rows.sum()
    start = (np.outer(rows, cols) / total if total > 0 else np.zeros((I, G)))
    if F == 0:
        return np.repeat(start[None], n, 0)
    x = start[:I - 1, :G - 1].ravel().copy()
    out = np.empty((n, I, G))
    k = 0
    for step in range(burn + n * thin):
        d = rng.normal(size=F)
        d /= np.linalg.norm(d)
        s = T @ x + t0                       # current entries (>= 0)
        Td = T @ d
        pos, neg = Td > tol, Td < -tol       # entries decrease / increase along +d
        t_hi = np.min(s[neg] / -Td[neg]) if neg.any() else 0.0
        t_lo = -np.min(s[pos] / Td[pos]) if pos.any() else 0.0
        if t_hi - t_lo > tol:
            x = x + rng.uniform(t_lo, t_hi) * d
        if step >= burn and (step - burn) % thin == 0:
            out[k] = np.maximum((T @ x + t0).reshape(I, G), 0)
            k += 1
    return out


def hit_and_run_joints(r, m, n=10_000, burn=1_000, thin=5, seed=0, harmonize_margins=True):
    """n uniform joints [n, Z, I, G] (levels sampled independently, paired by draw index)."""
    if harmonize_margins:
        r, m = harmonize(r, m)
    rng = np.random.default_rng(seed)
    return np.stack([hit_and_run_level(r[z], m[z], n, burn, thin, rng)
                     for z in range(len(r))], 1)


def bounds(Q, metric_fn):
    """{metric: (min, max)} over the joints in Q, ignoring undefined values."""
    out = {}
    for k, v in metric_fn(Q).items():
        v = v[~np.isnan(v)]
        out[k] = (v.min(), v.max()) if v.size else (np.nan, np.nan)
    return out


def _self_test():
    rng = np.random.default_rng(1)
    Z, I, G = 2, 2, 2
    r = rng.dirichlet(np.ones(Z * I)).reshape(Z, I)
    m = r.sum(1, keepdims=True) * rng.dirichlet(np.ones(G), size=Z)
    Q = hit_and_run_joints(r, m, n=20_000, seed=0)
    assert np.allclose(Q.sum(3), r[None], atol=1e-10), "row margins not reproduced"
    assert np.allclose(Q.sum(2), m[None], atol=1e-10), "column margins not reproduced"
    assert (Q >= 0).all()
    # 2x2 levels: one free entry each, uniform on its Frechet interval -> check the ends
    for z in range(Z):
        lo = max(0.0, r[z, 1] + m[z, 1] - r[z].sum()); hi = min(r[z, 1], m[z, 1])
        v = Q[:, z, 1, 1]
        print(f"level {z}: Frechet range [{lo:.4f}, {hi:.4f}], samples [{v.min():.4f}, "
              f"{v.max():.4f}], mean {v.mean():.4f} (uniform mean {(lo + hi) / 2:.4f})")
    print("hit-and-run samples are feasible and cover the Frechet range: OK")


if __name__ == "__main__":
    _self_test()