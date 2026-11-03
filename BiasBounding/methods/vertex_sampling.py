"""
Vertex enumeration / sampling baseline.

The feasible set is a product over proxy levels z of transportation polytopes
    {X_z >= 0 : X_z 1 = p(z, .) (rows = internal values),  1' X_z = p(z, a) (columns = groups)}
Its vertices are exactly the combinations of one vertex per level. Each level's vertices are
found by basis enumeration: a vertex has at most I + G - 1 non-zero entries, so we try every
set of I + G - 1 cells, solve the margin equations on those cells, and keep non-negative
solutions (duplicates from degenerate margins are removed).

  full enumeration  if the number of combinations is at most `cap`: every vertex of the
                    feasible set is evaluated. Exact for every bound attained at a vertex
                    (linear and linear-fractional metrics, ratios of rates, maxima of convex
                    functions such as mutual information); never reaches interior extremes.
  sampling          otherwise, `n` combinations are drawn uniformly at random: an inner
                    approximation.

Interface (shared with ci_baseline.py and hit_and_run.py)
    r : [Z, I]  p(z, internal value)   from the primary data
    m : [Z, G]  p(z, a)                from the auxiliary data
    level_vertices(rows, cols) -> [V, I, G]
    iter_vertex_joints(r, m, n, cap, chunk, seed) -> yields Q [k, Z, I, G] chunks
    vertex_bounds(r, m, metric_fn, n, cap, seed) -> ({metric: (min, max)}, info)

Self-test:  python vertex_sampling.py
"""
import itertools

import numpy as np

from ci_baseline import harmonize


def level_vertices(rows, cols, tol=1e-12):
    """All vertices [V, I, G] of {X >= 0 : X 1 = rows, 1' X = cols}."""
    rows, cols = np.asarray(rows, float), np.asarray(cols, float)
    I, G = len(rows), len(cols)
    A = np.zeros((I + G, I * G))
    for i in range(I):
        A[i, i * G:(i + 1) * G] = 1
    for j in range(G):
        A[I + j, j::G] = 1
    b = np.r_[rows, cols]
    V = []
    for S in itertools.combinations(range(I * G), I + G - 1):
        AS = A[:, S]
        x, *_ = np.linalg.lstsq(AS, b, rcond=None)
        if (x < -tol).any() or np.abs(AS @ x - b).max() > 1e-10:
            continue
        v = np.zeros(I * G)
        v[list(S)] = np.maximum(x, 0)
        V.append(v.reshape(I, G))
    return np.unique(np.round(np.array(V), 14), axis=0)


def iter_vertex_joints(r, m, n=100_000, cap=2_000_000, chunk=200_000, seed=0,
                       harmonize_margins=True, info=None):
    """
    Yield chunks of vertex joints [k, Z, I, G]: every combination if there are at most `cap`,
    otherwise `n` combinations drawn uniformly. `info` (a dict) receives the details.
    """
    if harmonize_margins:
        r, m = harmonize(r, m)
    V = [level_vertices(r[z], m[z]) for z in range(len(r))]
    sizes = [len(v) for v in V]
    total = int(np.prod(sizes, dtype=float))
    full = total <= cap
    if info is not None:
        info.update(vertices_per_level=sizes, combinations=total, full_enumeration=full,
                    evaluated=total if full else n)
    rng = np.random.default_rng(seed)
    count = total if full else n
    for start in range(0, count, chunk):
        k = min(chunk, count - start)
        if full:
            idx = np.unravel_index(np.arange(start, start + k), sizes)
        else:
            idx = [rng.integers(0, s, k) for s in sizes]
        yield np.stack([V[z][idx[z]] for z in range(len(V))], 1)


def vertex_bounds(r, m, metric_fn, n=100_000, cap=2_000_000, seed=0, harmonize_margins=True):
    """({metric: (min, max)}, info) over vertex joints, streamed in chunks."""
    info, best = {}, {}
    for Q in iter_vertex_joints(r, m, n, cap, seed=seed, harmonize_margins=harmonize_margins,
                                info=info):
        for k, v in metric_fn(Q).items():
            v = v[~np.isnan(v)]
            if v.size:
                lo, hi = best.get(k, (np.inf, -np.inf))
                best[k] = (min(lo, v.min()), max(hi, v.max()))
    return best, info


def _self_test():
    # a 2x2 level has exactly its two Frechet extremes as vertices
    rows, cols = np.array([0.3, 0.2]), np.array([0.1, 0.4])
    V = level_vertices(rows, cols)
    lo, hi = max(0.0, rows[1] + cols[1] - rows.sum()), min(rows[1], cols[1])
    assert len(V) == 2 and np.allclose(sorted(V[:, 1, 1]), [lo, hi])
    print(f"2x2 level: vertices at X[1,1] = {sorted(V[:, 1, 1])} (Frechet [{lo}, {hi}]): OK")
    # every vertex reproduces the margins; larger level
    rng = np.random.default_rng(2)
    rows, cols = rng.dirichlet(np.ones(4)) * 0.6, rng.dirichlet(np.ones(3)) * 0.6
    V = level_vertices(rows, cols)
    assert np.allclose(V.sum(2), rows) and np.allclose(V.sum(1), cols) and (V >= 0).all()
    print(f"4x3 level: {len(V)} vertices, all feasible: OK")
    # a linear function's extremes over the polytope equal its extremes over the vertices
    from scipy.optimize import linprog
    c = rng.normal(size=12)
    A_eq = np.vstack([np.kron(np.eye(4), np.ones(3)), np.kron(np.ones(4), np.eye(3))])
    lp = linprog(c, A_eq=A_eq, b_eq=np.r_[rows, cols], bounds=(0, None), method="highs")
    assert np.isclose(lp.fun, (V.reshape(len(V), -1) @ c).min())
    print("LP minimum of a linear function equals the minimum over vertices: OK")


if __name__ == "__main__":
    _self_test()