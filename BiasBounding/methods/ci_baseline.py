"""
Conditional-independence baseline: a point estimate, not bounds.

Assumption: the protected attribute is independent of the internal variable (decision, label)
given the proxy,  A _||_ internal | Z.  The joint is then fully determined:

    q(z, i, a) = p(z, i) * p(a | z)

This is the usual plug-in / weighting approach with proxy probabilities (e.g. weighting by
BISG probabilities). The resulting table is one of the joints consistent with both datasets
(the 'independent' coupling within each proxy level), so its metric values always lie inside
the sharp bounds; they equal the truth only if the assumption holds.

Interface (shared with hit_and_run.py and vertex_sampling.py)
    r : [Z, I]  p(z, internal value)   from the primary data
    m : [Z, G]  p(z, a)                from the auxiliary data
    ci_joint(r, m) -> Q [1, Z, I, G]

As in kallus_bounds.py, the auxiliary data provide only p(a | z); p(z) is taken from r
(harmonize=False keeps m as given).

Self-test:  python ci_baseline.py
"""
import numpy as np


def harmonize(r, m):
    """Keep p(a | z) from m and rescale it to the primary p(z)."""
    r, m = np.asarray(r, float), np.asarray(m, float)
    pz = r.sum(1)
    mz = m.sum(1, keepdims=True)
    return r, np.where(mz > 0, m / np.where(mz > 0, mz, 1), 0) * pz[:, None]


def ci_joint(r, m, harmonize_margins=True):
    """The conditional-independence joint q[z, i, a] = p(z, i) p(a | z), shape [1, Z, I, G]."""
    if harmonize_margins:
        r, m = harmonize(r, m)
    r, m = np.asarray(r, float), np.asarray(m, float)
    pz = r.sum(1)
    pa_z = np.where(pz[:, None] > 0, m / np.where(pz > 0, pz, 1)[:, None], 0)
    return (r[:, :, None] * pa_z[:, None, :])[None]


def _self_test():
    rng = np.random.default_rng(0)
    Z, I, G = 3, 2, 2
    r = rng.dirichlet(np.ones(Z * I)).reshape(Z, I)
    pa_z = rng.dirichlet(np.ones(G), size=Z)
    m = r.sum(1, keepdims=True) * pa_z
    Q = ci_joint(r, m)
    assert Q.shape == (1, Z, I, G)
    assert np.allclose(Q[0].sum(2), r), "row margins p(z, i) not reproduced"
    assert np.allclose(Q[0].sum(1), m), "column margins p(z, a) not reproduced"
    mu = Q[0].sum(0)[1] / Q[0].sum(0).sum(0)
    print("conditional-independence joint reproduces both margins: OK")
    print(f"P(D=1 | A=g) under the assumption: {np.round(mu, 4)}; DD = {mu[1] - mu[0]:.4f}")


if __name__ == "__main__":
    _self_test()