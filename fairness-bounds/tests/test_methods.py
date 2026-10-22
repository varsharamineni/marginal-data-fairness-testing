import numpy as np
import pandas as pd
import pytest

from fairness import data, methods

A = ["a", "b"]


def synthetic(**over):
    cfg = dict(seed=0, n_primary=200_000, n_auxiliary=200_000, n_z=10, p_z=None, p_a_given_z=None,
               y_intercept=0.0, y_coef_a=0.5, y_coef_z=1.0, x1_coef_y=1.5, x1_coef_a=0.5, x2_coef_y=1.0)
    cfg.update(over)
    prim, aux = data.generate_synthetic(cfg)
    # simple Yhat depending on A through x1, so the disparity is non-zero
    prim["yhat"] = (prim["x1"] > 0.75).astype(int)
    return prim, aux


def test_hand_example():
    """One Z level, P(A=a|Z)=0.5, P(Yhat=1)=0.5: w^L_a(1)=0, w^U_b(1)=1 -> DD in [-1, 1]."""
    prim = pd.DataFrame({"z": ["0"] * 4, "y": [1, 1, 0, 0], "yhat": [1, 0, 1, 0]})
    aux = pd.DataFrame({"z": ["0"] * 2, "a": ["a", "b"]})
    assert np.allclose(methods.kallus_bounds(prim, aux, A, "DD"), (-1.0, 1.0))
    assert np.isclose(methods.ci_estimate(prim, aux, A, "DD"), 0.0)


@pytest.mark.parametrize("metric", methods.METRICS)
def test_truth_and_ci_inside_bounds(metric):
    prim, aux = synthetic()
    a_true = prim.pop("a")
    lo, hi = methods.kallus_bounds(prim, aux, A, metric)
    truth = methods.true_disparity(prim["yhat"], prim["y"], a_true, A, metric)
    ci = methods.ci_estimate(prim, aux, A, metric)
    assert lo <= hi
    assert lo - 0.01 <= truth <= hi + 0.01
    assert lo - 1e-12 <= ci <= hi + 1e-12      # CI weights satisfy the LTP constraints


@pytest.mark.parametrize("metric", methods.METRICS)
def test_bounds_collapse_when_z_determines_a(metric):
    prim, aux = synthetic(p_a_given_z=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    a_true = prim.pop("a")
    lo, hi = methods.kallus_bounds(prim, aux, A, metric)
    truth = methods.true_disparity(prim["yhat"], prim["y"], a_true, A, metric)
    assert np.isclose(lo, hi) and np.isclose(lo, truth)
    assert np.isclose(methods.ci_estimate(prim, aux, A, metric), truth)


def small(n_z=2):
    prim, aux = synthetic(n_primary=5000, n_auxiliary=5000, n_z=n_z)
    prim.pop("a")
    return prim, aux


@pytest.mark.parametrize("method", ["grid", "random", "hit_and_run"])
def test_feasible_joints_match_marginals(method):
    prim, aux = small()
    joints = methods.feasible_joints(prim, aux, A, method=method, num=3, n_samples=50)
    p_zo = prim.groupby(["z", "yhat", "y"]).size() / len(prim)
    p_a = methods.p_a_given_z(aux, A)["a"]
    for j in joints:
        assert np.isclose(j["p"].sum(), 1.0) and (j["p"] >= 0).all()
        assert np.allclose(j.groupby(["z", "yhat", "y"])["p"].sum().reindex(p_zo.index), p_zo)
        by_z = j.groupby(["z", "a"])["p"].sum().unstack()
        assert np.allclose((by_z["a"] / by_z.sum(axis=1)).reindex(p_a.index), p_a)


def test_joint_dd_range_equals_kallus_bounds():
    """DD is linear in the joint, so its extremes are at grid endpoints and equal Eq. (11)."""
    prim, aux = small(n_z=3)
    joints = methods.feasible_joints(prim, aux, A, outcome=("yhat",), num=3)
    dd = methods.joint_metrics(joints, lambda j: methods.joint_disparity(j, A, "DD"))
    assert np.allclose((dd.min(), dd.max()), methods.kallus_bounds(prim, aux, A, "DD"))


@pytest.mark.parametrize("metric", methods.METRICS)
def test_joint_metrics_inside_kallus_bounds(metric):
    prim, aux = small()
    joints = methods.feasible_joints(prim, aux, A, num=4)
    vals = methods.joint_metrics(joints, lambda j: methods.joint_disparity(j, A, metric))
    lo, hi = methods.kallus_bounds(prim, aux, A, metric)
    vals = vals[np.isfinite(vals)]
    assert lo - 1e-9 <= vals.min() and vals.max() <= hi + 1e-9


def test_hit_and_run_is_uniform():
    """m=3 cells, sum=1, no active upper bounds: the slice is the 2-simplex, where
    q_0 ~ Beta(1, 2) under uniformity, so E[q_0] = 1/3 and P(q_0 < 1/2) = 3/4."""
    q = methods._hit_and_run(1.0, np.ones(3), 5000, np.random.default_rng(0), 500, 10)
    assert np.allclose(q.sum(axis=1), 1.0) and (q >= 0).all()
    assert abs(q[:, 0].mean() - 1 / 3) < 0.02
    assert abs((q[:, 0] < 0.5).mean() - 0.75) < 0.02


def test_prepare_rejects_unmatched_z():
    prim, aux = synthetic(n_primary=1000, n_auxiliary=1000)
    cols = {"yhat": "yhat", "y": "y", "a": "a", "z": ["z"], "features": ["x1", "x2"]}
    with pytest.raises(ValueError, match="not in auxiliary"):
        data.prepare(prim, aux[aux["z"] != 0], cols, A)
