"""
Bounds on fairness metrics over the joint distributions enumerated by grid_joint.py.

Every joint that grid_joint returns is consistent with both datasets, so evaluating a metric
on each joint and taking the min and max over joints bounds that metric. The metric is only
evaluated pointwise, so it can have any form: ratios, max/min over groups, maxima of several
gaps, information measures, etc. No linearity or convexity is needed.

Internal variables: the decision (required) and, for error-based metrics, the true label:
    --internal approved repaid --decision approved --label repaid
External variable(s): the protected attribute(s). With more than two groups, "gap" metrics
use the largest minus the smallest group value and "ratio" metrics the smallest over the
largest.

Randomised or proxy-only decisions (--randomised): if the decision depends only on the
common and other internal variables, plus independent noise (e.g. a synthetic predictor from
predictions.py without feature weights), it is independent of the protected attribute given
those. --randomised imposes this: the grid runs over the other internal variables only and
the decision is added as p(decision | common, others). Without it the bounds are still valid,
but wider than necessary. Do not use it for a model that uses features outside the common
variables: their relation to the protected attribute is unknown.

Command line:
  python fairness_metrics.py primary.csv auxiliary.csv --internal approved repaid \
      --common income_bin --external race --decision approved --positive 1 \
      --label repaid --label-positive 1 --num 5 --method sequential
Python:
  from grid_joint import solve_joint_distribution
  from fairness_metrics import fairness_bounds
  J, (I, C, E) = solve_joint_distribution(p_primary, p_aux, num=5, as_dicts=False)
  metric_table, rate_table, values = fairness_bounds(
      J, I, C, E, internal=["approved", "repaid"], decision="approved", label="repaid")

Adding a metric:
  from fairness_metrics import metric, gap
  @metric("selection_rate_gap_squared")
  def _(R):                  # R: dict from group_quantities, arrays [N, groups] (or [N, C, groups])
      return gap(R["sel"]) ** 2      # must return an array [N], one value per joint
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd

from grid_joint import from_datasets, solve_joint_distribution

EPS = 1e-12
METRICS = {}


def metric(name, needs_label=False):
    """Register f(R) -> array [N] under `name`; its docstring is used as the description."""
    def register(fn):
        METRICS[name] = dict(fn=fn, needs_label=needs_label,
                             description=" ".join((fn.__doc__ or "").split()))
        return fn
    return register


# ============================================================================ helpers
def _div(a, b):
    """a / b, NaN where b is (numerically) zero: the quantity is undefined for that joint."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.where(b > EPS, a / np.where(b > EPS, b, 1.0), np.nan)


def gap(x):
    """Largest minus smallest group value, per joint. x: [N, groups]."""
    return x.max(-1) - x.min(-1)


def ratio(x):
    """Smallest over largest group value, per joint (1 = parity)."""
    return _div(x.min(-1), x.max(-1))


def _mask(I, internal, col, value):
    d = internal.index(col)
    return np.array([str(i[d] if isinstance(i, tuple) else i) == str(value) for i in I])


def group_quantities(J, I, internal, decision, positive="1", label=None, label_positive="1"):
    """
    Masses and rates per protected group for every joint.
    J: [N, common, internal, external]. Returns a dict of arrays [N, groups]; keys ending in
    _c are [N, common, groups] (within each common value).
    """
    yhat = _mask(I, internal, decision, positive)
    theta = J.sum(1)                                            # [N, internal, external]
    mass = lambda sel: theta[:, sel, :].sum(1)                  # [N, external]
    R = {"n": mass(np.ones(len(I), bool)), "pos": mass(yhat)}
    R["sel"] = _div(R["pos"], R["n"])                           # P(decision=+ | group)
    n_c, pos_c = J.sum(2), J[:, :, yhat, :].sum(2)              # [N, common, external]
    R.update(n_c=n_c, sel_c=_div(pos_c, n_c))
    if label is not None:
        y = _mask(I, internal, label, label_positive)
        tp, fp = mass(yhat & y), mass(yhat & ~y)
        fn, tn = mass(~yhat & y), mass(~yhat & ~y)
        R.update(tp=tp, fp=fp, fn=fn, tn=tn,
                 base=_div(tp + fn, R["n"]),                    # P(label=+ | group)
                 tpr=_div(tp, tp + fn), fpr=_div(fp, fp + tn),
                 ppv=_div(tp, tp + fp), npv=_div(tn, tn + fn),
                 acc=_div(tp + tn, R["n"]), fn_fp=_div(fn, fp))
    return R


# ============================================================================ metrics
@metric("demographic_parity_gap")
def _(R):
    """Max - min over groups of P(decision=+ | group)."""
    return gap(R["sel"])


@metric("disparate_impact_ratio")
def _(R):
    """Min / max over groups of P(decision=+ | group); the 'four-fifths rule' compares it to 0.8."""
    return ratio(R["sel"])


@metric("conditional_dp_gap")
def _(R):
    """Largest demographic parity gap within any common value (e.g. within an income bin);
    groups with no mass in a common value are skipped there."""
    s = np.where(R["n_c"] > EPS, R["sel_c"], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmax(np.nanmax(s, -1) - np.nanmin(s, -1), -1)


@metric("mutual_information")
def _(R):
    """I(decision; group) in nats; 0 means the decision is independent of group."""
    p = np.stack([R["pos"], R["n"] - R["pos"]], 1)              # [N, 2, groups]
    p = p / p.sum((1, 2), keepdims=True)
    outer = p.sum(2, keepdims=True) * p.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > EPS, p * np.log(p / outer), 0.0)
    return terms.sum((1, 2))


@metric("equal_opportunity_gap", needs_label=True)
def _(R):
    """Max - min over groups of the true positive rate P(decision=+ | label=+, group)."""
    return gap(R["tpr"])


@metric("predictive_equality_gap", needs_label=True)
def _(R):
    """Max - min over groups of the false positive rate P(decision=+ | label=-, group)."""
    return gap(R["fpr"])


@metric("equalized_odds_gap", needs_label=True)
def _(R):
    """max(TPR gap, FPR gap)."""
    return np.maximum(gap(R["tpr"]), gap(R["fpr"]))


@metric("predictive_parity_gap", needs_label=True)
def _(R):
    """Max - min over groups of precision P(label=+ | decision=+, group)."""
    return gap(R["ppv"])


@metric("sufficiency_gap", needs_label=True)
def _(R):
    """max(PPV gap, NPV gap): calibration of the binary decision within groups."""
    return np.maximum(gap(R["ppv"]), gap(R["npv"]))


@metric("accuracy_parity_gap", needs_label=True)
def _(R):
    """Max - min over groups of accuracy."""
    return gap(R["acc"])


@metric("treatment_equality_gap", needs_label=True)
def _(R):
    """Max - min over groups of the ratio false negatives / false positives."""
    return gap(R["fn_fp"])


# signed differences: last protected group minus first (groups sorted as text), mainly for a
# binary protected attribute; these keep the direction of the disparity.
def diff(x):
    """Last group minus first group, per joint."""
    return x[:, -1] - x[:, 0]


@metric("demographic_parity_diff")
def _(R):
    """P(decision=+ | last group) - P(decision=+ | first group)."""
    return diff(R["sel"])


@metric("equal_opportunity_diff", needs_label=True)
def _(R):
    """TPR of the last group minus TPR of the first group."""
    return diff(R["tpr"])


@metric("predictive_equality_diff", needs_label=True)
def _(R):
    """FPR of the last group minus FPR of the first group."""
    return diff(R["fpr"])


@metric("predictive_parity_diff", needs_label=True)
def _(R):
    """PPV of the last group minus PPV of the first group."""
    return diff(R["ppv"])


# ============================================================================ randomised decisions
def _t(v):
    return v if isinstance(v, tuple) else (v,)


def decision_given_rest(p_full, internal, decision, positive, I_rest, C):
    """q[c, r] = P(decision = positive | common = c, other internal = r) from the primary data."""
    d = internal.index(decision)
    num, den = {}, {}
    for (i, c), v in p_full.items():
        t = _t(i)
        r = t[:d] + t[d + 1:]
        key = (str(r if len(r) > 1 else r[0]), str(c))
        den[key] = den.get(key, 0) + v
        if str(t[d]) == str(positive):
            num[key] = num.get(key, 0) + v
    return np.array([[num.get((str(r), str(c)), 0) / den[(str(r), str(c))]
                      if den.get((str(r), str(c)), 0) > 0 else 0.0 for r in I_rest] for c in C])


def expand_decision(J, I_rest, q, positive="1", negative="0"):
    """
    Add the decision to joints over the other internal variables, assuming
    decision independent of external given (common, other internal):
        p(decision=1, r, c, e) = q[c, r] * p(r, c, e).
    J: [N, common, rest, external]. Returns (J [N, common, 2*rest, external], I) with internal
    values (decision, *rest).
    """
    I = [(negative, *_t(r)) for r in I_rest] + [(positive, *_t(r)) for r in I_rest]
    J2 = np.concatenate([J * (1 - q)[None, :, :, None], J * q[None, :, :, None]], axis=2)
    return J2, I


# ============================================================================ bounds
def _bounds(v):
    ok = ~np.isnan(v)
    if not ok.any():
        return dict(lower=np.nan, upper=np.nan, argmin=-1, argmax=-1, undefined_share=1.0)
    idx = np.where(ok)[0]
    lo, hi = idx[v[ok].argmin()], idx[v[ok].argmax()]
    return dict(lower=v[lo], upper=v[hi], argmin=int(lo), argmax=int(hi),
                undefined_share=1 - ok.mean())


def fairness_bounds(J, I, C, E, internal, decision, positive="1", label=None,
                    label_positive="1", metrics=None):
    """
    Bounds (min and max over the grid joints) of every registered metric.
    Returns (metric_table, rate_table, values) where values[name] is the metric for every
    joint (array [N], NaN where undefined) and argmin/argmax index into J.
    """
    if label is None:
        skipped = [k for k, v in METRICS.items() if v["needs_label"]]
        if skipped and metrics is None:
            print(f"no --label: skipping {skipped}")
    R = group_quantities(J, I, internal, decision, positive, label, label_positive)
    names = metrics or [k for k, v in METRICS.items() if label is not None or not v["needs_label"]]
    values, rows = {}, []
    for name in names:
        if METRICS[name]["needs_label"] and label is None:
            raise SystemExit(f"metric '{name}' needs --label")
        v = np.asarray(METRICS[name]["fn"](R), float)
        values[name] = v
        rows.append(dict(metric=name, **_bounds(v), description=METRICS[name]["description"]))
    rates = [q for q in ("sel", "base", "tpr", "fpr", "ppv", "npv", "acc") if q in R]
    rate_rows = [dict(rate=q, group=str(e), **_bounds(R[q][:, k]))
                 for q in rates for k, e in enumerate(E)]
    return pd.DataFrame(rows), pd.DataFrame(rate_rows), values


# ============================================================================ command line
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("primary")
    ap.add_argument("auxiliary")
    ap.add_argument("--internal", nargs="+")
    ap.add_argument("--common", nargs="+")
    ap.add_argument("--external", nargs="+")
    ap.add_argument("--weight-primary")
    ap.add_argument("--weight-aux")
    ap.add_argument("--decision", required=True)
    ap.add_argument("--positive", default="1")
    ap.add_argument("--label")
    ap.add_argument("--label-positive", default="1")
    ap.add_argument("--metrics", nargs="+", choices=list(METRICS), help="default: all applicable")
    ap.add_argument("--randomised", action="store_true",
                    help="the decision depends only on the common and other internal variables "
                         "(plus independent noise), e.g. a synthetic predictor without "
                         "--feature-weights: impose decision independent of external given those")
    ap.add_argument("--num", type=int, default=5)
    ap.add_argument("--method", choices=["sweep", "sequential"], default="sequential")
    ap.add_argument("--max-categories", type=int, default=20)
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--max-joints", type=int, default=5_000_000)
    ap.add_argument("--out", default="fairness_results")
    a = ap.parse_args()

    primary, aux = pd.read_csv(a.primary), pd.read_csv(a.auxiliary)
    p_primary, p_aux, internal, common, external = from_datasets(
        primary, aux, a.internal, a.common, a.external, a.weight_primary, a.weight_aux,
        a.max_categories, a.bins)
    for col in [a.decision] + ([a.label] if a.label else []):
        if col not in internal:
            raise SystemExit(f"'{col}' is not an internal variable {internal}")
    if a.randomised:
        rest = [c for c in internal if c != a.decision]
        if not rest:
            primary["_const"], rest = 0, ["_const"]
        kw = dict(common=common, external=external, weight_primary=a.weight_primary,
                  weight_aux=a.weight_aux, max_categories=a.max_categories, bins=a.bins)
        p_full = from_datasets(primary, aux, [a.decision] + rest, **kw)[0]
        p_rest = from_datasets(primary, aux, rest, **kw)[0]
        J, (I_rest, C, E) = solve_joint_distribution(p_rest, p_aux, a.num, a.method,
                                                     max_joints=a.max_joints, as_dicts=False)
        q = decision_given_rest(p_full, [a.decision] + rest, a.decision, a.positive, I_rest, C)
        J, I = expand_decision(J, I_rest, q, a.positive)
        internal = [a.decision] + rest
    else:
        J, (I, C, E) = solve_joint_distribution(p_primary, p_aux, a.num, a.method,
                                                max_joints=a.max_joints, as_dicts=False)
    mt, rt, values = fairness_bounds(J, I, C, E, internal, a.decision, a.positive,
                                     a.label, a.label_positive, a.metrics)

    os.makedirs(a.out, exist_ok=True)
    mt.to_csv(os.path.join(a.out, "metric_bounds.csv"), index=False)
    rt.to_csv(os.path.join(a.out, "group_rate_bounds.csv"), index=False)
    if len(J) <= 200_000:                          # per-joint values, e.g. for trade-off plots
        pd.DataFrame(values).to_csv(os.path.join(a.out, "metric_values.csv"), index_label="joint")
    np.savez_compressed(os.path.join(a.out, "joints.npz"), joints=J,
                        internal_values=np.array(I, dtype=object),
                        common_values=np.array(C, dtype=object),
                        external_values=np.array(E, dtype=object))
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(mt.drop(columns="description").round(4).to_string(index=False))
        print()
        print(rt.round(4).to_string(index=False))
    print(f"\nsaved to {a.out}/")


if __name__ == "__main__":
    main()