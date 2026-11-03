"""
Experiments comparing grid enumeration of joints (grid_joint.py) with the Kallus, Mao & Zhou
bounds (kallus_bounds.py), on identical marginals.

  exp1  replication: DP, TPR (= EO), FPR (= PE), PPV (= PP) and NPV disparity intervals vs the
        Kallus closed forms. Data: synthetic problems, HMDA (DP only, no label) or warfarin.
  exp2  grid convergence: endpoint error vs Kallus as num grows, for both grid methods.
  exp3  additional metrics from the same enumerated joints: F1 and FDR parity (checked against
        the bounds implied by the Kallus closed forms), equalized odds and sufficiency (vs the
        looser combination of separate Kallus intervals and an LP reference), mutual
        information, and the joint (TPRD, FPRD) region vs the Kallus rectangle.
  exp4  computation: time and peak memory vs free parameters, num and number of metrics.

Usage (from the repo root):
  python methods/experiments.py exp1
  python methods/experiments.py exp1 --data hmda --kallus-repo ~/kallus
  python methods/experiments.py exp1 --data warfarin --kallus-repo ~/kallus --proxy Amiodarone.
  python methods/experiments.py exp2
  python methods/experiments.py exp3
  python methods/experiments.py exp4
  python methods/experiments.py all --quick

Disparities are always rate(group 1) - rate(group 0), where the groups are the external values
in grid_joint's (string-sorted) order; synthetic and real-data groups are named so that the
reference group sorts first. Results go to --out (CSV, and PNG if matplotlib is installed).
"""
import argparse
import contextlib
import io
import os
import sys
import time
import tracemalloc

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid_joint import solve_joint_distribution            # noqa: E402
from kallus_bounds import CouplingLP, closed_form_bounds, lp_bounds   # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:                                         # plots are optional
    plt = None


# =========================================================================== problems
def synthetic_problem(Z, internal="dy", seed=0, proxy_strength=1.0):
    """
    A population p(common, internal, external) with binary external group.
    internal = 'dy' -> internal values (decision, label); 'd' -> decision only.
    proxy_strength scales how much P(group | common, internal) varies across common values.
    Returns p_primary {(internal, common): p}, p_aux {(common, external): p}, truth joint.
    """
    rng = np.random.default_rng(seed)
    I = [(d, y) for d in (0, 1) for y in (0, 1)] if internal == "dy" else [0, 1]
    C = [f"z{k:02d}" for k in range(Z)]
    E = ["g0", "g1"]
    pc = rng.dirichlet(np.ones(Z) * 3)
    pi_c = rng.dirichlet(np.ones(len(I)) * 2, Z)                         # p(internal | common)
    logit = rng.normal(0, 1.5 * proxy_strength, (Z, 1)) + rng.normal(0, 1, (1, len(I)))
    pe = 1 / (1 + np.exp(-logit))                                       # P(g1 | common, internal)
    joint = pc[:, None, None] * pi_c[:, :, None] * np.stack([1 - pe, pe], -1)   # [C, I, E]
    p_primary = {(i, c): joint[k, a].sum() for k, c in enumerate(C) for a, i in enumerate(I)}
    p_aux = {(c, e): joint[k, :, b].sum() for k, c in enumerate(C) for b, e in enumerate(E)}
    return p_primary, p_aux, joint


def hmda_problem(repo, race="White", bins=7):
    """HMDA (paper's 1% sample), income proxy in quantile bins, race vs rest. Decision only."""
    df = pd.read_csv(os.path.join(repo, "HMDA", "small_proxy_income.csv"))
    edges = np.unique(np.quantile(df.applicant_income_000s, np.linspace(0, 1, bins + 1)))
    k = pd.cut(df.applicant_income_000s, edges, include_lowest=True, labels=False)
    df = df.assign(c=[f"b{int(v):02d}" for v in k],
                   g=np.where(df.race == race, f"1_{race}", "0_rest"))
    p_primary = df.groupby(["outcome", "c"]).size().to_dict()
    p_aux = df.groupby(["c", "g"]).size().to_dict()
    return p_primary, p_aux


def warfarin_problem(repo, race="White.", proxy="Amiodarone."):
    """Warfarin, one medication column as proxy, race vs rest. Internal = (Yhat, Y)."""
    df = pd.read_csv(os.path.join(repo, "Warfrin", "medicine_as_proxy.csv"))
    df = df.assign(c=df[proxy].fillna("NA").astype(str),
                   g=np.where(df.race == race, f"1_{race}", "0_rest"),
                   i=list(zip(df.Yhat.astype(int), df.Y.astype(int))))
    p_primary = df.groupby(["i", "c"]).size().to_dict()
    p_aux = df.groupby(["c", "g"]).size().to_dict()
    return p_primary, p_aux


# =========================================================================== grid + inputs
def run_grid(p_primary, p_aux, num, method="sweep", max_joints=5_000_000):
    """
    grid_joint enumeration with its printing silenced; None if it refuses (too large).
    method 'sequential-all' runs 'sequential' once per choice of which internal value is
    filled last (grid_joint derives that one from the others): the internal values are
    relabelled so the chosen one sorts last, and the joints are mapped back and pooled.
    """
    if method == "sequential-all":
        I0 = sorted({k[0] for k in p_primary}, key=str)
        Js = []
        for last in I0:
            lab = {i: (f"1|{i}" if i == last else f"0|{i}") for i in I0}
            g = run_grid({(lab[i], c): v for (i, c), v in p_primary.items()}, p_aux, num,
                         "sequential", max_joints)
            if g is None:
                return None
            J, I, C, E = g
            Js.append(J[:, :, [I.index(lab[i]) for i in I0], :])
        J = np.concatenate(Js)
        return (J, I0, C, E) if len(J) <= max_joints else None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            J, (I, C, E) = solve_joint_distribution(p_primary, p_aux, num=num, method=method,
                                                    max_joints=max_joints, as_dicts=False)
        return J, I, C, E
    except SystemExit:
        return None


def kallus_inputs(p_primary, p_aux, I, C, E):
    """r [C, I] and m [C, E] exactly as grid_joint normalises them (P(common) from aux)."""
    Paux = np.array([[p_aux.get((c, e), 0) for e in E] for c in C], float)
    Pprim = np.array([[p_primary.get((i, c), 0) for i in I] for c in C], float)
    Paux = Paux / Paux.sum()
    r = Pprim / Pprim.sum(1, keepdims=True) * Paux.sum(1, keepdims=True)
    return r, Paux


def dy_masks(I):
    t = lambda i: i if isinstance(i, tuple) else (i,)
    d = np.array([bool(int(t(i)[0])) for i in I])
    y = np.array([bool(int(t(i)[1])) for i in I]) if isinstance(I[0], tuple) else None
    return d, y


# =========================================================================== metrics
def rate_specs(d, y):
    """name -> (target, cond) masks over internal values; rate_g = P(target | cond, A=g)."""
    specs = {"DP": (d, np.ones_like(d))}
    if y is not None:
        specs.update({"TPR": (d & y, y), "FPR": (d & ~y, ~y), "PPV": (d & y, d),
                      "NPV": (~d & ~y, ~d)})
    return specs


def group_rates(theta, target, cond):
    """theta [N, I, E] -> rates [N, E]."""
    num_ = theta[:, target, :].sum(1)
    den = theta[:, cond, :].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num_ / den


def mutual_information(theta, d):
    """I(decision; A) for each joint, theta [N, I, E]."""
    pde = np.stack([theta[:, ~d, :].sum(1), theta[:, d, :].sum(1)], 1)   # [N, 2, E]
    pd_ = pde.sum(2, keepdims=True); pe = pde.sum(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = pde * np.log(pde / (pd_ * pe))
    return np.nansum(t, (1, 2))


def metric_values(theta, d, y):
    """All metrics for every joint: dict name -> array [N] (disparities are g1 - g0)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return _metric_values(theta, d, y)


def _metric_values(theta, d, y):
    out = {}
    rates = {k: group_rates(theta, *v) for k, v in rate_specs(d, y).items()}
    for k, r_ in rates.items():
        out[f"{k}_diff"] = r_[:, 1] - r_[:, 0]
    out["DI_ratio"] = rates["DP"][:, 1] / rates["DP"][:, 0]
    out["MI"] = mutual_information(theta, d)
    if y is not None:
        tp, fp, fn = (theta[:, d & y, :].sum(1), theta[:, d & ~y, :].sum(1),
                      theta[:, ~d & y, :].sum(1))
        f1 = 2 * tp / (2 * tp + fp + fn)
        out["F1_diff"] = f1[:, 1] - f1[:, 0]
        out["FDR_diff"] = (1 - rates["PPV"][:, 1]) - (1 - rates["PPV"][:, 0])
        out["EOdds"] = np.maximum(abs(out["TPR_diff"]), abs(out["FPR_diff"]))
        out["Sufficiency"] = np.maximum(abs(out["PPV_diff"]), abs(out["NPV_diff"]))
    return out


def grid_bounds(J, I):
    d, y = dy_masks(I)
    vals = metric_values(J.sum(1), d, y)
    nan_safe = lambda f, v: f(v) if np.isfinite(v).any() or np.isinf(v).any() else np.nan
    return {k: (nan_safe(np.nanmin, v), nan_safe(np.nanmax, v)) for k, v in vals.items()}, vals


# =========================================================================== Kallus references
def extremal_rates(r, m, target, cond):
    """Group rates at the two extremal couplings of Props 7-8:
    (mu1_min, mu0_max) at the lower coupling and (mu1_max, mu0_min) at the upper one."""
    P, ma = r.sum(1), m[:, 1]
    h, g = r[:, target].sum(1), r[:, cond & ~target].sum(1)
    H, G = h.sum(), g.sum()
    lo = lambda c: np.maximum(0, c + ma - P).sum()
    up = lambda c: np.minimum(c, ma).sum()
    def rates(T, S):
        return T / (T + S), (H - T) / (H + G - T - S)
    return rates(lo(h), up(g)) + rates(up(h), lo(g))


def kallus_bounds_all(r, m, I):
    """Kallus closed-form bounds, plus the bounds they imply for F1, FDR, DI ratio and the
    (valid but possibly loose) combination of separate intervals for EOdds / Sufficiency."""
    d, y = dy_masks(I)
    out = {f"{k}_diff": closed_form_bounds(r, m, *v, harmonize=False)
           for k, v in rate_specs(d, y).items()}
    out["DI_ratio"] = closed_form_bounds(r, m, d, np.ones_like(d), ratio=True, harmonize=False)
    if y is not None:
        a_lo, b_hi, a_hi, b_lo = extremal_rates(r, m, d & y, d | y)    # J = TP/(TP+FP+FN)
        f = lambda j: 2 * j / (1 + j)                                    # F1 is increasing in J
        out["F1_diff"] = (f(a_lo) - f(b_hi), f(a_hi) - f(b_lo))
        lo, hi = out["PPV_diff"]
        out["FDR_diff"] = (-hi, -lo)                                     # FDR = 1 - PPV
        gap = lambda b: (0.0 if b[0] <= 0 <= b[1] else min(abs(b[0]), abs(b[1])),
                         max(abs(b[0]), abs(b[1])))
        for name, (a, b) in {"EOdds": ("TPR_diff", "FPR_diff"),
                             "Sufficiency": ("PPV_diff", "NPV_diff")}.items():
            ga, gb = gap(out[a]), gap(out[b])
            out[name] = (max(ga[0], gb[0]), max(ga[1], gb[1]))         # separate intervals
    return out


def max_gap_lp_lower(r, m, specs, n_grid=41, refine=True):
    """
    Lower bound of max(|diff_1|, |diff_2|) over ONE coupling (exact up to a 2-d grid over the
    group-1 condition masses, then pattern-search refinement). specs = [(target, cond)] * 2.
    """
    lp = CouplingLP(r, m, harmonize=False)
    TX = [lp.masks(t, c) for t, c in specs]
    nv = lp.nvar
    rng_ = [(-lp.solve(-X[1])[0], lp.solve(X[1])[0]) for _, X in TX]
    Ks = [lp.solve(X.sum(0))[0] for _, X in TX]

    def value(xs):
        rows, b_ub, eq_rows, b_eq = [], [], [], []
        for (T, X), x1, K in zip(TX, xs, Ks):
            x0 = K - x1
            if x1 <= 1e-15 or x0 <= 1e-15:
                return np.inf
            diff = T[1] / x1 - T[0] / x0                   # linear in q once masses are fixed
            for s in (1, -1):
                rows.append(np.append(s * diff, -1.0)); b_ub.append(0.0)
            eq_rows += [np.append(X[0], 0.0), np.append(X[1], 0.0)]; b_eq += [x0, x1]
        A_eq = sparse.vstack([sparse.hstack([lp.A_eq, sparse.csr_matrix((lp.A_eq.shape[0], 1))]),
                              sparse.csr_matrix(np.array(eq_rows))])
        res = linprog(np.append(np.zeros(nv), 1.0), A_ub=np.array(rows), b_ub=b_ub,
                      A_eq=A_eq, b_eq=np.concatenate([lp.b_eq, b_eq]), bounds=(0, None),
                      method="highs")
        return res.fun if res.status == 0 else np.inf

    g1, g2 = [np.linspace(a, b, n_grid) for a, b in rng_]
    best, arg = np.inf, None
    for a in g1:
        for b in g2:
            v = value((a, b))
            if v < best:
                best, arg = v, np.array([a, b])
    if refine and arg is not None:
        step = np.array([(b - a) / max(n_grid - 1, 1) for a, b in rng_])
        for _ in range(80):
            moved = False
            for j in range(2):
                for s in (1, -1):
                    cand = arg.copy(); cand[j] += s * step[j]
                    v = value(cand)
                    if v < best - 1e-15:
                        best, arg, moved = v, cand, True
            if not moved:
                step /= 2
                if np.all(step < 1e-12):
                    break
    return best


def random_feasible_joints(r, m, n=2000, seed=0, iters=300):
    """Feasible joints [n, C, I, E] by iterative proportional fitting from heavy-tailed random
    starts (spread towards the boundary). Used to check whether grid bounds are attained."""
    rng = np.random.default_rng(seed)
    Q = rng.exponential(size=(n,) + r.shape + (m.shape[1],)) ** 3 + 1e-12
    for _ in range(iters):
        Q *= (r / Q.sum(3).clip(1e-300))[..., None]
        Q *= (m / Q.sum(2).clip(1e-300))[:, :, None, :]
    return Q


# =========================================================================== experiments
def _fmt(df):
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        return df.to_string(index=False, float_format=lambda v: f"{v:.3g}")


def exp1(a):
    rows = []
    if a.data == "synthetic":
        cases = []
        for s in range(a.reps):          # decision-only problems: DP is exact for any num >= 2
            cases.append((f"synth-d-{s}", *synthetic_problem(a.Z_d, "d", seed=s)[:2]))
        for s in range(a.reps):          # (decision, label) problems: all rate metrics
            cases.append((f"synth-dy-{s}", *synthetic_problem(a.Z, "dy", seed=1000 + s)[:2]))
    elif a.data == "hmda":
        cases = [(f"hmda-{race}", *hmda_problem(a.kallus_repo, race, a.bins))
                 for race in ["White", "Black", "API"]]
    else:
        cases = [(f"warfarin-{race}", *warfarin_problem(a.kallus_repo, race, a.proxy))
                 for race in ["White.", "Black.", "Asian."]]
    for name, pp, pa in cases:
        for method in a.methods:
            g = run_grid(pp, pa, a.num, method, a.max_joints)
            if g is None:
                print(f"{name}: too many joints at num={a.num}; skipped"); continue
            J, I, C, E = g
            r, m = kallus_inputs(pp, pa, I, C, E)
            gb, _ = grid_bounds(J, I)
            kb = kallus_bounds_all(r, m, I)
            d, y = dy_masks(I)
            for k in rate_specs(d, y):
                (gl, gu), (kl, ku) = gb[f"{k}_diff"], kb[f"{k}_diff"]
                rows.append(dict(case=name, internal="decision" if y is None else "decision,label",
                                 method=method, metric=k, n_joints=len(J),
                                 grid_lower=gl, grid_upper=gu, kallus_lower=kl, kallus_upper=ku,
                                 lower_gap=gl - kl, upper_gap=ku - gu,
                                 inside=(gl >= kl - 1e-9) and (gu <= ku + 1e-9)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, f"exp1_{a.data}.csv"), index=False)
    summ = (df.groupby(["internal", "method", "metric"])
              .agg(cases=("case", "nunique"), max_lower_gap=("lower_gap", "max"),
                   max_upper_gap=("upper_gap", "max"), all_inside=("inside", "all"))
              .reset_index())
    print(f"\nExperiment 1 ({a.data}, num={a.num}): gap = how far the grid interval sits inside "
          "Kallus (0 = identical; negative would mean the grid is OUTSIDE = invalid)")
    print(_fmt(summ))
    if a.data != "synthetic":
        print(_fmt(df[["case", "method", "metric", "grid_lower", "grid_upper",
                       "kallus_lower", "kallus_upper"]]))
    return df


def exp2(a):
    rows = []
    for Z in a.Zs:
        for s in range(a.reps):
            pp, pa, _ = synthetic_problem(Z, "dy", seed=2000 + 10 * Z + s)
            for method in a.methods:
                for num in a.nums:
                    est = float(num) ** (3 * Z) * (4 if method == "sequential-all" else 1)
                    if est > a.budget:                 # checked before enumerating (memory)
                        continue
                    t0 = time.perf_counter()
                    g = run_grid(pp, pa, num, method, a.max_joints)
                    if g is None:
                        continue
                    J, I, C, E = g
                    gb, _ = grid_bounds(J, I)
                    dt = time.perf_counter() - t0
                    kb = kallus_bounds_all(*kallus_inputs(pp, pa, I, C, E), I)
                    for k in ["DP", "TPR", "FPR", "PPV", "NPV"]:
                        (gl, gu), (kl, ku) = gb[f"{k}_diff"], kb[f"{k}_diff"]
                        rows.append(dict(Z=Z, free_params=3 * Z, problem=s, method=method,
                                         num=num, n_joints=len(J), seconds=dt, metric=k,
                                         lower_gap=gl - kl, upper_gap=ku - gu,
                                         max_gap=max(gl - kl, ku - gu)))
            print(f"exp2: Z={Z} problem {s} done")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "exp2_convergence.csv"), index=False)
    summ = df.groupby(["Z", "method", "num", "metric"]).max_gap.max().unstack("metric")
    print("\nExperiment 2: worst endpoint gap to Kallus (max over problems), by num")
    print(summ.to_string(float_format=lambda v: f"{v:.2e}"))
    if plt is not None:
        metrics = ["DP", "TPR", "FPR", "PPV", "NPV"]
        fig, axes = plt.subplots(len(a.Zs), len(a.methods), figsize=(5 * len(a.methods),
                                 3.6 * len(a.Zs)), squeeze=False)
        for i, Z in enumerate(a.Zs):
            for j, method in enumerate(a.methods):
                ax = axes[i, j]
                sub = df[(df.Z == Z) & (df.method == method)]
                for k in metrics:
                    s_ = sub[sub.metric == k].groupby("num").max_gap.max()
                    ax.plot(s_.index, np.maximum(s_.values, 1e-16), "o-", label=k)
                ax.set_yscale("log"); ax.set_xlabel("num (grid values per free parameter)")
                ax.set_ylabel("worst endpoint gap to Kallus")
                ax.set_title(f"{method}, Z={Z} ({3 * Z} free parameters)")
                ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "exp2_convergence.png"), dpi=150)
    return df


def exp3(a):
    rows, region = [], None
    for s in range(a.reps):
        pp, pa, _ = synthetic_problem(a.Z, "dy", seed=3000 + s, proxy_strength=a.proxy_strength)
        g = run_grid(pp, pa, a.num, a.method3, a.max_joints)
        if g is None:
            print("too many joints; lower --num or --Z"); return None
        J, I, C, E = g
        r, m = kallus_inputs(pp, pa, I, C, E)
        d, y = dy_masks(I)
        gb, vals = grid_bounds(J, I)
        kb = kallus_bounds_all(r, m, I)
        # LP references for the two metrics that need one coupling for two rates
        ref = {"EOdds": max_gap_lp_lower(r, m, [(d & y, y), (d & ~y, ~y)], a.lp_grid),
               "Sufficiency": max_gap_lp_lower(r, m, [(d & y, d), (~d & ~y, ~d)], a.lp_grid)}
        # random feasible joints: does anything escape the grid interval?
        Q = random_feasible_joints(r, m, a.n_random, seed=s)
        qv = metric_values(Q.sum(1), d, y)
        # MI at the conditional-independence coupling A _||_ internal | common (feasible)
        ci = (r[:, :, None] * (m / m.sum(1, keepdims=True))[:, None, :])[None]
        mi_ci = mutual_information(ci.sum(1), d)[0]
        for k in ["F1_diff", "FDR_diff", "DI_ratio", "EOdds", "Sufficiency", "MI"]:
            gl, gu = gb[k]
            kl, ku = kb.get(k, (np.nan, np.nan))
            rv = qv[k]
            rows.append(dict(problem=s, metric=k, grid_lower=gl, grid_upper=gu,
                             kallus_lower=kl, kallus_upper=ku,
                             lp_reference_lower=ref.get(k, np.nan),
                             mi_at_cond_indep=mi_ci if k == "MI" else np.nan,
                             random_min=np.nanmin(rv), random_max=np.nanmax(rv),
                             random_escape=np.mean((rv < gl - 1e-9) | (rv > gu + 1e-9))))
        if s == 0:
            region = (vals["TPR_diff"], vals["FPR_diff"], kb["TPR_diff"], kb["FPR_diff"],
                      qv["TPR_diff"], qv["FPR_diff"])
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "exp3_additional_metrics.csv"), index=False)
    print(f"\nExperiment 3 (Z={a.Z}, num={a.num}, {a.method3}). kallus_* = bounds implied by the closed "
          "forms (F1, FDR, DI: exact corollaries; EOdds/Sufficiency: separate intervals combined,"
          " valid but possibly loose). lp_reference_lower = one-coupling LP. random_escape = share "
          "of random feasible joints outside the grid interval (> 0 means the grid misses part "
          "of the identified set).")
    print(_fmt(df))
    if plt is not None and region is not None:
        tx, fx, (tl, tu), (fl, fu), rtx, rfx = region
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.scatter(rtx, rfx, s=2, c="0.7", label="random feasible joints")
        ax.scatter(tx, fx, s=2, c="C0", label="grid joints")
        ax.add_patch(plt.Rectangle((tl, fl), tu - tl, fu - fl, fill=False, ec="C3", lw=1.5,
                                   label="Kallus intervals (rectangle)"))
        ax.set_xlabel("TPR disparity"); ax.set_ylabel("FPR disparity")
        ax.set_title("Joint identified set of (TPRD, FPRD)"); ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "exp3_region.png"), dpi=150)
    return df


def _measure(fn, repeats):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter(); out = fn(); times.append(time.perf_counter() - t0)
    tracemalloc.start(); fn(); peak = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
    return float(np.median(times)), peak / 2 ** 20, out


def exp4(a):
    rows, kallus_cache = [], {}
    configs = [("d", Z, num) for Z in a.Zs_d for num in a.nums4] + \
              [("dy", Z, num) for Z in a.Zs for num in a.nums4]
    for (internal, Z, num), method in [(c, mt) for c in configs for mt in a.methods]:
        F = Z * (3 if internal == "dy" else 1)
        if float(num) ** F > a.max_joints:
            continue
        pp, pa, _ = synthetic_problem(Z, internal, seed=4000 + Z)
        t_enum, mem_enum, g = _measure(lambda: run_grid(pp, pa, num, method, a.max_joints),
                                       a.repeats)
        if g is None:
            continue
        J, I, C, E = g
        d, y = dy_masks(I)
        theta = J.sum(1)
        t_eval, mem_eval, vals = _measure(lambda: metric_values(theta, d, y), a.repeats)
        if (internal, Z) not in kallus_cache:          # Kallus cost does not depend on num
            r, m = kallus_inputs(pp, pa, I, C, E)
            specs = rate_specs(d, y)
            kallus_cache[(internal, Z)] = (
                _measure(lambda: kallus_bounds_all(r, m, I), a.repeats)[:2]
                + _measure(lambda: [lp_bounds(r, m, *v, harmonize=False)
                                    for v in specs.values()], 1)[:2])
        t_cf, mem_cf, t_lp, mem_lp = kallus_cache[(internal, Z)]
        rows.append(dict(method=method, internal=internal, Z=Z, free_params=F, num=num,
                         n_joints=len(J),
                         n_metrics=len(vals), grid_enum_s=t_enum, grid_enum_MB=mem_enum,
                         grid_eval_all_metrics_s=t_eval, grid_eval_MB=mem_eval,
                         kallus_closed_form_s=t_cf, kallus_closed_form_MB=mem_cf,
                         kallus_lp_rate_metrics_s=t_lp, kallus_lp_MB=mem_lp))
        print(f"exp4: {method} internal={internal} Z={Z} num={num}: {len(J):,} joints, "
              f"enum {t_enum:.3f}s / {mem_enum:.0f} MB")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "exp4_computation.csv"), index=False)
    print("\nExperiment 4: median seconds and peak MB (tracemalloc)")
    print(_fmt(df))
    if plt is not None and len(df):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for (method, internal, num), sub in df.groupby(["method", "internal", "num"]):
            sub = sub.sort_values("free_params")
            lab = f"{method} ({internal}, num={num})"
            axes[0].plot(sub.free_params, sub.grid_enum_s + sub.grid_eval_all_metrics_s, "o-",
                         label=lab, ms=3)
            axes[1].plot(sub.free_params, sub.grid_enum_MB, "o-", label=lab, ms=3)
        for internal, sub in df.groupby("internal"):
            sub = sub.groupby("free_params").mean(numeric_only=True)
            axes[0].plot(sub.index, sub.kallus_closed_form_s, "k--", lw=1)
            axes[0].plot(sub.index, sub.kallus_lp_rate_metrics_s, "k:", lw=1)
        axes[0].plot([], [], "k--", label="Kallus closed form (all metrics)")
        axes[0].plot([], [], "k:", label="Kallus LP (rate metrics)")
        for ax, lab in zip(axes, ["seconds", "peak MB (enumeration)"]):
            ax.set_yscale("log"); ax.set_xlabel("free parameters"); ax.set_ylabel(lab)
            ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "exp4_computation.png"), dpi=150)
    return df


# =========================================================================== command line
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=["exp1", "exp2", "exp3", "exp4", "all"])
    ap.add_argument("--out", default="results/experiments")
    ap.add_argument("--quick", action="store_true", help="small settings for a smoke test")
    ap.add_argument("--data", choices=["synthetic", "hmda", "warfarin"], default="synthetic")
    ap.add_argument("--kallus-repo", default=os.path.expanduser("~/kallus"))
    ap.add_argument("--bins", type=int, default=7, help="HMDA income bins")
    ap.add_argument("--proxy", default="Amiodarone.", help="warfarin proxy column")
    ap.add_argument("--num", type=int, default=None)
    ap.add_argument("--Z", type=int, default=2, help="common values, (decision, label) problems")
    ap.add_argument("--Z-d", dest="Z_d", type=int, default=6, help="common values, decision-only")
    ap.add_argument("--reps", type=int, default=None, help="random problems")
    ap.add_argument("--methods", nargs="+", default=["sweep", "sequential", "sequential-all"],
                    choices=["sweep", "sequential", "sequential-all"])
    ap.add_argument("--proxy-strength", type=float, default=1.0,
                    help="synthetic problems: how strongly P(group | common) varies (exp3)")
    ap.add_argument("--method3", default="sequential-all", help="grid method for exp3")
    ap.add_argument("--max-joints", type=int, default=5_000_000)
    ap.add_argument("--budget", type=float, default=1e6,
                    help="exp2: skip settings whose grid could exceed this many joints")
    ap.add_argument("--lp-grid", type=int, default=41)
    ap.add_argument("--n-random", type=int, default=2000)
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    q = a.quick
    reps = a.reps
    if a.experiment in ("exp1", "all"):
        a.reps = reps or (3 if q else 20)
        a.num = a.num or (4 if a.data == "warfarin" else 5)
        exp1(a)
    if a.experiment in ("exp2", "all"):
        a.reps = reps or (2 if q else 5)
        a.Zs = [1] if q else [1, 2]
        a.nums = [2, 3, 5] if q else [2, 3, 4, 5, 7, 9, 13, 17, 25]
        exp2(a)
    if a.experiment in ("exp3", "all"):
        a.reps = reps or (1 if q else 5)
        a.num = 5 if q else (a.num or 7)
        a.lp_grid = 21 if q else a.lp_grid
        exp3(a)
    if a.experiment in ("exp4", "all"):
        a.Zs_d = [1, 2, 4] if q else [1, 2, 4, 6, 8]
        a.Zs = [1, 2] if q else [1, 2, 3]
        a.nums4 = [2, 3] if q else [2, 3, 5, 7]
        a.repeats = 1 if q else a.repeats
        exp4(a)


if __name__ == "__main__":
    main()