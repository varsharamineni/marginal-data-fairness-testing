"""Validation of grid exploration of the feasible joints, feasible_joints(method="grid").

RQ1 validity     do the grid ranges recover the exact bounds?  Grid vs exact optimisation
                 (linear programming) vs the Kallus closed form on small instances.
RQ2 resolution   error of the grid bounds as the grid step h shrinks.
RQ3 runtime      grid size and runtime against the number of free parameters d = 1..6.
RQ4 adaptive     full grid vs adaptive (zoom) grid vs random / hit-and-run sampling:
                 error against the number of evaluated joints.
RQ5 overlap      interval width when a fraction of the primary rows is linked to A.

Free parameters. For each Z level z the joint is fixed by q(o) = P(A=a, O=o | Z=z) over the
m outcome cells o, with 0 <= q(o) <= P(O=o | z) and sum_o q(o) = P(A=a | z): m - 1 free
parameters per level, d = K (m - 1) in total (O = Yhat: m = 2, d = K; O = (Yhat, Y): m = 4).
Resolution. The grid puts num points on each free parameter's feasible range
(np.linspace in methods._grid_splits), so h = 1 / (num - 1) is a fraction of that range.

Exact bounds. All three disparities have the form
    f = N_a / D_a - (N - N_a) / (D - D_a),   N_a = P(A=a, num cells), D_a = P(A=a, den cells),
where N, D are fixed by the primary data and N_a, D_a are linear in q. For DD, D_a = P(A=a)
is fixed, so f is linear and its range is two LPs. For TPRD/TNRD, f is increasing in N_a at
fixed D_a = t, so max f = max_t f(max N_a | D_a = t): an LP for each t, with t swept on a
fine grid and refined by bounded 1-D search. min |f| follows from the range (f is continuous
on a convex set, so its image is an interval).

The grid, random and hit-and-run joints come from methods.feasible_joints unchanged; the
adaptive grid and the LP use the same parameterisation and live only in this script.

Usage:  python scripts/grid_validation.py {rq1,rq2,rq3,rq4,rq5,all} [--quick] [--jobs 32]
                                          [--out-dir outputs/grid_validation]
"""
import argparse
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # one thread per worker
    os.environ.setdefault(_v, "1")

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize_scalar

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fairness import data, methods  # noqa: E402
from metric_comparison import from_config  # noqa: E402

A = ["a", "b"]
MAX_BUILD = 300_000          # grids larger than this are counted (and timed by extrapolation) only
N_T = 101                    # t values in the TPRD/TNRD sweep before refinement
LP_TOL = {"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10}

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#8a63d2"]
INK, MUTED, GRID_C = "#1f1f1e", "#6b6a63", "#e4e3dc"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID_C, "grid.linewidth": 0.6, "legend.frameon": False})


# ------------------------------------------------------------------ data
def synthetic_instance(K, seed, n=20_000):
    """Synthetic data with randomised parameters, so instances differ in P(Z), P(A|Z) and
    in how Y and Yhat depend on A. Returns (primary without A, auxiliary, true A, a_levels)."""
    rng = np.random.default_rng(1000 + seed)
    cfg = dict(seed=seed, n_primary=n, n_auxiliary=n, n_z=K,
               p_z=list(rng.dirichlet(np.full(K, 3.0))), p_a_given_z=list(rng.uniform(0.1, 0.9, K)),
               y_intercept=rng.normal(0, 0.5), y_coef_a=rng.uniform(-1, 1), y_coef_z=1.0,
               x1_coef_y=1.5, x1_coef_a=rng.uniform(-1, 1), x2_coef_y=1.0)
    cfg["p_z"][-1] = 1.0 - sum(cfg["p_z"][:-1])
    prim, aux = data.generate_synthetic(cfg)
    prim["yhat"] = (prim["x1"] > 0.75).astype(int)
    return prim.drop(columns="a"), aux, prim["a"].to_numpy(), A


def german():
    return from_config("configs/german.yaml")


# ------------------------------------------------------------------ disparities from aggregated joints
def masks(cells, metric):
    """(numerator cells, denominator cells) of the group rate behind `metric`."""
    yhat = np.array([c[0] for c in cells])
    if metric == "DD":
        return yhat == 1, np.ones(len(cells), bool)
    k, c = methods._CELL[metric]
    y = np.array([o[1] for o in cells])
    return (yhat == k) & (y == c), y == c


def disparity(G, cells, metric):
    """Disparity from G[..., g, o] = P(A = a_levels[g], O = o) (summed over Z)."""
    num, den = masks(cells, metric)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = G[..., num].sum(-1) / G[..., den].sum(-1)
    return r[..., 0] - r[..., 1]


def metrics_for(outcome):
    return list(methods.METRICS) if "y" in outcome else ["DD"]


# ------------------------------------------------------------------ the feasible set of one instance
class Problem:
    """Free parameters q[z, o] = P(A=a, O=o | Z=z), 0 <= q <= p_o, sum_o q = p_a, and
    G(q) = w * sum_z p_z [q, p_o - q] + F, where F (2 x m) is mass with known A (RQ5:
    linked rows, w = 1 - overlap). Without linked rows w = 1 and F = 0."""

    def __init__(self, prim, aux, a_levels, outcome, w=1.0, F=None):
        self.prim, self.aux, self.a_levels, self.outcome = prim, aux, a_levels, list(outcome)
        self.z_levels, self.cells, self.p_z, self.p_o, self.p_a = methods._joint_marginals(
            prim, aux, a_levels, self.outcome)
        self.K, self.m = self.p_o.shape
        self.d = self.K * (self.m - 1)
        self.w = w
        self.F = np.zeros((2, self.m)) if F is None else F

    # joints -> G
    def G_from_q(self, Q):
        pz = self.p_z[:, None]
        return self.w * np.stack([(pz * Q).sum(-2), (pz * (self.p_o - Q)).sum(-2)], axis=-2) + self.F

    def G_from_joints(self, joints):
        p = np.stack([j["p"].to_numpy() for j in joints]).reshape(len(joints), 2, self.K, self.m)
        return self.w * p.sum(2) + self.F

    # the user's method
    def joints(self, **kw):
        return methods.feasible_joints(self.prim, self.aux, self.a_levels, self.outcome,
                                       max_joints=np.inf, **kw)

    def grid_size(self, num):
        return methods.grid_size(self.prim, self.aux, self.a_levels, self.outcome, num)

    # exact optimisation
    def _lp(self, c, cD=None, t=None):
        """argmin c.x over the feasible q (flattened), optionally with cD.x = t; None if infeasible."""
        A_eq, b_eq = np.kron(np.eye(self.K), np.ones((1, self.m))), self.p_a
        if cD is not None:
            A_eq, b_eq = np.vstack([A_eq, cD]), np.append(b_eq, t)
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, method="highs", options=LP_TOL,
                      bounds=np.c_[np.zeros(self.K * self.m), self.p_o.ravel()])
        return np.clip(res.x, 0.0, self.p_o.ravel()) if res.status == 0 else None

    def exact_range(self, metric):
        num, den = masks(self.cells, metric)
        coef = lambda mask: self.w * (self.p_z[:, None] * mask[None, :]).ravel()   # noqa: E731
        cN, cD = coef(num), coef(den)
        N0, D0 = self.F[0, num].sum(), self.F[0, den].sum()
        tot = self.w * (self.p_z[:, None] * self.p_o).sum(0) + self.F.sum(0)
        N, D = tot[num].sum(), tot[den].sum()
        f = lambda Na, Da: Na / Da - (N - Na) / (D - Da)                             # noqa: E731

        def best(t, sign, fixed_t=True):
            """max (sign=1) or min (sign=-1) of f at D_a = t, evaluated at the LP solution."""
            x = self._lp(-sign * cN, cD, t - D0) if fixed_t else self._lp(-sign * cN)
            return np.nan if x is None else f(N0 + cN @ x, D0 + cD @ x)

        tlo, thi = D0 + cD @ self._lp(cD), D0 + cD @ self._lp(-cD)
        if thi - tlo < 1e-12:                                            # DD: D_a fixed
            return best(tlo, -1, False), best(tlo, 1, False)
        eps = 1e-9 * D
        ts = np.linspace(max(tlo, eps), min(thi, D - eps), N_T)
        out = []
        for sign in (-1, 1):
            vals = np.array([sign * best(t, sign) for t in ts])
            i = np.nanargmax(vals)
            r = minimize_scalar(lambda t: -sign * best(t, sign), method="bounded",
                                bounds=(ts[max(i - 1, 0)], ts[min(i + 1, N_T - 1)]),
                                options={"xatol": 1e-12})
            out.append(sign * max(vals[i], -r.fun if np.isfinite(r.fun) else -np.inf))
        return tuple(out)

    def exact(self):
        """{metric: (lo, hi, min|f|)} by optimisation."""
        res = {}
        for m in metrics_for(self.outcome):
            lo, hi = self.exact_range(m)
            res[m] = (lo, hi, 0.0 if lo <= 0 <= hi else min(abs(lo), abs(hi)))
        return res


def summarise(G, cells, metrics):
    """{metric: (lo, hi, min|f|)} over a set of joints."""
    out = {}
    for m in metrics:
        v = disparity(G, cells, m)
        out[m] = (np.nanmin(v), np.nanmax(v), np.nanmin(np.abs(v)))
    return out


QUANTS = ("lo", "hi", "min_abs")


def errors(est, exact):
    """|estimate - exact| / exact width, per metric and quantity."""
    return {f"{m}_{q}": abs(est[m][i] - exact[m][i]) / (exact[m][1] - exact[m][0])
            for m in est for i, q in enumerate(QUANTS)}


# ------------------------------------------------------------------ adaptive (zoom) grid
def _window_splits(s, u, lo_w, hi_w, num):
    """methods._grid_splits with the free parameters restricted to [lo_w, hi_w]."""
    if len(u) == 1:
        return [[s]]
    flo, fhi = max(0.0, s - u[1:].sum()), min(u[0], s)
    lo, hi = max(flo, lo_w[0]), min(fhi, hi_w[0])
    if lo > hi:
        lo = hi = min(max((lo_w[0] + hi_w[0]) / 2, flo), fhi)
    return [[q] + rest for q in np.unique(np.linspace(lo, hi, num))
            for rest in _window_splits(s - q, u[1:], lo_w[1:], hi_w[1:], num)]


def _window_grid(prob, center, half, num):
    """All q (n x K x m) with a num-point grid on each free parameter of [center - half, center + half]."""
    options = [np.array(_window_splits(s, u, c - h, c + h, num))
               for s, u, c, h in zip(prob.p_a, prob.p_o, center, half)]
    idx = np.stack(np.meshgrid(*[np.arange(len(o)) for o in options], indexing="ij"), -1).reshape(-1, prob.K)
    return np.stack([o[idx[:, k]] for k, o in enumerate(options)], axis=1)


def adaptive_grid(prob, objectives, num=3, rounds=20, shrink=0.5):
    """Minimise each objective(G) by zooming. Round 0 is the full num-point grid (the same
    points as feasible_joints(method="grid", num=num)), evaluated once for all objectives;
    then, per objective, a num-point grid on every free parameter of a box around the
    incumbent, the box halving each round.
    Returns (round-0 evaluations, {name: [(zoom evaluations so far, best value)] per round})."""
    Q0 = _window_grid(prob, prob.p_o / 2, prob.p_o, num)
    G0 = prob.G_from_q(Q0)
    traces = {}
    for name, objective in objectives.items():
        v = objective(G0)
        i = np.nanargmin(v)
        best, center, half, evals = v[i], Q0[i], prob.p_o * shrink, 0
        trace = [(0, best)]
        for _ in range(rounds - 1):
            Q = _window_grid(prob, center, half, num)
            v = objective(prob.G_from_q(Q))
            evals += len(Q)
            i = np.nanargmin(v)
            if v[i] < best:
                best, center = v[i], Q[i]
            trace.append((evals, best))
            half = half * shrink
        traces[name] = trace
    return len(Q0), traces


# ------------------------------------------------------------------ helpers
def h_to_num(h):
    return int(round(1 / h)) + 1


def pmap(func, tasks, jobs):
    if jobs <= 1:
        return [func(t) for t in tasks]
    with ProcessPoolExecutor(jobs) as ex:
        return list(ex.map(func, tasks))


def save(df, out_dir, name):
    df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
    print(f"saved {os.path.join(out_dir, name)}.csv")


# ================================================================== RQ1 validity
def _rq1_task(task):
    name, K, seed, outcome, nums = task
    prim, aux, a_true, a_levels = german() if name == "german" else synthetic_instance(K, seed)
    prob = Problem(prim, aux, a_levels, outcome)
    t = time.perf_counter()
    exact = prob.exact()
    lp_s = time.perf_counter() - t
    rows = []
    for num in nums:
        if prob.grid_size(num) > MAX_BUILD:
            continue
        t = time.perf_counter()
        joints = prob.joints(method="grid", num=num)
        G = prob.G_from_joints(joints)
        grid_s = time.perf_counter() - t
        # every joint reproduces both marginals
        p = np.stack([j["p"].to_numpy() for j in joints]).reshape(len(joints), 2, prob.K, prob.m)
        marg_err = max(np.abs(p.sum(1) - prob.p_z[:, None] * prob.p_o).max(),
                       np.abs(p[:, 0].sum(-1) - prob.p_z * prob.p_a).max())
        # vectorised disparity agrees with methods.joint_disparity
        with np.errstate(divide="ignore", invalid="ignore"):
            chk = np.nanmax([abs(disparity(G[i], prob.cells, m) - methods.joint_disparity(joints[i], a_levels, m))
                             for i in np.linspace(0, len(joints) - 1, 5).astype(int) for m in metrics_for(outcome)])
        grid = summarise(G, prob.cells, metrics_for(outcome))
        for m in metrics_for(outcome):
            k_lo, k_hi = methods.kallus_bounds(prim, aux, a_levels, m)
            truth = methods.true_disparity(prim["yhat"], prim["y"], a_true, a_levels, m)
            (g_lo, g_hi, g_abs), (e_lo, e_hi, e_abs) = grid[m], exact[m]
            rows.append({"dataset": name, "K": prob.K, "seed": seed, "outcome": ",".join(outcome),
                         "d": prob.d, "num": num, "h": 1 / (num - 1), "n_joints": len(joints),
                         "metric": m, "grid_lo": g_lo, "grid_hi": g_hi, "exact_lo": e_lo,
                         "exact_hi": e_hi, "kallus_lo": k_lo, "kallus_hi": k_hi, "truth": truth,
                         "err_lo": g_lo - e_lo, "err_hi": e_hi - g_hi,
                         "coverage": (g_hi - g_lo) / (e_hi - e_lo),
                         "grid_min_abs": g_abs, "exact_min_abs": e_abs,
                         "kallus_vs_exact": max(abs(k_lo - e_lo), abs(k_hi - e_hi)),
                         "inside_exact": (g_lo >= e_lo - 1e-7) and (g_hi <= e_hi + 1e-7),   # LP precision
                         "truth_inside_exact": e_lo <= truth <= e_hi,
                         "max_marginal_error": marg_err, "joint_disparity_check": chk,
                         "grid_s": grid_s, "lp_s": lp_s})
    return rows


def rq1(args):
    n_inst = 3 if args.quick else 20
    tasks = [("german", 2, 0, ("yhat", "y"), (2, 3, 5)), ("german", 2, 0, ("yhat",), (2, 3, 5, 11))]
    for K in ([1, 2] if args.quick else [1, 2, 3]):
        for seed in range(n_inst):
            tasks.append(("synthetic", K, seed, ("yhat", "y"), (2, 3, 5)))
            tasks.append(("synthetic", K, seed, ("yhat",), (2, 3, 5, 11)))
    df = pd.DataFrame([r for rows in pmap(_rq1_task, tasks, args.jobs) for r in rows])
    save(df, args.out_dir, "rq1")

    print("\nRQ1  grid vs exact (errors as a fraction of the exact width; 0 = exact)")
    df["rel_err"] = np.maximum(df["err_lo"], df["err_hi"]) / (df["exact_hi"] - df["exact_lo"])
    tab = df.groupby(["outcome", "metric", "num"]).agg(
        instances=("rel_err", "size"), max_rel_err=("rel_err", "max"), mean_coverage=("coverage", "mean"),
        all_inside_exact=("inside_exact", "all"), max_kallus_vs_exact=("kallus_vs_exact", "max"),
        truth_inside=("truth_inside_exact", "mean"), max_marginal_error=("max_marginal_error", "max"))
    print(tab.to_string(float_format=lambda x: f"{x:.2e}" if abs(x) < 1e-3 and x != 0 else f"{x:.4f}"))
    plot_rq1(df, os.path.join(args.out_dir, "rq1.png"))


def plot_rq1(df, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    for ax, m in zip(axes, methods.METRICS):
        d = df[df["metric"] == m]
        for i, (num, g) in enumerate(d.groupby("num")):
            for side, mk in (("lo", "v"), ("hi", "^")):
                ax.scatter(g[f"exact_{side}"], g[f"grid_{side}"], s=22, marker=mk, color=SERIES[i],
                           edgecolor="white", lw=0.5, label=f"num={num} (h={1 / (num - 1):g})" if side == "lo" else None)
        lim = [d[["exact_lo", "grid_lo"]].min().min() - 0.05, d[["exact_hi", "grid_hi"]].max().max() + 0.05]
        ax.plot(lim, lim, color=MUTED, lw=1, ls="--")
        ax.set(xlabel="exact bound (LP)", ylabel="grid bound", title=f"{m}  (▼ lower, ▲ upper)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("RQ1: grid bounds vs exact bounds, every instance (on the diagonal = exact)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


# ================================================================== RQ2 resolution
RQ2_SETTINGS = [("d=1  (K=1, O=Ŷ)", 1, ("yhat",)), ("d=2  (K=2, O=Ŷ)", 2, ("yhat",)),
                ("d=3  (K=3, O=Ŷ)", 3, ("yhat",)), ("d=3  (K=1, O=Ŷ,Y)", 1, ("yhat", "y")),
                ("d=6  (K=2, O=Ŷ,Y; German)", 2, ("yhat", "y"))]
RQ2_H = [0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.0025]


def _rq2_task(task):
    label, K, outcome, seed, hs = task
    prim, aux, _, a_levels = german() if "German" in label else synthetic_instance(K, seed)
    prob = Problem(prim, aux, a_levels, outcome)
    exact = prob.exact()
    rows = []
    for h in hs:
        num = h_to_num(h)
        size = prob.grid_size(num)
        if size > MAX_BUILD:
            continue
        t = time.perf_counter()
        G = prob.G_from_joints(prob.joints(method="grid", num=num))
        rows.append({"setting": label, "seed": seed, "h": h, "num": num, "n_joints": size,
                     "seconds": time.perf_counter() - t,
                     **errors(summarise(G, prob.cells, metrics_for(outcome)), exact)})
    return rows


def rq2(args):
    n_inst = 2 if args.quick else 10
    hs = RQ2_H[:4] if args.quick else RQ2_H
    tasks = [(label, K, outcome, seed, hs) for label, K, outcome in RQ2_SETTINGS
             for seed in range(1 if "German" in label else n_inst)]
    df = pd.DataFrame([r for rows in pmap(_rq2_task, tasks, args.jobs) for r in rows])
    save(df, args.out_dir, "rq2")
    errs = [c for c in df.columns if c.endswith(QUANTS)]
    print("\nRQ2  max error over instances (fraction of exact width) by grid step h")
    tab = df.groupby(["setting", "h"], sort=False)[["n_joints"] + errs].max()
    print(tab.to_string(float_format=lambda x: f"{x:.1e}"))
    plot_rq2(df, os.path.join(args.out_dir, "rq2.png"))


def plot_rq2(df, path):
    settings = df["setting"].unique()
    fig, axes = plt.subplots(1, len(settings), figsize=(3.3 * len(settings), 3.8),
                             constrained_layout=True, sharey=True)
    floor = 1e-9
    for ax, s in zip(np.atleast_1d(axes), settings):
        d = df[df["setting"] == s]
        for i, m in enumerate(methods.METRICS):
            if f"{m}_lo" not in d or d[f"{m}_lo"].isna().all():
                continue
            g = d.groupby("h")
            bounds = np.maximum(g[f"{m}_lo"].max(), g[f"{m}_hi"].max()).clip(lower=floor)
            ax.plot(bounds.index, bounds.values, marker="o", ms=4, lw=2, color=SERIES[i],
                    label=f"{m} bounds")
            mabs = g[f"{m}_min_abs"].max().clip(lower=floor)
            ax.plot(mabs.index, mabs.values, marker="s", ms=3.5, lw=1.5, ls="--", color=SERIES[i],
                    label=f"min |{m}|")
        hs = np.sort(d["h"].unique())
        ax.plot(hs, hs / 2, color=MUTED, lw=1, ls=":")
        ax.set(xscale="log", yscale="log", xlabel="grid step h", title=s, ylim=(floor / 2, 2))
        ax.set_xticks(hs, [f"{h:g}" for h in hs], rotation=45, fontsize=7)
        ax.minorticks_off()
        ax.invert_xaxis()
        ax.legend(fontsize=7, loc="center left")
    ax0 = np.atleast_1d(axes)[0]
    ax0.set_ylabel("max error over instances / exact width")
    fig.suptitle(f"RQ2: error of the grid bounds vs grid step h  (dotted: h/2; values at {floor:g} = exact)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


# ================================================================== RQ3 runtime
def rq3(args):
    ds = [1, 2, 3] if args.quick else [1, 2, 3, 4, 5, 6]
    hs = [0.5, 0.25, 0.1, 0.05]
    rows = []
    settings = [(d, d, ("yhat",)) for d in ds] + [(3, 1, ("yhat", "y")), (6, 2, ("yhat", "y"))]
    for d, K, outcome in settings:
        prim, aux, _, a_levels = synthetic_instance(K, 0)
        prob = Problem(prim, aux, a_levels, outcome)
        t = time.perf_counter()
        prob.exact()
        lp_s = time.perf_counter() - t
        for h in hs:
            num = h_to_num(h)
            size = prob.grid_size(num)
            row = {"d": d, "K": K, "outcome": ",".join(outcome), "h": h, "num": num,
                   "n_joints": size, "lp_s": lp_s, "measured": size <= MAX_BUILD}
            if row["measured"]:
                reps = 3 if size <= 20_000 else 1
                build, ev = [], []
                for _ in range(reps):
                    t = time.perf_counter()
                    joints = prob.joints(method="grid", num=num)
                    build.append(time.perf_counter() - t)
                    t = time.perf_counter()
                    summarise(prob.G_from_joints(joints), prob.cells, metrics_for(outcome))
                    ev.append(time.perf_counter() - t)
                # per-joint cost of the pandas evaluation (methods.joint_disparity) on a sample
                sample = joints[:: max(1, len(joints) // 500)]
                t = time.perf_counter()
                for m in metrics_for(outcome):
                    methods.joint_metrics(sample, lambda j: methods.joint_disparity(j, a_levels, m))
                pandas_per_joint = (time.perf_counter() - t) / len(sample)
                row |= {"build_s": min(build), "eval_s": min(ev),
                        "pandas_eval_s": pandas_per_joint * size}
                del joints
            rows.append(row)
            print(row, flush=True)
    df = pd.DataFrame(rows)
    # extrapolate the grids too large to build from the measured per-joint cost
    m = df[df["measured"] & (df["n_joints"] >= 1000)]
    per_joint = ((m["build_s"] + m["eval_s"]) / m["n_joints"]).median()
    df["total_s"] = np.where(df["measured"], df["build_s"] + df["eval_s"], df["n_joints"] * per_joint)
    save(df, args.out_dir, "rq3")
    print(f"\nRQ3  per-joint cost (build + vectorised eval): {per_joint * 1e6:.0f} µs")
    print(df[["d", "outcome", "h", "n_joints", "measured", "total_s", "pandas_eval_s", "lp_s"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    plot_rq3(df, os.path.join(args.out_dir, "rq3.png"))


def plot_rq3(df, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    d = df[df["outcome"] == "yhat"]
    for i, (h, g) in enumerate(d.groupby("h", sort=False)):
        for ax, col in zip(axes, ("n_joints", "total_s")):
            ax.plot(g["d"], g[col], color=SERIES[i], lw=2, label=f"h={h:g} (num={g['num'].iloc[0]})")
            ax.plot(g["d"][g["measured"]], g[col][g["measured"]], "o", ms=6, color=SERIES[i])
            ax.plot(g["d"][~g["measured"]], g[col][~g["measured"]], "o", ms=6, mfc="white", color=SERIES[i])
    lp = d.groupby("d")["lp_s"].first()
    axes[1].plot(lp.index, lp.values, color=INK, lw=2, ls="--", label="exact LP (DD)")
    for s in (60, 3600, 86400):
        axes[1].axhline(s, color=MUTED, lw=0.8, ls=":")
        axes[1].annotate({60: "1 min", 3600: "1 hour", 86400: "1 day"}[s], (1, s), xytext=(0, 2),
                         textcoords="offset points", fontsize=7, color=MUTED)
    axes[0].set(yscale="log", xlabel="free parameters d  (O = Ŷ, K = d)", ylabel="grid joints",
                title="Grid size = num^d")
    axes[1].set(yscale="log", xlabel="free parameters d  (O = Ŷ, K = d)",
                ylabel="seconds (build + bounds)", title="Runtime  (hollow = extrapolated)")
    axes[1].legend(fontsize=8)
    fig.suptitle("RQ3: computational cost of the full grid", x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


# ================================================================== RQ4 adaptive refinement
RQ4_SETTINGS = [("d=2  (K=2, O=Ŷ)", 2, ("yhat",)), ("d=4  (K=4, O=Ŷ)", 4, ("yhat",)),
                ("d=6  (K=6, O=Ŷ)", 6, ("yhat",)), ("d=6  (K=2, O=Ŷ,Y)", 2, ("yhat", "y"))]
RQ4_SAMPLES = [30, 100, 300, 1000, 3000, 10_000, 30_000, 100_000]


def _objectives(cells, metric):
    f = lambda G: disparity(G, cells, metric)                                         # noqa: E731
    return {"lo": f, "hi": lambda G: -f(G), "min_abs": lambda G: np.abs(f(G))}


def _rq4_task(task):
    label, K, outcome, seed, quick = task
    prim, aux, _, a_levels = synthetic_instance(K, seed)
    prob = Problem(prim, aux, a_levels, outcome)
    exact = prob.exact()
    metrics = metrics_for(outcome)
    rows = []
    base = {"setting": label, "seed": seed, "d": prob.d}

    def add(method, evals, est, seconds, n_seed=0):
        e = errors(est, exact)
        rows.append({**base, "method": method, "evals": evals, "sampler_seed": n_seed,
                     "seconds": seconds, "max_err": max(e.values()), **e})

    for num in range(2, 60):                                             # full grid
        size = prob.grid_size(num)
        if size > (20_000 if quick else MAX_BUILD):
            break
        t = time.perf_counter()
        G = prob.G_from_joints(prob.joints(method="grid", num=num))
        add(f"grid", size, summarise(G, prob.cells, metrics), time.perf_counter() - t)

    samples = RQ4_SAMPLES[:5] if quick else RQ4_SAMPLES
    for method in ("random", "hit_and_run"):
        for s in range(1 if quick else 3):
            for n in samples:
                t = time.perf_counter()
                G = prob.G_from_joints(prob.joints(method=method, n_samples=n, seed=s))
                add(method, n, summarise(G, prob.cells, metrics), time.perf_counter() - t, s)

    # adaptive: shared coarse grid, then one zoom per target; evaluations summed over targets
    t = time.perf_counter()
    n0, traces = adaptive_grid(prob, {(m, q): obj for m in metrics
                                      for q, obj in _objectives(prob.cells, m).items()})
    seconds = time.perf_counter() - t
    for r in range(len(next(iter(traces.values())))):
        est = {m: tuple((1 if q != "hi" else -1) * traces[(m, q)][r][1] for q in QUANTS) for m in metrics}
        add("adaptive", n0 + sum(tr[r][0] for tr in traces.values()), est, seconds)
    return rows


def rq4(args):
    n_inst = 2 if args.quick else 5
    tasks = [(label, K, outcome, seed, args.quick) for label, K, outcome in RQ4_SETTINGS
             for seed in range(n_inst)]
    df = pd.DataFrame([r for rows in pmap(_rq4_task, tasks, args.jobs) for r in rows])
    save(df, args.out_dir, "rq4")

    print("\nRQ4  fewest evaluations reaching max error <= tol (median over instances; '-' = never)")
    out = []
    for (label, method), g in df.groupby(["setting", "method"], sort=False):
        row = {"setting": label, "method": method}
        for tol in (1e-2, 1e-3, 1e-6):
            per_inst = [gg.loc[gg["max_err"] <= tol, "evals"].min() for _, gg in g.groupby(["seed", "sampler_seed"])]
            row[f"tol={tol:g}"] = np.median(per_inst) if np.isfinite(per_inst).all() else np.nan
        row["best max_err"] = g.groupby(["seed", "sampler_seed"])["max_err"].min().median()
        out.append(row)
    print(pd.DataFrame(out).to_string(index=False, na_rep="-", float_format=lambda x: f"{x:.3g}"))
    plot_rq4(df, os.path.join(args.out_dir, "rq4.png"))


def plot_rq4(df, path):
    settings = df["setting"].unique()
    fig, axes = plt.subplots(1, len(settings), figsize=(3.4 * len(settings), 3.9),
                             constrained_layout=True, sharey=True)
    floor = 1e-9
    for ax, s in zip(axes, settings):
        d = df[df["setting"] == s]
        for i, method in enumerate(("grid", "adaptive", "random", "hit_and_run")):
            g = d[d["method"] == method]
            # median over instances (and sampler seeds) at each budget
            g = g.assign(b=g["evals"] if method != "adaptive" else
                         g.groupby(["seed"]).cumcount())
            agg = g.groupby("b").agg(evals=("evals", "median"), err=("max_err", "median"))
            ax.plot(agg["evals"], agg["err"].clip(lower=floor), marker="o", ms=3.5, lw=2,
                    color=SERIES[i], label=method.replace("_", "-"))
        ax.set(xscale="log", yscale="log", xlabel="joints evaluated", title=s, ylim=(floor / 2, 2))
    axes[0].set_ylabel("max error over bounds / exact width")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("RQ4: error vs evaluation budget (median over instances; floor = exact)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


# ================================================================== RQ5 overlap
OVERLAPS = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50]


def _aux_from_probs(p, a_levels, n=100_000):
    """An auxiliary frame whose P(A = a_levels[0] | Z) equals p (to 1/n)."""
    rows = [(z, a_levels[0] if i < round(pz * n) else a_levels[1]) for z, pz in p.items() for i in range(n)]
    return pd.DataFrame(rows, columns=["z", "a"])


def linked_problem(prim, aux, a_true, a_levels, outcome, linked):
    """Feasible set when the rows `linked` (boolean) of the primary data also carry A.
    Population joint = rho * (linked joint, known) + (1 - rho) * (unlinked joint, feasible),
    the unlinked part constrained by its own P(Z, O) and by
    P(A | Z, unlinked) = (n_z P(A | Z) - linked A counts) / unlinked n_z.
    No assumption is made about how the linked rows were selected."""
    if not linked.any():
        return Problem(prim, aux, a_levels, outcome)
    unl, al = prim[~linked], a_true[linked]
    cells = list(itertools.product([0, 1], repeat=len(outcome)))
    vals = prim.loc[linked, list(outcome)].to_numpy()
    F = np.array([[((al == g) & (vals == c).all(1)).sum() for c in cells] for g in a_levels]) / len(prim)
    pa = methods.p_a_given_z(aux, a_levels)[a_levels[0]]
    n_z, n_unl = prim["z"].value_counts(), unl["z"].value_counts()
    n_link_a = prim.loc[linked, "z"][al == a_levels[0]].value_counts().reindex(n_unl.index, fill_value=0)
    p_unl = ((n_z[n_unl.index] * pa[n_unl.index] - n_link_a) / n_unl).clip(0, 1)
    return Problem(unl, _aux_from_probs(p_unl, a_levels), a_levels, outcome, w=1 - linked.mean(), F=F)


def _rq5_task(task):
    name, selection, rho, rep, num = task
    prim, aux, a_true, a_levels = german() if name == "german" else synthetic_instance(3, 0)
    rng = np.random.default_rng(rep)
    n, n_link = len(prim), int(round(rho * len(prim)))
    # selective linkage: rows with A = a_levels[0] and Yhat = 1 are 4x as likely to be linked
    wt = np.ones(n) if selection == "random" else np.where((a_true == a_levels[0]) & (prim["yhat"] == 1), 4.0, 1.0)
    linked = np.zeros(n, bool)
    linked[rng.choice(n, n_link, replace=False, p=wt / wt.sum())] = True
    prob = linked_problem(prim, aux, a_true, a_levels, ("yhat", "y"), linked)
    exact = prob.exact()
    grid = summarise(prob.G_from_joints(prob.joints(method="grid", num=num)), prob.cells, methods.METRICS)
    rows, al = [], a_true[linked]
    for m in methods.METRICS:
        truth = methods.true_disparity(prim["yhat"], prim["y"], a_true, a_levels, m)
        with np.errstate(divide="ignore", invalid="ignore"):             # complete-case estimate
            cc = methods.true_disparity(prim.loc[linked, "yhat"], prim.loc[linked, "y"], al, a_levels, m)
        rows.append({"dataset": name, "selection": selection, "overlap": rho, "rep": rep, "metric": m,
                     "n_linked": n_link, "exact_lo": exact[m][0], "exact_hi": exact[m][1],
                     "grid_lo": grid[m][0], "grid_hi": grid[m][1], "width": exact[m][1] - exact[m][0],
                     "grid_width": grid[m][1] - grid[m][0], "truth": truth, "complete_case": cc,
                     "truth_inside": exact[m][0] - 1e-9 <= truth <= exact[m][1] + 1e-9,
                     "excludes_zero": exact[m][0] > 0 or exact[m][1] < 0})
    return rows


def rq5(args):
    reps = 3 if args.quick else 20
    tasks = [(name, sel, rho, rep, num) for name, num in (("synthetic", 5), ("german", 5))
             for sel in ("random", "selective") for rho in OVERLAPS
             for rep in range(1 if rho == 0 else reps)]
    df = pd.DataFrame([r for rows in pmap(_rq5_task, tasks, args.jobs) for r in rows])
    save(df, args.out_dir, "rq5")
    print("\nRQ5  exact interval width by overlap (median over linkage draws)")
    tab = df.groupby(["dataset", "selection", "metric", "overlap"]).agg(
        width=("width", "median"), grid_width=("grid_width", "median"),
        rel_width=("width", "median"), truth_inside=("truth_inside", "mean"),
        excludes_zero=("excludes_zero", "mean"), truth=("truth", "first"),
        cc_median=("complete_case", "median"),
        cc_abs_err=("complete_case", lambda s: np.nanmedian(np.abs(s - df.loc[s.index, "truth"]))))
    w0 = tab["width"].xs(0.0, level="overlap")
    tab["rel_width"] = tab["width"] / w0.reindex(tab.index.droplevel("overlap")).to_numpy()
    print(tab.to_string(float_format=lambda x: f"{x:.3f}"))
    plot_rq5(df, os.path.join(args.out_dir, "rq5.png"))


def plot_rq5(df, path):
    combos = list(df.groupby(["dataset", "metric"], sort=False).groups)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, (name, m) in zip(axes.ravel(), combos):
        d = df[(df["dataset"] == name) & (df["metric"] == m)]
        x = np.arange(len(OVERLAPS))
        for j, (sel, off) in enumerate((("random", -0.12), ("selective", 0.12))):
            g = d[d["selection"] == sel].groupby("overlap")
            lo, hi = g["exact_lo"].median(), g["exact_hi"].median()
            ax.vlines(x + off, lo, hi, color=SERIES[j], lw=5, alpha=0.85,
                      label=f"bounds, {sel} linkage" if ax is axes[0, 0] else None)
            cc = g["complete_case"]
            ax.errorbar(x[1:] + off, cc.median().iloc[1:], fmt="o", ms=4, mfc="white", color=INK,
                        yerr=[(cc.median() - cc.quantile(0.1)).iloc[1:], (cc.quantile(0.9) - cc.median()).iloc[1:]],
                        lw=1, capsize=2, label="linked rows only (10–90%)" if ax is axes[0, 0] and j == 0 else None)
        ax.axhline(d["truth"].iloc[0], color=INK, lw=1.2, ls="--", label="truth (true A)" if ax is axes[0, 0] else None)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(x, [f"{o:.0%}" for o in OVERLAPS])
        ax.set(xlabel="overlap (rows with A linked)", title=f"{name}: {m}")
    fig.legend(loc="outside lower center", ncol=4, fontsize=8)
    fig.suptitle("RQ5: fairness interval vs overlap (median bounds over linkage draws)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


# ==================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("rq", choices=["rq1", "rq2", "rq3", "rq4", "rq5", "all"])
    ap.add_argument("--out-dir", default="outputs/grid_validation")
    ap.add_argument("--quick", action="store_true", help="fewer instances and settings")
    ap.add_argument("--jobs", type=int, default=16, help="worker processes")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    pd.set_option("display.width", 250)
    for rq in (["rq1", "rq2", "rq3", "rq4", "rq5"] if args.rq == "all" else [args.rq]):
        globals()[rq](args)
