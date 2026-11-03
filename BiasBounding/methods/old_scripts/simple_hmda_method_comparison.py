"""
Grid enumeration vs vertex enumeration vs LP, on one HMDA case, for each type of metric, and
then for joint regions of two metrics.

Case: Black vs rest, proxy = income in 3 equal-count bins, classifier = gradient boosting
(income + county), label = historical approval. 3 free parameters per income bin, 9 in total.

Methods
  grid      grid_joint 'sweep': each free cell of each income bin's table over num values of its
            Frechet range, invalid tables dropped, all combinations (full joints)
  vertices  exact vertices of the set of feasible aggregated tables (greedy rule over cell
            orderings: the set is the base polytope of U(S) = sum_z min(P(z, S), P(z, group)));
            metrics whose extremes are not at vertices: + interior search (convex combinations
            of vertices, constrained refinement)
  LP        coupling LP (kallus_bounds.CouplingLP): linear objectives exact; ratio disparities by a
            grid over the group's condition mass (as in the Kallus et al. paper); |.| and max-type
            metrics by epigraph LPs; no formulation for nonlinear metrics such as conditional MI

Metric types
  DP_diff       linear in the joint (denominators fixed by the auxiliary data)
  TPR_diff      difference of two ratios (denominators depend on the joint)
  DI_ratio      ratio of two rates
  DP_gap        |DP_diff|: convex, minimum where DP crosses 0
  EOdds         max(|TPR_diff|, |FPR_diff|): kinked, minimum between vertices
  CMI           I(prediction; group | label): smooth, neither convex nor concave

Usage (repo root): python methods/method_comparison.py --data ~/kallus/HMDA/small_proxy_county_income.csv
Writes results/method_comparison.csv and results/method_comparison.png.
"""
import argparse
import contextlib
import io
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from BiasBounding.methods.old_scripts.hmda_pipeline import (load, split, fit_classifiers, make_proxy, marginals, _hull, to_theta,   # noqa: E402
                           all_metrics, interior_search, masks)
from kallus_bounds import CouplingLP, lp_bounds, closed_form_bounds                               # noqa: E402
from BiasBounding.methods.old_scripts.experiments import run_grid, max_gap_lp_lower, plt                                           # noqa: E402

METRICS = {"DP_diff": "linear", "TPR_diff": "difference of ratios", "DI_ratio": "ratio",
           "DP_gap": "convex |linear|", "EOdds": "max of |ratio diffs|", "CMI": "nonconvex smooth"}


# ------------------------------------------------------------------------------------ methods
def greedy_vertices(R, M):
    """Vertices of the feasible aggregated group tables: for each ordering of the cells, give
    each cell U(first k cells) - U(first k-1 cells), U(S) = sum_z min(P(z, S), P(z, group))."""
    I = R.shape[1]
    U = lambda S: np.minimum(R[:, list(S)].sum(1), M[:, 1]).sum() if S else 0.0
    out = []
    for p in itertools.permutations(range(I)):
        th, prev = np.zeros(I), 0.0
        for k in range(I):
            v = U(p[:k + 1]); th[p[k]] = v - prev; prev = v
        out.append(th)
    return np.unique(np.round(out, 13), axis=0)


def method_grid(R, M, I, Zs, num):
    pp = {(I[i], Zs[z]): R[z, i] for z in range(len(Zs)) for i in range(len(I))}
    pa = {(Zs[z], g): M[z, k] for z in range(len(Zs)) for k, g in enumerate(["0_rest", "1_grp"])}
    t0 = time.perf_counter()
    g = run_grid(pp, pa, num, "sweep", 5e6)
    if g is None:
        return None
    J, Ig, _, _ = g
    theta = J.sum(1)[:, [Ig.index(v) for v in I], :]
    vals = all_metrics(theta, I)
    return vals, time.perf_counter() - t0, len(J), theta


def method_vertices(R, M, I, samples):
    t0 = time.perf_counter()
    V = greedy_vertices(R, M)
    _, info = _hull(V)
    Ragg = R.sum(0)
    vals = all_metrics(to_theta(V, Ragg), I)
    t_v = time.perf_counter() - t0
    out = {}
    for k in METRICS:
        lo, hi = np.nanmin(vals[k]), np.nanmax(vals[k])
        how = "vertices"
        if k in ("DP_gap", "EOdds", "CMI"):          # extremes may lie inside the polytope
            t1 = time.perf_counter()
            slo, shi = interior_search(lambda th, k=k: all_metrics(th, I)[k], V, info, Ragg,
                                       samples, seed=0, RM=(R, M))
            lo, hi = np.nanmin([lo, slo]), np.nanmax([hi, shi])
            how = "vertices + interior search"
            out[k] = (lo, hi, t_v + time.perf_counter() - t1, how)
        else:
            out[k] = (lo, hi, t_v, how)
    return out, V, info


def method_lp(R, M, I):
    d, y = masks(I)
    ones = np.ones_like(d)
    out = {}
    t0 = time.perf_counter()
    b = lp_bounds(R, M, d, ones, harmonize=False).iloc[0]
    out["DP_diff"] = (b.lower, b.upper, time.perf_counter() - t0, "LP (exact: linear)")
    gl = 0.0 if b.lower <= 0 <= b.upper else min(abs(b.lower), abs(b.upper))
    out["DP_gap"] = (gl, max(abs(b.lower), abs(b.upper)), time.perf_counter() - t0,
                     "from the DP_diff LP interval (exact)")
    t0 = time.perf_counter()
    b = lp_bounds(R, M, d & y, y, harmonize=False).iloc[0]
    out["TPR_diff"] = (b.lower, b.upper, time.perf_counter() - t0, "LP + grid over denominator")
    t0 = time.perf_counter()
    lo = max_gap_lp_lower(R, M, [(d & y, y), (d & ~y, ~y)], n_grid=41)
    bt = lp_bounds(R, M, d & y, y, harmonize=False).iloc[0]
    bf = lp_bounds(R, M, d & ~y, ~y, harmonize=False).iloc[0]
    up = max(abs(bt.lower), abs(bt.upper), abs(bf.lower), abs(bf.upper))
    out["EOdds"] = (lo, up, time.perf_counter() - t0, "epigraph LP + 2-d grid (lower); max of |.| (upper)")
    out["DI_ratio"] = (np.nan, np.nan, np.nan, "not formulated (needs a Charnes-Cooper LP)")
    out["CMI"] = (np.nan, np.nan, np.nan, "no LP formulation (nonlinear)")
    return out


def lp_support_pair(R, M, I, rho, n_grid=81):
    """LP support function of (DP_diff, TPR_diff): max rho1*DP + rho2*TPR over couplings. DP is
    linear; fixing the group's mass x on Y=1 makes TPR linear too; x is searched on a grid.
    Returns the maximising (DP, TPR) point."""
    d, y = masks(I)
    lp = CouplingLP(R, M, harmonize=False)
    Tdp, _ = lp.masks(d, np.ones_like(d))
    Tt, Xt = lp.masks(d & y, y)
    p1, p0 = M[:, 1].sum(), M[:, 0].sum()
    K = lp.solve(Xt.sum(0))[0]
    xr = (-lp.solve(-Xt[1])[0], lp.solve(Xt[1])[0])
    best, arg = -np.inf, None
    for x1 in np.linspace(*xr, n_grid)[1:-1]:
        x0 = K - x1
        cdp = Tdp[1] / p1 - Tdp[0] / p0
        ctp = Tt[1] / x1 - Tt[0] / x0
        A_eq = sparse.vstack([lp.A_eq, sparse.csr_matrix(Xt)])
        b_eq = np.concatenate([lp.b_eq, [x0, x1]])
        res = linprog(-(rho[0] * cdp + rho[1] * ctp), A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
        if res.status == 0 and -res.fun > best:
            q = res.x                              # both groups' masses are LP variables, so
            best, arg = -res.fun, (cdp @ q, ctp @ q)   # these products are the disparities exactly
    return arg


# ------------------------------------------------------------------------------------ set-up
def describe_setup(a, df, sp, z_ev, z_ext, yhat, y, race_ev, race_ext, Zs, I):
    """Print (and return) the exact data set-up: variables, values and counts of each dataset."""
    L = []
    p = L.append
    g = a.group
    inc = df.applicant_income_000s
    p(f"DATA SET-UP  ({os.path.basename(a.data)}: {len(df):,} mortgage applications, HMDA 1% sample)")
    p(f"columns used: outcome (1 = approved), race, applicant_income_000s, county_id")
    p(f"race counts: " + ", ".join(f"{k} {v:,}" for k, v in df.race.value_counts().items()))
    p("")
    p("SPLIT (random, seed 0; disjoint rows)")
    p(f"  classifier training rows : {len(sp['train']):>7,}  (internal; features + label)")
    p(f"  internal evaluation rows : {len(sp['eval']):>7,}  (the audited dataset; race removed)")
    p(f"  external rows            : {len(sp['external']):>7,}  (proxy + race only)")
    p("")
    p("CLASSIFIER  gbm_income_county: gradient-boosted trees on log(1 + income) and a county score")
    p("  (county approval rate in the training rows, shrunk towards the overall rate); threshold set")
    p(f"  so the training approval rate is matched. On the evaluation rows: accuracy {(yhat == y).mean():.3f}, "
      f"approval rate {yhat.mean():.3f}")
    p("")
    p("VARIABLES")
    p("  internal  (primary data only):  Yhat = prediction in {0, 1};  Y = historical approval in {0, 1}")
    p("            -> 4 internal values: " + ", ".join(f"(Yhat={v[0]}, Y={v[1]})" for v in I))
    edges = np.unique(np.quantile(inc, np.linspace(0, 1, int(a.proxy[8:]) + 1))) if a.proxy.startswith("income_q") else None
    p(f"  common    (both datasets):      Z = proxy '{a.proxy}', {len(Zs)} values"
      + (f" (income terciles of the full data, edges {np.round(edges, 0).tolist()} in $1000s)" if edges is not None else ""))
    p(f"  external  (auxiliary data only): A = race, one vs rest: {{{g}, rest}}")
    p("  never observed together: (Yhat, Y) with A  -> the joint p(Z, Yhat, Y, A) is only partially identified")
    p(f"  unknowns: per Z value a {len(I)} x 2 table with fixed row and column sums -> "
      f"({len(I)}-1) x (2-1) = {len(I) - 1} free parameters; {len(Zs) * (len(I) - 1)} in total")
    p("")
    p(f"INTERNAL DATA: counts of (Z, Yhat, Y), n = {len(z_ev):,}")
    ic = pd.Series([f"({a_},{b_})" for a_, b_ in zip(yhat, y)])
    tab = pd.crosstab(pd.Series(z_ev, name="Z"), ic.rename("(Yhat,Y)"))
    rng_ = pd.Series(inc.to_numpy()[sp["eval"]]).groupby(z_ev).agg(["min", "max"])
    tab.index = [f"{z} (income {rng_.loc[z, 'min']:.0f}-{rng_.loc[z, 'max']:.0f})" for z in tab.index]
    tab["total"] = tab.sum(1)
    p(tab.to_string())
    p("")
    p(f"EXTERNAL DATA: counts of (Z, A), n = {len(z_ext):,}")
    te = pd.crosstab(pd.Series(z_ext, name="Z"), pd.Series(np.where(race_ext == g, g, "rest"), name="A"))
    te[f"P({g} | Z)"] = (te[g] / te.sum(1)).round(4)
    p(te.to_string())
    p("")
    p(f"HIDDEN TRUTH (evaluation rows with their real race; used only to check the bounds)")
    th = pd.crosstab([pd.Series(z_ev, name="Z"), pd.Series(np.where(race_ev == g, g, "rest"), name="A")],
                     ic.rename("(Yhat,Y)"))
    p(th.to_string())
    p("")
    p("WHAT THE METHODS RECEIVE: p(Z, Yhat, Y) from the internal counts, and P(A | Z) from the external")
    p("counts combined with P(Z) from the internal data: p(Z, A) = P(Z) P(A | Z).")
    text = "\n".join(L)
    print(text)
    with open(os.path.join(a.out, "data_setup.txt"), "w") as fh:
        fh.write(text + "\n")
    return text


# ------------------------------------------------------------------------------------ bound plots
def plot_bounds(tab, out):
    """Figure 1: every method's interval per metric, against the exact (reference) interval and the
    true value. Figure 2: each method's worst endpoint error (log scale)."""
    if plt is None:
        return
    mets = list(METRICS)
    methods = list(dict.fromkeys(tab.method))
    colours = {m: c for m, c in zip(methods, ["#b9b9b9", "#8c8c8c", "#5f5f5f", "#2a78d6", "#eb6834"])}
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    for ax, k in zip(axes.ravel(), mets):
        s = tab[tab.metric == k].set_index("method")
        fin = s.dropna(subset=["lower"])
        ref_lo = (fin.lower - fin.lower_gap).iloc[0]
        ref_hi = (fin.upper + fin.upper_gap).iloc[0]
        ax.axvspan(ref_lo, ref_hi, color="#e8eef8", zorder=0, label="reference interval")
        for j, m in enumerate(methods):
            yy = len(methods) - 1 - j
            r = s.loc[m]
            if np.isfinite(r.lower):
                ax.plot([r.lower, r.upper], [yy, yy], lw=5, color=colours[m], solid_capstyle="butt")
                ax.plot([r.lower, r.upper], [yy, yy], "|", ms=14, mew=2, color=colours[m])
            else:
                ax.text((ref_lo + ref_hi) / 2, yy, "not available", ha="center", va="center", fontsize=8,
                        color=colours[m], style="italic")
        ax.axvline(s.truth.iloc[0], color="k", lw=1.5, label="true value")
        ax.set_yticks(range(len(methods)), methods[::-1], fontsize=8)
        ax.set_title(f"{k}  ({METRICS[k]})", fontsize=10, loc="left")
        ax.grid(axis="x", color="#eeeeee"); ax.set_axisbelow(True)
        for sd in ("top", "right"):
            ax.spines[sd].set_visible(False)
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Bounds by method (shaded: exact or best-known interval; black line: true value)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "bounds_by_method.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    w = 0.8 / len(methods)
    for j, m in enumerate(methods):
        s = tab[tab.method == m].set_index("metric").reindex(mets)
        err = np.maximum(s.lower_gap.abs(), s.upper_gap.abs()).to_numpy()
        xs = np.arange(len(mets)) + (j - (len(methods) - 1) / 2) * w
        ok = np.isfinite(err)
        ax.bar(xs[ok], np.maximum(err[ok], 1e-16), w, color=colours[m], label=m)
        for x in xs[~ok]:
            ax.text(x, 2e-16, "n/a", rotation=90, ha="center", va="bottom", fontsize=7, color=colours[m])
    ax.set_yscale("log"); ax.set_ylim(1e-16, 1)
    ax.axhspan(1e-16, 1e-12, color="#eef6ee", zorder=0)
    ax.text(len(mets) - 0.5, 3e-15, "exact (rounding)", ha="right", fontsize=8, color="#3a7a3a")
    ax.set_xticks(range(len(mets)), [f"{k}\n({METRICS[k]})" for k in mets], fontsize=8)
    ax.set_ylabel("worst endpoint error vs reference")
    ax.legend(fontsize=7, ncol=len(methods), loc="upper left")
    ax.set_title("How far each method's bounds are from the exact (or best-known) interval", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "bound_errors_by_method.png"), dpi=150)
    plt.close(fig)
    print(f"figures: {os.path.join(out, 'bounds_by_method.png')}, {os.path.join(out, 'bound_errors_by_method.png')}")


# ------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/kallus/HMDA/small_proxy_county_income.csv"))
    ap.add_argument("--group", default="Black")
    ap.add_argument("--proxy", default="income_q3")
    ap.add_argument("--nums", nargs="+", type=int, default=[3, 5, 7])
    ap.add_argument("--samples", type=int, default=60000)
    ap.add_argument("--directions", type=int, default=48)
    ap.add_argument("--out", default="results")
    ap.add_argument("--plot-only", action="store_true", help="redraw the bound figures from the saved CSV")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.plot_only:
        plot_bounds(pd.read_csv(os.path.join(a.out, "method_comparison.csv")), a.out)
        return

    df = load(a.data)
    sp = split(df, 0, 0.5, 0.5, None)
    with contextlib.redirect_stdout(io.StringIO()):
        pred = fit_classifiers(df, sp["train"], sp["eval"], ["gbm_income_county"], 0)
    y = df.outcome.to_numpy()[sp["eval"]].astype(int)
    R, M, T, I, Zs = marginals(make_proxy(df.iloc[sp["eval"]], a.proxy, ref=df).to_numpy(),
                               np.c_[pred["gbm_income_county"], y],
                               make_proxy(df.iloc[sp["external"]], a.proxy, ref=df).to_numpy(),
                               df.race.to_numpy()[sp["external"]], df.race.to_numpy()[sp["eval"]], a.group)
    truth = {k: float(v[0]) for k, v in all_metrics(T.sum(0)[None], I).items()}
    d, yv = masks(I)
    describe_setup(a, df, sp, make_proxy(df.iloc[sp["eval"]], a.proxy, ref=df).to_numpy(),
                   make_proxy(df.iloc[sp["external"]], a.proxy, ref=df).to_numpy(),
                   pred["gbm_income_county"], y, df.race.to_numpy()[sp["eval"]],
                   df.race.to_numpy()[sp["external"]], Zs, I)
    print()
    print(f"{a.group} vs rest, proxy {a.proxy} ({len(Zs)} levels, {3 * len(Zs)} free parameters), "
          f"gbm_income_county")

    # ---------------------------------------------------------------- part 1: bounds per metric
    res = {}
    for num in a.nums:
        g = method_grid(R, M, I, Zs, num)
        if g is None:
            print(f"grid num={num}: too many joints, skipped"); continue
        vals, dt, n, _ = g
        res[f"grid num={num}"] = {k: (np.nanmin(vals[k]), np.nanmax(vals[k]), dt, f"{n:,} joints") for k in METRICS}
    vres, V, info = method_vertices(R, M, I, a.samples)
    res["vertices"] = {k: (lo, hi, t, f"{len(V)} vertices; {how}") for k, (lo, hi, t, how) in vres.items()}
    res["LP"] = method_lp(R, M, I)

    # reference: exact where a closed form exists, otherwise the widest interval any method found
    ref = {}
    for k, (tg, cd) in {"DP_diff": (d, np.ones_like(d)), "TPR_diff": (d & yv, yv)}.items():
        ref[k] = closed_form_bounds(R, M, tg, cd, harmonize=False)
    ref["DI_ratio"] = closed_form_bounds(R, M, d, np.ones_like(d), ratio=True, harmonize=False)
    lo_, hi_ = ref["DP_diff"]
    ref["DP_gap"] = (0.0 if lo_ <= 0 <= hi_ else min(abs(lo_), abs(hi_)), max(abs(lo_), abs(hi_)))
    for k in ("EOdds", "CMI"):
        ref[k] = (np.nanmin([r[k][0] for r in res.values()]), np.nanmax([r[k][1] for r in res.values()]))
    rows = []
    for k, typ in METRICS.items():
        for meth, r in res.items():
            lo, hi, t, note = r[k]
            rows.append(dict(metric=k, type=typ, method=meth, lower=lo, upper=hi,
                             lower_gap=lo - ref[k][0], upper_gap=ref[k][1] - hi,
                             seconds=t, note=note, truth=truth[k]))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(a.out, "method_comparison.csv"), index=False)
    plot_bounds(tab, a.out)
    print("\nPART 1: bounds per metric type. gap = how far inside the reference interval "
          "(0 = exact; reference = closed form for DP/TPR/DI/DP_gap, best found for EOdds/CMI)")
    for k, typ in METRICS.items():
        s = tab[tab.metric == k]
        print(f"\n{k}  ({typ}; reference [{ref[k][0]:+.4f}, {ref[k][1]:+.4f}], truth {truth[k]:+.4f})")
        print(s[["method", "lower", "upper", "lower_gap", "upper_gap", "seconds", "note"]].to_string(
            index=False, float_format=lambda x: f"{x:+.4f}" if abs(x) < 100 else f"{x:.3g}"))

    # ---------------------------------------------------------------- part 2: joint regions
    print("\nPART 2: joint regions")
    rng = np.random.default_rng(0)
    Ragg = R.sum(0)
    pts = [V]
    for kk in (2, 3, 4):                                   # dense filling of the polytope
        idx = rng.integers(0, len(V), (200000, kk)); w = rng.dirichlet(np.ones(kk), 200000)
        pts.append((V[idx] * w[:, :, None]).sum(1))
    dense = all_metrics(to_theta(np.vstack(pts), Ragg), I)
    gbest = max((n for n in a.nums if f"grid num={n}" in res), default=None)
    gvals = method_grid(R, M, I, Zs, gbest)[0] if gbest else None

    def raster_share(xs, ys, poly_xy, bins=60):
        """share of the polygon's area (cells whose centre is inside) reached by the points"""
        from matplotlib.path import Path
        x0, x1 = poly_xy[:, 0].min(), poly_xy[:, 0].max()
        y0, y1 = poly_xy[:, 1].min(), poly_xy[:, 1].max()
        H, xe, ye = np.histogram2d(xs, ys, bins=bins, range=[[x0, x1], [y0, y1]])
        cx, cy = np.meshgrid((xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2, indexing="ij")
        inside = Path(poly_xy).contains_points(np.c_[cx.ravel(), cy.ravel()]).reshape(cx.shape)
        return (H[inside] > 0).mean()

    # region A: DP_diff (linear) x TPR_diff (difference of ratios): all three methods
    t0 = time.perf_counter()
    angles = np.linspace(0, 2 * np.pi, a.directions, endpoint=False)
    sup = np.array([lp_support_pair(R, M, I, (np.cos(t), np.sin(t))) for t in angles])
    t_lp = time.perf_counter() - t0
    hullA = sup[ConvexHull(sup).vertices]
    ok = np.isfinite(dense["DP_diff"]) & np.isfinite(dense["TPR_diff"])
    shareA = raster_share(dense["DP_diff"][ok], dense["TPR_diff"][ok], hullA)
    print(f"A. DP_diff x TPR_diff: LP support function ({a.directions} directions, {t_lp:.1f}s) gives the convex hull;")
    print(f"   share of that hull reached by feasible tables: {shareA:.2f} "
          f"(1 - share = combinations the LP description includes but no joint attains)")

    # region B: DP_gap (convex) x CMI (nonconvex): no LP formulation; vertices + dense vs grid
    xb, yb = dense["DP_gap"], dense["CMI"]
    okb = np.isfinite(xb) & np.isfinite(yb)
    hullB = np.c_[xb[okb], yb[okb]][ConvexHull(np.c_[xb[okb], yb[okb]]).vertices]
    shareB = raster_share(xb[okb], yb[okb], hullB)
    print(f"B. DP_gap x CMI: share of the region's convex hull actually reached: {shareB:.2f} "
          "(the LP has no formulation for CMI at all)")
    if gvals is not None:
        cover = raster_share(gvals["DP_gap"][np.isfinite(gvals["CMI"])], gvals["CMI"][np.isfinite(gvals["CMI"])], hullB)
        print(f"   grid (num={gbest}) covers {cover:.2f} of that hull's cells; dense vertex mixtures {shareB:.2f}")

    if plt is not None:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax = axes[0]
        ax.scatter(dense["DP_diff"], dense["TPR_diff"], s=1, c="#a9c7ef", rasterized=True, label="feasible tables (vertex mixtures)")
        if gvals is not None:
            ax.scatter(gvals["DP_diff"], gvals["TPR_diff"], s=3, c="#6b6b6b", rasterized=True, label=f"grid joints (num={gbest})")
        vv = all_metrics(to_theta(V, Ragg), I)
        ax.scatter(vv["DP_diff"], vv["TPR_diff"], s=30, c="#2a78d6", zorder=3, label="vertices")
        ax.fill(hullA[:, 0], hullA[:, 1], fill=False, ec="#eb6834", lw=1.6, label="LP support function (convex hull)")
        ax.plot(truth["DP_diff"], truth["TPR_diff"], "k*", ms=13, label="truth")
        ax.set_xlabel("DP disparity (linear)"); ax.set_ylabel("TPR disparity (difference of ratios)")
        ax.set_title(f"A. linear x ratio metric: hull reached {shareA:.0%}"); ax.legend(fontsize=7, loc="lower right")
        ax = axes[1]
        ax.scatter(xb, yb, s=1, c="#a9c7ef", rasterized=True, label="feasible tables (vertex mixtures)")
        if gvals is not None:
            ax.scatter(gvals["DP_gap"], gvals["CMI"], s=3, c="#6b6b6b", rasterized=True, label=f"grid joints (num={gbest})")
        ax.scatter(vv["DP_gap"], vv["CMI"], s=30, c="#2a78d6", zorder=3, label="vertices")
        ax.fill(hullB[:, 0], hullB[:, 1], fill=False, ec="#eb6834", lw=1.6, ls="--", label="convex hull (what a convex description would report)")
        ax.plot(truth["DP_gap"], truth["CMI"], "k*", ms=13, label="truth")
        ax.set_xlabel("|DP disparity| (convex)"); ax.set_ylabel("conditional MI (nonconvex)")
        ax.set_title(f"B. convex x nonconvex metric: hull reached {shareB:.0%}"); ax.legend(fontsize=7, loc="upper left")
        fig.tight_layout()
        path = os.path.join(a.out, "method_comparison.png"); fig.savefig(path, dpi=150)
        print(f"\nfigure: {path}")


if __name__ == "__main__":
    main()