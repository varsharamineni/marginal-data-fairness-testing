"""
Two-panel figure for the extensions of the feasible-set approach, in the style of
hmda_figure.py (one PDF, full text width):

  (a) Sensitivity to conditional independence: bounds on one metric over the joints whose
      dependence I(internal; A | Z) is at most delta, against delta. At delta = 0 the bounds
      reduce to the conditional-independence (CI) estimate; the dashed line marks the breakdown
      point (the smallest delta at which the target, e.g. parity, enters the bounds) and the star
      the true value at its true dependence.
  (b) Classifier comparison: each enumerated joint gives the metric for both rules at once, so
      the panel shows the separate bounds of the two rules and the bounds on their difference,
      which can exclude 0 even when the separate bounds overlap.

The computations are those of hmda_questions.py (same joints, same CI joint, same breakdown rule,
same rule B), so the numbers printed here match its csv files for the same options.

Usage
  python hmda_extensions_figure.py --repo ~/kallus
  python hmda_extensions_figure.py --repo ~/kallus --b proxy --frac 0.25 --cmp-proxy both
  python hmda_extensions_figure.py --repo ~/kallus --metric "DI 1/0" --target 0.8
Writes <out>/figures/extensions_<tag>.png and .pdf
"""
import argparse
import os
import pickle

import numpy as np

from hmda_common import OUT, PREPARED
from hmda_questions import adjusted_b, all_metrics, classifier_b, cond_mi, enumerate_joints
import hmda_figure as _hf
# style of the main figure, with fallbacks for versions of hmda_figure.py that lack a constant
COLORS = getattr(_hf, "COLORS", {"grid": "darkorange"})
STYLE = getattr(_hf, "STYLE", {})
INK = getattr(_hf, "INK", "#222222")
TRUTH = getattr(_hf, "TRUTH", "#E69F00")
REF = getattr(_hf, "REF", "#BDBDBD")
GRID = getattr(_hf, "GRID", "#ECECEC")
from ci_baseline import ci_joint, harmonize  # noqa: E402
from exp0_hmda import load_hmda  # noqa: E402
from synth import metrics  # noqa: E402
from hmda_prepare import GROUPS, classifier_decision, proxy_levels, split_data  # noqa: E402

YLABEL = {"DD 1-0": "Demographic disparity", "DI 1/0": "Disparate impact",
          "TPRD 1-0": "TPR difference", "PPVD 1-0": "PPV difference",
          "F1 diff 1-0": "F1 difference"}
BAND = COLORS["grid"]        # the feasible set's colour in the main figure
RULE_A, RULE_B, DIFF = "#0072B2", "#CC79A7", "#009E73"


# ------------------------------------------------------------------------- (a) data
def sensitivity_data(a):
    with open(PREPARED, "rb") as f:
        units = pickle.load(f)
    key = ("model", a.cut_source, a.rep, a.sens_setting, a.sens_proxy, a.run, a.version)
    if key not in units:
        raise SystemExit(f"unit {key} not prepared; run hmda_prepare.py --reps {a.rep}")
    u = units[key]
    r, m = u["r"], u["m"]
    truth = dict(u["truth_pooled"])
    if a.run == "DY":
        kd = key[:5] + ("D",) + key[6:]
        if kd in units:
            truth.update(units[kd]["truth_pooled"])
    J, num = enumerate_joints(r, m, a.nums, a.max_joints)
    J = np.concatenate([J, ci_joint(r, m, harmonize_margins=False)])   # the CI joint (I = 0)
    dep = cond_mi(J, r, m)
    vals = all_metrics(J, a.run)
    if a.metric not in vals:
        raise SystemExit(f"metric {a.metric} not available for run {a.run}")
    v = vals[a.metric]
    ci = float(v[-1])
    qt = u["q_pooled"]
    true_dep = float(cond_mi(qt[None], qt.sum(2), qt.sum(1))[0])
    order = np.argsort(dep)
    ds, vs = dep[order], v[order]
    ok = np.isfinite(vs)
    lo = np.minimum.accumulate(np.where(ok, vs, np.inf))
    hi = np.maximum.accumulate(np.where(ok, vs, -np.inf))
    side = np.sign(ci - a.target)
    reach = np.isfinite(v) & ((np.sign(v - a.target) != side) | (np.abs(v - a.target) <= a.tol))
    brk = float(dep[reach].min()) if reach.any() else np.nan
    return dict(dep=ds, lo=lo, hi=hi, ci=ci, truth=float(truth.get(a.metric, np.nan)),
                true_dep=true_dep, brk=brk, max_dep=float(dep.max()), num=num, n=len(J))


# ------------------------------------------------------------------------- (b) data
def compare_data(a):
    d = load_hmda(a.repo)
    split = split_data(d, a.rep)
    dA, _ = classifier_decision(d, split)
    groups = GROUPS[a.cmp_setting]
    if a.b in ("threshold", "proxy", "proxy-swap"):
        county_bin, _ = proxy_levels(d, split, groups, a.cut_source)["county"]
        dB = adjusted_b(d, split, dA, a.b, county_bin, 1, a.shift, a.frac)
    else:
        dB = classifier_b(d, split, a.b)
    keep = d.race.isin(groups).to_numpy()
    acode = d.race.map({g: j for j, g in enumerate(groups)}).to_numpy()
    z_all, K = proxy_levels(d, split, groups, a.cut_source)[a.cmp_proxy]
    G = len(groups)

    def table(sel):                     # internal cells (D_A, D_B) in the order of hmda_questions
        q = np.zeros((K, 4, G))
        np.add.at(q, (z_all[sel], 2 * dA[sel] + dB[sel], acode[sel].astype(int)), 1.0)
        return q / q.sum()

    s_int, s_ext = keep & (split == "internal"), keep & (split == "external")
    q_int, q_ext = table(s_int), table(s_ext)
    n_int, n_ext = s_int.sum(), s_ext.sum()
    q_pool = (q_int * n_int + q_ext * n_ext) / (n_int + n_ext)
    r, m = (q_pool.sum(2), q_pool.sum(1)) if a.version == "exact" else \
        harmonize(q_int.sum(2), q_ext.sum(1))

    def rule_metric(Q, pos):
        Qc = np.zeros(Q.shape[:2] + (2,) + Q.shape[3:])
        for i, c in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            Qc[:, :, c[pos]] += Q[:, :, i]
        return dict(metrics(Qc, "D"))[a.cmp_metric]

    J, num = enumerate_joints(r, m, a.nums, a.max_joints)
    A, B = rule_metric(J, 0), rule_metric(J, 1)
    ok = np.isfinite(A) & np.isfinite(B)
    return dict(A=A[ok], B=B[ok], tA=float(rule_metric(q_pool[None], 0)[0]),
                tB=float(rule_metric(q_pool[None], 1)[0]), num=num, n=int(ok.sum()))


# ------------------------------------------------------------------------- plotting
def interval(ax, x, lo, hi, color):
    mid = (lo + hi) / 2
    ax.errorbar(x, mid, yerr=[[mid - lo], [hi - mid]], fmt="none", ecolor=color,
                elinewidth=1.3, capsize=3, capthick=1.3, zorder=3)


def star(ax, x, y):
    ax.plot(x, y, "*", ms=8.5, color=TRUTH, mec=INK, mew=0.5, zorder=6)


def plot(sens, cmp_, a, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(a.width, a.height), layout="constrained",
                             gridspec_kw=dict(width_ratios=[1.45, 0.7, 0.45]))
    for ax in axes:
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=GRID, lw=0.6)

    # (a) sensitivity to conditional independence
    ax = axes[0]
    ax.axhline(a.target, color=REF, lw=0.8, zorder=1)
    ax.fill_between(sens["dep"], sens["lo"], sens["hi"], step="post", color=BAND, alpha=0.3,
                    lw=0)
    for y in ("lo", "hi"):
        ax.plot(sens["dep"], sens[y], color=BAND, lw=1.1, drawstyle="steps-post")
    if np.isfinite(sens["brk"]):
        ax.axvline(sens["brk"], color=INK, lw=0.8, ls="--", zorder=2)
    ax.plot(0, sens["ci"], "o", ms=3.8, mfc="white", mec=INK, mew=0.8, zorder=6)
    star(ax, sens["true_dep"], sens["truth"])
    ax.set_xscale("symlog", linthresh=a.linthresh)
    ax.set_xlim(0, sens["max_dep"] * 1.15)
    ax.set_xlabel(r"Maximum dependence $\delta$ (bits)")
    ax.set_ylabel(YLABEL.get(a.metric, a.metric))
    ax.set_title("(a) Sensitivity to cond. independence", loc="left", pad=4)

    # (b) separate bounds of the two rules ...
    ax = axes[1]
    ax.axhline(0, color=REF, lw=0.8, zorder=1)
    A, B = cmp_["A"], cmp_["B"]
    D = A - B
    for x, (lo, hi, t, col) in enumerate([(A.min(), A.max(), cmp_["tA"], RULE_A),
                                          (B.min(), B.max(), cmp_["tB"], RULE_B)]):
        interval(ax, x, lo, hi, col)
        star(ax, x, t)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Classifier", "Adjusted"], fontsize=7)
    ax.tick_params(axis="x", length=2.5)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel(YLABEL.get(a.cmp_metric, a.cmp_metric))
    ax.set_title("(b) Classifier comparison", loc="left", pad=4)

    # ... and the bound on their difference, on its own zoomed scale
    ax = axes[2]
    lo, hi, t = D.min(), D.max(), cmp_["tA"] - cmp_["tB"]
    span = max(hi, 0) - min(lo, 0)
    ax.set_ylim(min(lo, 0) - 0.25 * span, max(hi, 0) + 0.25 * span)
    ax.axhspan(lo, hi, color=DIFF, alpha=0.12, lw=0, zorder=0)
    ax.axhline(0, color=INK, lw=0.9, ls=":", zorder=1)
    mid = (lo + hi) / 2
    ax.errorbar(0, mid, yerr=[[mid - lo], [hi - mid]], fmt="none", ecolor=DIFF,
                elinewidth=2.2, capsize=5, capthick=2.2, zorder=3)
    star(ax, 0, t)
    ax.set_xticks([0])
    ax.set_xticklabels(["Difference\n(zoomed)"], fontsize=7, color=DIFF, fontweight="bold")
    ax.tick_params(axis="x", length=2.5)
    ax.set_xlim(-0.6, 0.6)
    ax.yaxis.tick_right()
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(DIFF)
        ax.spines[side].set_linewidth(1.0)
    ax.yaxis.grid(False)

    h = [Line2D([], [], color=BAND, lw=5, alpha=0.4, label="Bounds"),
         Line2D([], [], ls="", marker="o", ms=3.8, mfc="white", mec=INK, mew=0.8,
                label="Conditional independence"),
         Line2D([], [], color=INK, lw=0.8, ls="--", label="Breakdown point"),
         Line2D([], [], ls="", marker="*", ms=8.5, color=TRUTH, mec=INK, mew=0.5,
                label="True value")]
    fig.legend(handles=h, loc="outside upper center", ncol=len(h), frameon=False,
               handlelength=1.4, columnspacing=1.2)
    for ext_ in ("png", "pdf"):
        fig.savefig(f"{path}.{ext_}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"saved {path}.png and .pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="~/kallus")
    ap.add_argument("--cut-source", default="external", choices=["external", "full"])
    ap.add_argument("--version", default="split", choices=["split", "exact"])
    ap.add_argument("--rep", type=int, default=0)
    # (a)
    ap.add_argument("--sens-setting", default="Black-White", choices=list(GROUPS))
    ap.add_argument("--sens-proxy", default="county", choices=["county", "income", "both"])
    ap.add_argument("--run", default="DY", choices=["D", "DY"])
    ap.add_argument("--metric", default="DD 1-0")
    ap.add_argument("--target", type=float, default=0.0, help="e.g. 0.8 with --metric 'DI 1/0'")
    ap.add_argument("--tol", type=float, default=0.005, help="as in hmda_questions.py")
    ap.add_argument("--linthresh", type=float, default=1e-3, help="symlog x-axis of panel (a)")
    # (b)
    ap.add_argument("--cmp-setting", default="Black-White", choices=list(GROUPS))
    ap.add_argument("--cmp-proxy", default="both", choices=["county", "income", "both"])
    ap.add_argument("--cmp-metric", default="DD 1-0")
    ap.add_argument("--b", default="proxy",
                    choices=["lender", "income", "county", "threshold", "proxy", "proxy-swap"])
    ap.add_argument("--shift", type=float, default=0.02)
    ap.add_argument("--frac", type=float, default=0.25)
    # shared
    ap.add_argument("--nums", type=int, nargs="+", default=[2, 3, 4, 5, 10, 20])
    ap.add_argument("--max-joints", type=int, default=2_000_000)
    ap.add_argument("--width", type=float, default=7.0)
    ap.add_argument("--height", type=float, default=3.4, help="figure height in inches")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    a.repo = os.path.expanduser(a.repo)

    sens = sensitivity_data(a)
    print(f"(a) {sens['n']:,} joints (num={sens['num']}); CI {sens['ci']:.4f}, truth "
          f"{sens['truth']:.4f}; breakdown {sens['brk']:.4f} bits, true dependence "
          f"{sens['true_dep']:.4f}, largest {sens['max_dep']:.4f}")
    cmp_ = compare_data(a)
    A, B = cmp_["A"], cmp_["B"]
    print(f"(b) {cmp_['n']:,} joints (num={cmp_['num']}); classifier [{A.min():.3f}, "
          f"{A.max():.3f}], adjusted [{B.min():.3f}, {B.max():.3f}], difference "
          f"[{(A - B).min():.3f}, {(A - B).max():.3f}] (true {cmp_['tA'] - cmp_['tB']:.3f})")

    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)
    tag = (f"{a.cut_source}_{a.version}_rep{a.rep}_{a.sens_setting}-{a.sens_proxy}-"
           f"{a.metric.split()[0]}_{a.cmp_setting}-{a.cmp_proxy}-B-{a.b}")
    plot(sens, cmp_, a, os.path.join(a.out, "figures", f"extensions_{tag}"))


if __name__ == "__main__":
    main()