"""
Two questions answered from the grid's joints for one HMDA setting.

Part 1, sensitivity: how far from conditional independence must the truth be?
  The conditional-independence (CI) estimate assumes race carries no information about the
  internal variable beyond the common variable: I(internal; A | Z) = 0. Every enumerated joint
  has its own value delta = I(internal; A | Z) (bits). For a range of delta, the bounds on each
  metric over the joints with I <= delta are reported: at delta = 0 they reduce to the CI point
  (the CI joint is added to the set), at the largest delta they are the full grid bounds.
  The breakdown point of a metric is the smallest delta at which its interval reaches a target
  (parity, or 0.8 for disparate impact): the least departure from CI under which the target
  cannot be ruled out (within --tol, needed for boundary targets such as worst-case DI = 1).
  The true dependence (from the withheld race labels) is marked.

Part 2, comparison: is classifier A fairer than classifier B?
  Both classifiers decide on the same applicants, whose race is unknown in the same way, so the
  internal variable is the pair (D_A, D_B) (with --with-y also Y). Every enumerated joint gives
  both classifiers' metrics at once, which bounds their difference A - B. The difference can
  exclude 0 even when the separate intervals of A and B overlap.
  A = the classifier of hmda_prepare.py. B (--b):
    lender      the lender's decisions
    income      logistic regression on log income only (train split)
    county      logistic regression on the smoothed county approval rate only (train split)
    threshold   A's model with a lower threshold: approves --shift more of the internal
                applicants (e.g. 0.02 = two percentage points more)
    proxy       A, plus approving the highest-scoring --frac of the applicants A denies in the
                high-minority-share county bin (a proxy-based "debiasing" adjustment)
    proxy-swap  as proxy, and additionally denying the same number of the lowest-scoring
                approved applicants in the other bins, so the overall approval rate is unchanged
  For lender, income and county, B's threshold matches the lender's approval rate on the internal
  split, as for A. The county bins used by proxy and proxy-swap are those of --proxy county.

The grid is an inner approximation: an interval is attained by actual joints; breakdown points
are upper bounds on the true breakdown (a finer grid can only lower them).

Usage
  python hmda_questions.py                                  # both parts, Black-White, county
  python hmda_questions.py --part sensitivity --run D --setting API-White
  python hmda_questions.py --part compare --b income --proxy both
Outputs in <out>/questions/: sensitivity_<tag>.csv, breakdown_<tag>.csv, compare_<tag>.csv and
figures/sensitivity_<tag>.png/.pdf, figures/compare_<tag>.png/.pdf
"""
import argparse
import contextlib
import io
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from hmda_common import OUT, PREPARED
from ci_baseline import ci_joint, harmonize  # noqa: E402
from grid_joint import solve_joint_distribution  # noqa: E402
from exp0_hmda import load_hmda  # noqa: E402
from synth import metrics  # noqa: E402
from hmda_prepare import GROUPS, classifier_decision, proxy_levels, split_data  # noqa: E402

LABEL = {"DD 1-0": "DD", "DI 1/0": "DI", "worst-case DI": "worst-case DI",
         "max DD gap": "max DD gap", "I(D;A)": r"$I(\hat{Y};A)$", "TPRD 1-0": "TPR diff.",
         "PPVD 1-0": "PPV diff.", "BA diff 1-0": "BA diff.", "F1 diff 1-0": "F1 diff.",
         "I(D;A|Y)": r"$I(\hat{Y};A\mid Y)$"}
TARGETS = {"DD 1-0": [0.0], "DI 1/0": [1.0, 0.8], "worst-case DI": [1.0, 0.8],
           "max DD gap": [0.0], "TPRD 1-0": [0.0], "PPVD 1-0": [0.0], "BA diff 1-0": [0.0],
           "F1 diff 1-0": [0.0]}


# ------------------------------------------------------------------------- shared
def enumerate_joints(r, m, nums, max_joints, method="sequential"):
    K, I = r.shape
    G = m.shape[1]
    for num in sorted(nums, reverse=True):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                J, _ = solve_joint_distribution(
                    {(i, z): r[z, i] for z in range(K) for i in range(I)},
                    {(z, g): m[z, g] for z in range(K) for g in range(G)},
                    num=num, method=method, max_joints=max_joints, as_dicts=False)
            return J, num
        except SystemExit:
            continue
    raise SystemExit(f"no num in {nums} stays under {max_joints:,} joints")


def cond_mi(Q, r, m):
    """I(internal; A | Z) in bits for each joint Q[n, z, i, a] with margins r[z, i], m[z, a]."""
    pz = r.sum(1)
    expected = np.where(pz[:, None, None] > 0,
                        r[:, :, None] * m[:, None, :] / np.where(pz > 0, pz, 1)[:, None, None], 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where((Q > 0) & (expected[None] > 0), Q * np.log2(Q / expected[None]), 0.0)
    return np.maximum(t.sum(axis=(1, 2, 3)), 0.0)


def all_metrics(Q, run):
    vals = dict(metrics(Q, run))
    if run == "DY":
        QD = np.stack([Q[:, :, 2] + Q[:, :, 3], Q[:, :, 0] + Q[:, :, 1]], axis=2)
        vals.update(metrics(QD, "D"))
    return vals


def stem(a, part):
    if part == "compare":
        par = (f"-{a.shift}" if a.b == "threshold" else
               f"-{a.frac}" if a.b in ("proxy", "proxy-swap") else "")
        extra = f"_B-{a.b}{par}{'-Y' if a.with_y else ''}"
    else:
        extra = f"_{a.run}"
    return (f"{part}_{a.decision}_{a.cut_source}_{a.setting}_{a.proxy}_{a.version}{extra}"
            f"_rep{a.rep}")


# ------------------------------------------------------------------------- part 1
def sensitivity(a, out_dir, fig_dir, plt):
    with open(PREPARED, "rb") as f:
        units = pickle.load(f)
    key = (a.decision, a.cut_source, a.rep, a.setting, a.proxy, a.run, a.version)
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
    ci_vals = {k: v[-1] for k, v in vals.items()}
    true_dep = None
    if "q_pooled" in u:      # the truth's own dependence, measured against the unit's margins
        qt = u["q_pooled"]
        rt, mt = qt.sum(2), qt.sum(1)
        true_dep = float(cond_mi(qt[None], rt, mt)[0])
    print(f"part 1: {len(J):,} joints (num={num} plus the CI joint); I(internal; A | Z) up to "
          f"{dep.max():.4f} bits" + (f"; truth {true_dep:.4f}" if true_dep is not None else ""))

    names = [n for n in (a.metrics or list(vals)) if n in vals]
    deltas = np.unique(np.concatenate([[0.0], np.quantile(dep, np.linspace(0, 1, a.points)),
                                       np.geomspace(max(dep[dep > 0].min(), 1e-6), dep.max(),
                                                    a.points)]))
    order = np.argsort(dep)
    dep_s = dep[order]
    rows = []
    for n in names:
        v = vals[n][order]
        ok = np.isfinite(v)
        # running min / max over joints sorted by dependence
        lo_run = np.minimum.accumulate(np.where(ok, v, np.inf))
        hi_run = np.maximum.accumulate(np.where(ok, v, -np.inf))
        idx = np.searchsorted(dep_s, deltas, side="right") - 1
        for d_, i_ in zip(deltas, idx):
            rows.append(dict(metric=n, delta=d_, lower=lo_run[i_], upper=hi_run[i_]))
    curve = pd.DataFrame(rows)
    curve.to_csv(os.path.join(out_dir, stem(a, "sensitivity") + ".csv"), index=False)

    brk = []
    for n in names:
        v = vals[n]
        for t in TARGETS.get(n, []):
            side = np.sign(ci_vals[n] - t)
            reach = np.isfinite(v) & ((np.sign(v - t) != side) | (np.abs(v - t) <= a.tol))
            d_star = float(dep[reach].min()) if reach.any() else np.nan
            brk.append(dict(metric=n, target=t, ci_value=ci_vals[n], truth=truth.get(n, np.nan),
                            breakdown_delta=d_star, truth_delta=true_dep,
                            full_lower=np.nanmin(v), full_upper=np.nanmax(v)))
    brk = pd.DataFrame(brk)
    brk.to_csv(os.path.join(out_dir, stem(a, "breakdown") + ".csv"), index=False)
    with pd.option_context("display.width", 200):
        print(brk.round(4).to_string(index=False))

    k = len(names)
    cols = min(k, 4)
    rows_ = int(np.ceil(k / cols))
    fig, axes = plt.subplots(rows_, cols, figsize=(3.4 * cols, 2.8 * rows_), squeeze=False)
    for ax, n in zip(axes.flat, names):
        c = curve[curve.metric == n]
        ax.fill_between(c.delta, c.lower, c.upper, color="tab:orange", alpha=0.35, lw=0,
                        step="post")
        ax.plot(c.delta, c.lower, color="tab:orange", lw=1.2, drawstyle="steps-post")
        ax.plot(c.delta, c.upper, color="tab:orange", lw=1.2, drawstyle="steps-post")
        ax.plot(0, ci_vals[n], "o", color="tab:cyan", mec="k", mew=0.5, ms=5, zorder=5)
        if true_dep is not None:
            ax.plot(true_dep, truth.get(n, np.nan), "x", color="tab:red", ms=8, mew=2, zorder=6)
        for t in TARGETS.get(n, []):
            ax.axhline(t, color="grey", lw=0.7, ls=":")
            b = brk[(brk.metric == n) & (brk.target == t)].breakdown_delta
            if len(b) and np.isfinite(b.iloc[0]):
                ax.axvline(b.iloc[0], color="tab:purple", lw=1, ls="--")
        ax.set_xscale("symlog", linthresh=1e-3)
        ax.set_xlim(-2e-4, dep.max() * 1.15)
        ax.set_title(LABEL.get(n, n), fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[k:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel(r"$\delta$ = max. $I(\mathrm{internal};A\mid Z)$ (bits)", fontsize=8)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color="tab:orange", lw=6, alpha=0.35, label="bounds with I <= delta"),
               Line2D([], [], color="tab:cyan", marker="o", mec="k", ls="", label="CI estimate"),
               Line2D([], [], color="tab:red", marker="x", ls="", mew=2,
                      label="true value (at its dependence)"),
               Line2D([], [], color="tab:purple", ls="--", label="breakdown point")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    s = os.path.join(fig_dir, stem(a, "sensitivity"))
    for ext in ("png", "pdf"):
        fig.savefig(f"{s}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {s}.png/.pdf and the csv files in {out_dir}")


# ------------------------------------------------------------------------- part 2
def model_score(d, split):
    """Scores of A's model (same features and fit as hmda_prepare.classifier_decision)."""
    tr = split == "train"
    y = d.outcome.to_numpy(int)
    prior, k = y[tr].mean(), 20.0
    st = d[tr].groupby("county_id").outcome.agg(["sum", "count"])
    rate = d.county_id.map((st["sum"] + k * prior) / (st["count"] + k)).fillna(prior)
    X = np.column_stack([np.log1p(d.income), rate])
    return LogisticRegression().fit(X[tr], y[tr]).predict_proba(X)[:, 1]


def adjusted_b(d, split, dA, kind, county_bin, high_bin, shift, frac):
    """B derived from A's model: threshold shift, or proxy-based adjustments."""
    score = model_score(d, split)
    internal = split == "internal"
    if kind == "threshold":
        rate = dA[internal].mean() + shift
        t = np.quantile(score[internal], 1 - rate)
        return (score >= t).astype(int)
    dB = dA.copy()
    hi = county_bin == high_bin
    # approve the top `frac` of A's denials in the high-minority bin (cut-off set on internal)
    denied_hi = hi & (dA == 0)
    t_hi = np.quantile(score[denied_hi & internal], 1 - frac)
    dB[denied_hi & (score >= t_hi)] = 1
    if kind == "proxy-swap":
        n_add = int((denied_hi & internal & (score >= t_hi)).sum())
        approved_lo = ~hi & (dA == 1)
        q = n_add / max(int((approved_lo & internal).sum()), 1)
        t_lo = np.quantile(score[approved_lo & internal], q)
        dB[approved_lo & (score < t_lo)] = 0
    return dB


def classifier_b(d, split, kind):
    y = d.outcome.to_numpy(int)
    if kind == "lender":
        return y
    tr = split == "train"
    if kind == "income":
        X = np.log1p(d.income.to_numpy())[:, None]
    else:
        prior, k = y[tr].mean(), 20.0
        st = d[tr].groupby("county_id").outcome.agg(["sum", "count"])
        X = d.county_id.map((st["sum"] + k * prior) / (st["count"] + k)).fillna(prior)
        X = X.to_numpy()[:, None]
    score = LogisticRegression().fit(X[tr], y[tr]).predict_proba(X)[:, 1]
    internal = split == "internal"
    t = np.quantile(score[internal], 1 - y[internal].mean())
    return (score >= t).astype(int)


def compare(a, out_dir, fig_dir, plt):
    d = load_hmda(a.repo)
    split = split_data(d, a.rep)
    dA, _ = classifier_decision(d, split)
    groups = GROUPS[a.setting]
    if a.b in ("threshold", "proxy", "proxy-swap"):
        county_bin, _ = proxy_levels(d, split, groups, a.cut_source)["county"]
        dB = adjusted_b(d, split, dA, a.b, county_bin, 1, a.shift, a.frac)
    else:
        dB = classifier_b(d, split, a.b)
    internal = split == "internal"
    print(f"B = {a.b}: agrees with A on {(dA[internal] == dB[internal]).mean():.3f} of internal "
          f"applicants; approval rate A {dA[internal].mean():.3f}, B {dB[internal].mean():.3f}")
    keep = d.race.isin(groups).to_numpy()
    acode = d.race.map({g: j for j, g in enumerate(groups)}).to_numpy()
    z_all, K = proxy_levels(d, split, groups, a.cut_source)[a.proxy]
    y = d.outcome.to_numpy(int)
    G = len(groups)
    cells = ([(da, db, yy) for da in (0, 1) for db in (0, 1) for yy in (0, 1)] if a.with_y
             else [(da, db) for da in (0, 1) for db in (0, 1)])
    I = len(cells)
    index = {c: i for i, c in enumerate(cells)}

    def table(sel):
        q = np.zeros((K, I, G))
        ii = np.array([index[(da, db, yy) if a.with_y else (da, db)]
                       for da, db, yy in zip(dA[sel], dB[sel], y[sel])])
        np.add.at(q, (z_all[sel], ii, acode[sel].astype(int)), 1.0)
        return q / q.sum()

    q_int = table(keep & (split == "internal"))
    q_ext = table(keep & (split == "external"))
    n_int, n_ext = (keep & (split == "internal")).sum(), (keep & (split == "external")).sum()
    q_pool = (q_int * n_int + q_ext * n_ext) / (n_int + n_ext)
    r, m = (q_pool.sum(2), q_pool.sum(1)) if a.version == "exact" else \
        harmonize(q_int.sum(2), q_ext.sum(1))

    def split_metrics(Q):
        """Metrics of A and of B from joints over (D_A, D_B[, Y])."""
        out = {}
        for who, pos in (("A", 0), ("B", 1)):
            if a.with_y:      # -> TP, FP, FN, TN for this classifier
                agg = {(1, 1): 0, (1, 0): 1, (0, 1): 2, (0, 0): 3}
                Qc = np.zeros(Q.shape[:2] + (4,) + Q.shape[3:])
                for c, i in index.items():
                    Qc[:, :, agg[(c[pos], c[2])]] += Q[:, :, i]
                out[who] = all_metrics(Qc, "DY")
            else:
                Qc = np.zeros(Q.shape[:2] + (2,) + Q.shape[3:])
                for c, i in index.items():
                    Qc[:, :, c[pos]] += Q[:, :, i]
                out[who] = dict(metrics(Qc, "D"))
        return out

    J, num = enumerate_joints(r, m, a.nums, a.max_joints)
    mv = split_metrics(J)
    tv = split_metrics(q_pool[None])
    print(f"part 2: A = classifier, B = {a.b}; internal variable "
          f"{'(D_A, D_B, Y)' if a.with_y else '(D_A, D_B)'}; {len(J):,} joints at num={num}")
    rows = []
    names = [n for n in (a.metrics or list(mv["A"])) if n in mv["A"]]
    for n in names:
        A, B = mv["A"][n], mv["B"][n]
        ok = np.isfinite(A) & np.isfinite(B)
        diff = A[ok] - B[ok]
        tA, tB = float(tv["A"][n][0]), float(tv["B"][n][0])
        rows.append(dict(metric=n, A_lower=np.nanmin(A), A_upper=np.nanmax(A), truth_A=tA,
                         B_lower=np.nanmin(B), B_upper=np.nanmax(B), truth_B=tB,
                         diff_lower=diff.min(), diff_upper=diff.max(), truth_diff=tA - tB,
                         separate_overlap=bool(np.nanmax(A) >= np.nanmin(B)
                                               and np.nanmax(B) >= np.nanmin(A)),
                         diff_excludes_0=bool(diff.min() > 0 or diff.max() < 0)))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, stem(a, "compare") + ".csv"), index=False)
    with pd.option_context("display.width", 220):
        print(res.round(4).to_string(index=False))

    k = len(names)
    fig, axes = plt.subplots(1, k, figsize=(2.3 * k, 3.2), squeeze=False)
    for ax, row in zip(axes[0], rows):
        for x, (lo, hi, t, col) in enumerate([
                (row["A_lower"], row["A_upper"], row["truth_A"], "tab:orange"),
                (row["B_lower"], row["B_upper"], row["truth_B"], "tab:blue"),
                (row["diff_lower"], row["diff_upper"], row["truth_diff"], "tab:purple")]):
            ax.plot([x, x], [lo, hi], color=col, lw=2.5)
            ax.plot([x - 0.12, x + 0.12], [lo, lo], color=col, lw=1.5)
            ax.plot([x - 0.12, x + 0.12], [hi, hi], color=col, lw=1.5)
            ax.plot(x, t, "x", color="tab:red", ms=7, mew=2, zorder=5)
        ax.axhline(0, color="grey", lw=0.7, ls=":")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["A", "B", "A $-$ B"], fontsize=8)
        ax.set_xlim(-0.5, 2.5)
        ax.set_title(LABEL.get(row["metric"], row["metric"]), fontsize=9)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"Classifier A vs B = {a.b} ({a.setting}, {a.proxy}); "
                 f"{len(J):,} joints, num = {num}", fontsize=9)
    fig.tight_layout()
    s = os.path.join(fig_dir, stem(a, "compare"))
    for ext in ("png", "pdf"):
        fig.savefig(f"{s}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {s}.png/.pdf and the csv in {out_dir}")


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="both", choices=["sensitivity", "compare", "both"])
    ap.add_argument("--repo", default="~/kallus", help="HMDA repo (part 2 rebuilds its tables)")
    ap.add_argument("--decision", default="model", choices=["model", "lender"],
                    help="part 1 only")
    ap.add_argument("--cut-source", default="external", choices=["external", "full"])
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--setting", default="Black-White", choices=list(GROUPS))
    ap.add_argument("--proxy", default="county", choices=["county", "income", "both"])
    ap.add_argument("--version", default="split", choices=["split", "exact"])
    ap.add_argument("--run", default="DY", choices=["D", "DY"], help="part 1 only")
    ap.add_argument("--b", default="lender",
                    choices=["lender", "income", "county", "threshold", "proxy", "proxy-swap"],
                    help="part 2: the classifier compared with ours")
    ap.add_argument("--shift", type=float, default=0.02,
                    help="part 2, --b threshold: extra share of applicants approved")
    ap.add_argument("--frac", type=float, default=0.25,
                    help="part 2, --b proxy/proxy-swap: share of A's denials approved in the "
                         "high-minority bin")
    ap.add_argument("--with-y", action="store_true",
                    help="part 2: include Y (TPR, PPV, ... differences; 8 internal cells)")
    ap.add_argument("--nums", type=int, nargs="+", default=[2, 3, 4, 5, 10, 20])
    ap.add_argument("--max-joints", type=int, default=2_000_000)
    ap.add_argument("--metrics", nargs="+", default=None)
    ap.add_argument("--points", type=int, default=60, help="part 1: number of delta values")
    ap.add_argument("--tol", type=float, default=0.005,
                    help="part 1: a target counts as reached within this distance (needed for "
                         "targets on the boundary, e.g. worst-case DI = 1, max gap = 0)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    a.repo = os.path.expanduser(a.repo)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = os.path.join(a.out, "questions")
    fig_dir = os.path.join(a.out, "figures")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    if a.part in ("sensitivity", "both"):
        sensitivity(a, out_dir, fig_dir, plt)
    if a.part in ("compare", "both"):
        compare(a, out_dir, fig_dir, plt)


if __name__ == "__main__":
    main()