"""Audit verdicts for common fairness thresholds when A is only in the auxiliary data.

Rules (thresholds as used in practice):
    Four-fifths rule       min/max selection rate >= 0.8   (EEOC Uniform Guidelines, 29 CFR 1607.4(D);
                                                          NYC Local Law 144 reports this impact ratio)
    SPD                    |P(Yhat=1|a) - P(Yhat=1|b)| <= 0.1           (AIF360 fair range)
    Equal opp. diff        |TPR_a - TPR_b| <= 0.1                       (AIF360 fair range)
    Equal opp. ratio       min/max TPR >= 0.9  (TPRs within 10%)
    Average odds diff      |(TPRD + FPRD) / 2| <= 0.1                   (AIF360 fair range)
    Equalized odds         max(|TPRD|, |FPRD|) <= 0.1
    FPR / FNR / PPV ratio  min/max >= 0.8                               (Aequitas default tau = 0.8)
    AIF360 bundle          four-fifths, SPD, equal opp. diff and average odds all pass
    Aequitas bundle        selection, FPR, FNR and PPV ratios all >= 0.8

Verdict of a method:  pass  - every joint it allows passes
                      fail  - no joint it allows passes
                      undetermined - some pass, some fail
Methods compared:
    joints    the feasible joints (grid); the rule is evaluated on each joint
    separate  Kallus per-group rate bounds combined by interval arithmetic
              (the only option when each quantity is bounded separately; n/a for PPV)
    CI        the single joint implied by conditional independence of A and (Yhat, Y) given Z
    truth     the joint with the true A

A sweep over synthetic data where Z predicts A with strength s (0 = not at all, 1 = Z
determines A) shows when each method reaches a verdict.

Usage:  python scripts/audit_rules.py [--strengths 0 0.5 0.9 1] [--out-dir outputs/audit]
"""
import argparse
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fairness import methods  # noqa: E402
from metric_comparison import from_config, group_table, synthetic, true_joint  # noqa: E402


# ------------------------------------------------------------------ group rates
def rates(T):
    """Per-group rates for T[n, a, yhat, y]; each value has shape (n, 2)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        tpr = T[:, :, 1, 1] / T[:, :, :, 1].sum(2)
        fpr = T[:, :, 1, 0] / T[:, :, :, 0].sum(2)
        return {"sr": T[:, :, 1, :].sum(2) / T.sum((2, 3)), "tpr": tpr, "fpr": fpr,
                "fnr": 1 - tpr, "ppv": T[:, :, 1, 1] / T[:, :, 1, :].sum(2)}


def _ratio(r):
    with np.errstate(divide="ignore", invalid="ignore"):
        return r.min(1) / r.max(1)


def _diff(r):
    return r[:, 0] - r[:, 1]


# name: (statistic from rates, passes(stat), rule text); every statistic lies in [0, 1]
RULES = {
    "Four-fifths rule":   (lambda R: _ratio(R["sr"]), lambda s: s >= 0.8, "selection ratio ≥ 0.8"),
    "SPD":                (lambda R: abs(_diff(R["sr"])), lambda s: s <= 0.1, "|DD| ≤ 0.1"),
    "Equal opp. diff":    (lambda R: abs(_diff(R["tpr"])), lambda s: s <= 0.1, "|TPRD| ≤ 0.1"),
    "Equal opp. ratio":   (lambda R: _ratio(R["tpr"]), lambda s: s >= 0.9, "TPR ratio ≥ 0.9"),
    "Average odds diff":  (lambda R: abs(_diff(R["tpr"]) + _diff(R["fpr"])) / 2, lambda s: s <= 0.1,
                           "|(TPRD+FPRD)/2| ≤ 0.1"),
    "Equalized odds":     (lambda R: np.maximum(abs(_diff(R["tpr"])), abs(_diff(R["fpr"]))),
                           lambda s: s <= 0.1, "max(|TPRD|,|FPRD|) ≤ 0.1"),
    "FPR ratio":          (lambda R: _ratio(R["fpr"]), lambda s: s >= 0.8, "FPR ratio ≥ 0.8"),
    "FNR ratio":          (lambda R: _ratio(R["fnr"]), lambda s: s >= 0.8, "FNR ratio ≥ 0.8"),
    "PPV ratio":          (lambda R: _ratio(R["ppv"]), lambda s: s >= 0.8, "PPV ratio ≥ 0.8"),
}
BUNDLES = {"AIF360 bundle": ["Four-fifths rule", "SPD", "Equal opp. diff", "Average odds diff"],
           "Aequitas bundle": ["Four-fifths rule", "FPR ratio", "FNR ratio", "PPV ratio"]}


def passes(T):
    """Pass/fail (NaN where undefined) of every rule and bundle for each joint in T."""
    R = rates(T)
    stats = pd.DataFrame({r: f(R) for r, (f, _, _) in RULES.items()})
    ok = pd.DataFrame({r: np.where(stats[r].isna(), np.nan, RULES[r][1](stats[r]))
                       for r in RULES})
    for b, rs in BUNDLES.items():
        ok[b] = ok[rs].min(axis=1, skipna=False)
    return stats, ok


def verdict(ok):
    ok = ok.dropna()
    if len(ok) == 0:
        return "undefined"
    return "pass" if ok.all() else "fail" if not ok.any() else "undetermined"


# ------------------------------------------------------------------ separate (Kallus) bounds
def separate_rate_bounds(prim, aux, a_levels):
    """[lower, upper] of each group's rate from the Kallus weights, bounded one rate at a time."""
    yhat, y, pa_a, pa_b = methods._inputs(prim, aux, a_levels)
    p1 = methods._cell_prob_rows(prim, ["yhat"])
    p2 = methods._cell_prob_rows(prim, ["yhat", "y"])
    out = {}
    for g, pa in enumerate((pa_a, pa_b)):
        wL, wU = methods.frechet_weights(pa, p1)
        out[("sr", g)] = (methods.mu(wL, yhat, pa.mean()), methods.mu(wU, yhat, pa.mean()))
        wL, wU = methods.frechet_weights(pa, p2)
        tpr = (methods.mu_prime(wL, wU, yhat, y, 1, 1), methods.mu_prime(wU, wL, yhat, y, 1, 1))
        tnr = (methods.mu_prime(wL, wU, yhat, y, 0, 0), methods.mu_prime(wU, wL, yhat, y, 0, 0))
        out[("tpr", g)], out[("fnr", g)] = tpr, (1 - tpr[1], 1 - tpr[0])
        out[("fpr", g)] = (1 - tnr[1], 1 - tnr[0])
    return out


def _ratio_iv(a, b):
    """Range of min(x, y) / max(x, y) for x in a, y in b (positive intervals)."""
    lo = min(a[0] / b[1], b[0] / a[1]) if min(a[1], b[1]) > 0 else 0.0
    overlap = a[0] <= b[1] and b[0] <= a[1]
    hi = 1.0 if overlap else (a[1] / b[0] if a[1] < b[0] else b[1] / a[0])
    return lo, hi


def _diff_iv(a, b):
    return a[0] - b[1], a[1] - b[0]


def _abs_iv(d):
    return (0.0 if d[0] <= 0 <= d[1] else min(abs(d[0]), abs(d[1]))), max(abs(d[0]), abs(d[1]))


def separate_verdicts(B):
    """Interval-arithmetic range of each rule's statistic and the verdict it allows."""
    d = {k: _diff_iv(B[(k, 0)], B[(k, 1)]) for k in ("sr", "tpr", "fpr")}
    aod = _abs_iv(((d["tpr"][0] + d["fpr"][0]) / 2, (d["tpr"][1] + d["fpr"][1]) / 2))
    eo = tuple(np.maximum(_abs_iv(d["tpr"]), _abs_iv(d["fpr"])))
    iv = {"Four-fifths rule": _ratio_iv(B[("sr", 0)], B[("sr", 1)]), "SPD": _abs_iv(d["sr"]),
          "Equal opp. diff": _abs_iv(d["tpr"]),
          "Equal opp. ratio": _ratio_iv(B[("tpr", 0)], B[("tpr", 1)]),
          "Average odds diff": aod, "Equalized odds": eo,
          "FPR ratio": _ratio_iv(B[("fpr", 0)], B[("fpr", 1)]),
          "FNR ratio": _ratio_iv(B[("fnr", 0)], B[("fnr", 1)])}
    out = {}
    for r, (lo, hi) in iv.items():
        test = RULES[r][1]
        out[r] = ("pass" if test(lo) and test(hi) else
                  "fail" if not test(lo) and not test(hi) else "undetermined", (lo, hi))
    for b, rs in BUNDLES.items():
        vs = [out[r][0] if r in out else "n/a" for r in rs]
        out[b] = ("fail" if "fail" in vs else "n/a" if "n/a" in vs else
                  "pass" if all(v == "pass" for v in vs) else "undetermined", None)
    return out


def ci_table(prim, aux, a_levels):
    """T for the joint under conditional independence: P(A, Z, O) = P(Z, O) P(A | Z)."""
    p_zo = prim.groupby(["z", "yhat", "y"]).size().rename("p_zo").reset_index()
    p_zo["p_zo"] /= len(prim)
    pa = methods.p_a_given_z(aux, a_levels)
    joint = pd.concat([p_zo.assign(a=g, p=p_zo["p_zo"] * p_zo["z"].map(pa[g])) for g in a_levels])
    return group_table(joint, a_levels)[None]


# ------------------------------------------------------------------ experiment
def audit(name, prim, aux, a_true, a_levels, num):
    joints = methods.feasible_joints(prim, aux, a_levels, method="grid", num=num)
    stats, ok = passes(methods.joint_metrics(joints, lambda j: group_table(j, a_levels)))
    t_stats, t_ok = passes(group_table(true_joint(prim, a_true), a_levels)[None])
    c_stats, c_ok = passes(ci_table(prim, aux, a_levels))
    sep = separate_verdicts(separate_rate_bounds(prim, aux, a_levels))
    rows = []
    for r in list(RULES) + list(BUNDLES):
        row = {"dataset": name, "rule": r, "threshold": RULES[r][2] if r in RULES else " & ".join(BUNDLES[r]),
               "joints": verdict(ok[r]), "separate": sep.get(r, ("n/a",))[0],
               "CI": verdict(c_ok[r]), "truth": verdict(t_ok[r]),
               "pct_joints_pass": 100 * ok[r].mean(), "n_undefined": int(ok[r].isna().sum())}
        if r in RULES:
            row |= {"joint_min": stats[r].min(), "joint_max": stats[r].max(),
                    "truth_stat": t_stats[r].iloc[0], "ci_stat": c_stats[r].iloc[0]}
            if r in sep:
                row["separate_min"], row["separate_max"] = sep[r][1]
        rows.append(row)
    print(f"{name}: {len(joints)} feasible joints")
    return pd.DataFrame(rows)


def sweep_data(s, n=20_000, seed=0):
    """Synthetic data with 3 Z levels; s in [0, 1] sets how strongly Z predicts A:
    P(A=a | Z) = [0.5 - s/2, 0.5 + s/2, 0.5 - s/2]  (s = 1: Z determines A)."""
    from fairness import data
    cfg = dict(seed=seed, n_primary=n, n_auxiliary=n, n_z=3, p_z=None,
               p_a_given_z=[0.5 - s / 2, 0.5 + s / 2, 0.5 - s / 2], y_intercept=0.0, y_coef_a=0.5,
               y_coef_z=1.0, x1_coef_y=1.5, x1_coef_a=0.5, x2_coef_y=1.0)
    prim, aux = data.generate_synthetic(cfg)
    prim["yhat"] = (prim["x1"] > 0.75).astype(int)
    return prim.drop(columns="a"), aux, prim["a"].to_numpy(), ["a", "b"]


def sweep(strengths, num):
    return pd.concat([audit(f"s={s}", *sweep_data(s), num).assign(s=s) for s in strengths])


# ------------------------------------------------------------------ figure
BLUE, GREEN_BG, INK, MUTED, ORANGE, GRID = "#2a78d6", "#d5f0e3", "#1f1f1e", "#6b6a63", "#eb6834", "#e4e3dc"
VERDICT_STYLE = {"pass": ("#0ca30c", "✓", "white"), "fail": ("#d03b3b", "✗", "white"),
                 "undetermined": ("#ecebe6", "?", MUTED), "n/a": ("white", "–", MUTED),
                 "undefined": ("white", "–", MUTED)}


def plot_sweep(table, path):
    rules = list(RULES) + list(BUNDLES)
    strengths = sorted(table["s"].unique())
    panels = [("truth", "Truth (true A)"), ("CI", "CI point estimate  (bold = wrong)"),
              ("separate", "Separate bounds"), ("joints", "Feasible joints")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 5.2), sharey=True, constrained_layout=True)
    for ax, (col, title) in zip(axes, panels):
        for i, r in enumerate(rules):
            for j, s in enumerate(strengths):
                row = table[(table["rule"] == r) & (table["s"] == s)].iloc[0]
                v = row[col]
                face, sym, txt = VERDICT_STYLE[v]
                wrong = col == "CI" and v != row["truth"]
                ax.add_patch(plt.Rectangle((j - 0.46, i - 0.44), 0.92, 0.88, color=face,
                                           ec=INK if wrong else "white", lw=2 if wrong else 0))
                ax.text(j, i, sym, ha="center", va="center", color=txt, fontsize=10,
                        fontweight="bold" if wrong else "normal")
        ax.set_xlim(-0.5, len(strengths) - 0.5)
        ax.set_ylim(len(rules) - 0.5, -0.5)
        ax.set_xticks(range(len(strengths)), [f"{s:g}" for s in strengths])
        ax.set_yticks(range(len(rules)), rules)
        ax.set_xlabel("strength s of Z → A  (1 = Z determines A)")
        ax.set_title(title, loc="left")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=VERDICT_STYLE[v][0], ec=MUTED, lw=0.5)
               for v in ("pass", "fail", "undetermined", "n/a")]
    fig.legend(handles, ["✓ certain pass", "✗ certain fail", "? undetermined", "– not available"],
               loc="outside lower center", ncol=4, fontsize=9)
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


def plot(tables, path):
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "legend.frameon": False})
    fig, axes = plt.subplots(1, len(tables), figsize=(7.2 * len(tables), 5.2), constrained_layout=True)
    for ax, t in zip(np.atleast_1d(axes), tables):
        t = t[t["rule"].isin(RULES)].reset_index(drop=True)
        for i, r in t.iterrows():
            y = len(t) - 1 - i
            lo_pass = RULES[r["rule"]][1](0.0)                         # pass region on the left?
            thr = float(r["threshold"].split()[-1])
            ax.axhspan(y - 0.4, y + 0.4, xmin=0 if lo_pass else thr, xmax=thr if lo_pass else 1,
                       color=GREEN_BG, lw=0)
            if not np.isnan(r.get("separate_min", np.nan)):
                ax.plot([r["separate_min"], r["separate_max"]], [y + 0.22] * 2, color=MUTED, lw=1.5,
                        ls=(0, (2, 1.5)), solid_capstyle="butt")
            ax.plot([r["joint_min"], r["joint_max"]], [y] * 2, color=BLUE, lw=6, solid_capstyle="butt")
            ax.plot(r["ci_stat"], y, "o", color=ORANGE, ms=7, mec="white", mew=1)
            ax.plot(r["truth_stat"], y, "D", color=INK, ms=6, mec="white", mew=1)
            ax.text(1.02, y, f"joints: {r['joints']}   (truth: {r['truth']})", va="center",
                    fontsize=8, color=INK, transform=ax.get_yaxis_transform())
        ax.set_yticks(range(len(t))[::-1], [f"{r}\n{th}" for r, th in zip(t["rule"], t["threshold"])],
                      fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.6, len(t) - 0.4)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("rule statistic")
        ax.set_title(t["dataset"].iloc[0], loc="left", fontsize=11)
    handles = [plt.Line2D([], [], color=BLUE, lw=6), plt.Line2D([], [], color=MUTED, lw=1.5, ls=(0, (2, 1.5))),
               plt.Rectangle((0, 0), 1, 1, color=GREEN_BG),
               plt.Line2D([], [], marker="D", color=INK, lw=0, ms=6),
               plt.Line2D([], [], marker="o", color=ORANGE, lw=0, ms=7)]
    fig.legend(handles, ["range over feasible joints", "separate bounds (interval arithmetic)",
                         "passes the rule", "truth (true A)", "CI point estimate"],
               loc="outside lower center", ncol=5, fontsize=8.5)
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-synthetic", type=int, default=3)
    ap.add_argument("--num-german", type=int, default=5)
    ap.add_argument("--strengths", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0])
    ap.add_argument("--out-dir", default="outputs/audit")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tables = [audit("synthetic", *synthetic(), args.num_synthetic),
              audit("german", *from_config("configs/german.yaml"), args.num_german)]
    table = pd.concat(tables)
    table.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    plot(tables, os.path.join(args.out_dir, "audit.png"))
    pd.set_option("display.width", 250)
    print(table[["dataset", "rule", "joints", "separate", "CI", "truth", "pct_joints_pass",
                 "joint_min", "joint_max", "separate_min", "separate_max", "truth_stat"]]
          .round(3).to_string(index=False))

    sw = sweep(args.strengths, args.num_synthetic)
    sw.to_csv(os.path.join(args.out_dir, "sweep.csv"), index=False)
    plot_sweep(sw, os.path.join(args.out_dir, "sweep.png"))
    print(sw.pivot(index="rule", columns="s", values="joints").to_string())
    print(sw.pivot(index="rule", columns="s", values="separate").to_string())
