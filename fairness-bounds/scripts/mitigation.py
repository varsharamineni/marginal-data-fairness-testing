"""Before vs after bias mitigation: bound the change in fairness metrics between two classifiers.

Both classifiers are scored on the same individuals, so they share the same unknown A.
The joints method uses one joint P(A, Z, Yhat_before, Yhat_after, Y)
(feasible_joints with outcome=("yhat1", "yhat2", "y")) and evaluates both models on it:

    delta m    = m(after) - m(before)           change in the disparity
    delta |m|  = |m(after)| - |m(before)|       change in unfairness (< 0: mitigation helped)

Separate bounds (the alternative): bound each model on its own with Kallus and subtract,
    delta m in [lo_after - hi_before, hi_after - lo_before],
which ignores that the two models share A.

Mitigation is certified to help when max delta |m| < 0 over everything the method allows.

Datasets
  synthetic  2 Z levels, P(A=a | Z) = [0.5 - s/2, 0.5 + s/2], swept over s:
             before Yhat1 = 1{x1 > 0.75} (x1 depends on A), after Yhat2 = 1{x2 > 0.5}
  german     configs/german.yaml: before = config classifier plus `age` (a proxy for sex),
             after = the config classifier (age dropped); same split, same held-out rows

Joints: grid (num=2) plus random samples. Checks: each model's own range over the joints vs
its Kallus bounds (coverage), and the exact range of delta DD (it is linear in the joint).

Usage:  python scripts/mitigation.py [--datasets synthetic german german_methods] [--out-dir outputs/mitigation]
"""
import argparse
import os
import sys

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fairness import classifier, data, methods  # noqa: E402

METRICS = ("DD", "TPRD", "TNRD")
OUTCOME = ("yhat1", "yhat2", "y")


# ------------------------------------------------------------------ data
def synthetic(s, n=20_000, seed=0):
    cfg = dict(seed=seed, n_primary=n, n_auxiliary=n, n_z=2, p_z=None,
               p_a_given_z=[0.5 - s / 2, 0.5 + s / 2], y_intercept=0.0, y_coef_a=0.5, y_coef_z=1.0,
               x1_coef_y=1.5, x1_coef_a=0.5, x2_coef_y=1.0)
    prim, aux = data.generate_synthetic(cfg)
    prim["yhat1"] = (prim["x1"] > 0.75).astype(int)
    prim["yhat2"] = (prim["x2"] > 0.5).astype(int)
    return prim.drop(columns="a"), aux, prim["a"].to_numpy(), ["a", "b"]


def german(config="configs/german.yaml", proxy="age"):
    """German credit data: before = config classifier plus the sex proxy `proxy`,
    after = the config classifier (proxy dropped). Same train split and held-out rows."""
    with open(config) as f:
        cfg = yaml.safe_load(f)
    feats, a_levels = list(cfg["columns"]["features"]), list(cfg["a_levels"])
    cols = dict(cfg["columns"], features=feats + [proxy])
    prim, aux = data.load_csv(cfg["data"]["csv"])
    prim, aux = data.prepare(prim, aux, cols, a_levels, true_a=cfg["data"]["csv"]["true_a"])
    before, info_b = classifier.add_yhat(prim, feats + [proxy], cfg["classifier"])
    after, info_a = classifier.add_yhat(prim, feats, cfg["classifier"])
    assert (before["y"] == after["y"]).all() and (before["z"] == after["z"]).all()   # same rows
    print(f"german: before {feats + [proxy]} acc {info_b['held_out_accuracy']:.3f}, "
          f"after {feats} acc {info_a['held_out_accuracy']:.3f}, {len(before)} held-out rows")
    held = before.rename(columns={"yhat": "yhat1"}).assign(yhat2=after["yhat"].to_numpy())
    return held.drop(columns="a_true"), aux, held["a_true"].to_numpy(), a_levels


# ------------------------------------------------------------------ mitigation methods (German)
def fit_rate_threshold(scores, rate):
    """(t, pi) such that 'score > t, or score == t with probability pi' selects `rate` of scores."""
    s = np.sort(np.asarray(scores))[::-1]
    k = rate * len(s)
    if k <= 0:
        return np.inf, 0.0
    t = s[min(int(np.ceil(k)) - 1, len(s) - 1)]
    above, at = (s > t).sum(), (s == t).sum()
    return t, float(np.clip((k - above) / at, 0, 1))


def apply_threshold(scores, t, pi, rng):
    scores = np.asarray(scores)
    return ((scores > t) | ((scores == t) & (rng.random(len(scores)) < pi))).astype(int)


def german_mitigations(config="configs/german.yaml", proxy="age"):
    """Before model and four mitigated models on the German data.

    Like the auditor, the developer never sees A: only the features, Y and Z on the training
    rows, plus the separate auxiliary marginal P(A | Z). Every model approves the same share of
    training rows as the before model (same approval budget), so methods differ in who is
    approved, not how many. Returns (primary with yhat1 = before and one yhat_<method> column
    per method, auxiliary, true A of the held-out rows, a_levels, info table).
    """
    from sklearn.linear_model import LogisticRegression

    with open(config) as f:
        cfg = yaml.safe_load(f)
    feats, a_levels, ccfg = list(cfg["columns"]["features"]), list(cfg["a_levels"]), cfg["classifier"]
    cols = dict(cfg["columns"], features=feats + [proxy])
    prim, aux = data.load_csv(cfg["data"]["csv"])
    prim, aux = data.prepare(prim, aux, cols, a_levels, true_a=cfg["data"]["csv"]["true_a"])
    idx = np.random.default_rng(ccfg["seed"]).permutation(len(prim))  # same split as classifier.add_yhat
    n_train = int(round(ccfg["train_fraction"] * len(prim)))
    tr, te = prim.iloc[idx[:n_train]], prim.iloc[idx[n_train:]].reset_index(drop=True)
    rng = np.random.default_rng(ccfg["seed"])

    def lr(features, weight=None):
        m = LogisticRegression(max_iter=1000).fit(tr[features], tr["y"], sample_weight=weight)
        return m.predict_proba(tr[features])[:, 1], m.predict_proba(te[features])[:, 1]

    full = feats + [proxy]
    s_tr, s_te = lr(full)
    before = (s_te >= ccfg["threshold"]).astype(int)
    budget = float((s_tr >= ccfg["threshold"]).mean())               # approval rate of before model

    out = {}
    # 1. drop the proxy (fairness through unawareness), same budget
    d_tr, d_te = lr(feats)
    out["drop age"] = apply_threshold(d_te, *fit_rate_threshold(d_tr, budget), rng)
    # 2. proxy reweighing: Kamiran & Calders w(a, y) = P(a) P(y) / P(a, y), with each training
    #    row's unknown A replaced by P(A | Z) from the auxiliary data (soft group membership)
    pa_z = methods.p_a_given_z(aux, a_levels).loc[tr["z"]].to_numpy()     # rows x a_levels
    y1 = tr["y"].to_numpy()[:, None]
    pa, py1 = pa_z.mean(0), y1.mean()
    pay = np.stack([(pa_z * (1 - y1)).mean(0), (pa_z * y1).mean(0)])     # P^(y, a)
    py = np.where(y1 == 1, py1, 1 - py1)                                 # P(y_i)
    w = (pa_z * pa * py / pay[y1[:, 0]]).sum(1)
    r_tr, r_te = lr(full, w)
    out["proxy reweighing"] = apply_threshold(r_te, *fit_rate_threshold(r_tr, budget), rng)
    # 3. Z thresholds, demographic parity: each Z level approved at the budget rate
    zt = np.zeros(len(te), int)
    for z in te["z"].unique():
        zt[te["z"] == z] = apply_threshold(s_te[te["z"] == z],
                                           *fit_rate_threshold(s_tr[(tr["z"] == z).to_numpy()], budget), rng)
    out["Z thr. (DP)"] = zt
    # 4. Z thresholds, equal opportunity: each Z level's TPR set to the before model's TPR
    tpr = float((s_tr[tr["y"] == 1] >= ccfg["threshold"]).mean())
    ze = np.zeros(len(te), int)
    for z in te["z"].unique():
        pos = ((tr["z"] == z) & (tr["y"] == 1)).to_numpy()
        ze[te["z"] == z] = apply_threshold(s_te[te["z"] == z], *fit_rate_threshold(s_tr[pos], tpr), rng)
    out["Z thr. (EOpp)"] = ze

    held = te.assign(yhat1=before)
    info = [{"method": "before", "accuracy": (before == te["y"]).mean(), "approval_rate": before.mean()}]
    for name, yh in out.items():
        held[f"yhat_{name}"] = yh
        info.append({"method": name, "accuracy": (yh == te["y"]).mean(), "approval_rate": yh.mean()})
    return held.drop(columns="a_true"), aux, held["a_true"].to_numpy(), a_levels, pd.DataFrame(info)


# ------------------------------------------------------------------ metrics of a joint
def disparities(T):
    """DD, TPRD, TNRD from T[a, yhat, y] (joint probabilities)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        pos = T[:, 1, :].sum(1) / T.sum((1, 2))
        tpr = T[:, 1, 1] / T[:, :, 1].sum(1)
        tnr = T[:, 0, 0] / T[:, :, 0].sum(1)
    return {"DD": pos[0] - pos[1], "TPRD": tpr[0] - tpr[1], "TNRD": tnr[0] - tnr[1]}


def joint_values(joint, a_levels):
    """Both models' disparities on one joint, and their changes."""
    t = joint.groupby(["a", *OUTCOME])["p"].sum()
    P = np.array([[[[t.get((g, i, j, c), 0.0) for c in (0, 1)] for j in (0, 1)] for i in (0, 1)]
                  for g in a_levels])                                   # P[a, yhat1, yhat2, y]
    before, after = disparities(P.sum(2)), disparities(P.sum(1))
    out = {}
    for m in METRICS:
        out[f"{m}_before"], out[f"{m}_after"] = before[m], after[m]
        out[f"d{m}"] = after[m] - before[m]
        out[f"d|{m}|"] = abs(after[m]) - abs(before[m])
    return out


def exact_dd_change(prim, aux, a_levels):
    """Exact range of DD(after) - DD(before). It is linear in q = P(A=a, Yhat1, Yhat2 | Z), so
    each Z level is filled greedily: mass of A=a goes first to cells where Yhat2 - Yhat1 is
    largest (for the max) or smallest (for the min)."""
    pa = methods.p_a_given_z(aux, a_levels)[a_levels[0]]
    P_a = prim["z"].map(pa).mean()
    cells = prim.groupby(["z", "yhat1", "yhat2"]).size() / len(prim)
    out = []
    for sign in (-1, 1):
        lin = const = 0.0
        for z, g in cells.groupby(level=0):
            c = sign * np.array([y2 - y1 for _, y1, y2 in g.index])
            p, budget, q = g.to_numpy(), pa[z] * g.sum(), np.zeros(len(g))
            for i in np.argsort(-c):
                q[i] = min(p[i], budget)
                budget -= q[i]
            lin, const = lin + (c * q).sum(), const + (c * p).sum()
        out.append(sign * (lin * (1 / P_a + 1 / (1 - P_a)) - const / (1 - P_a)))
    return tuple(out)


def _abs_iv(lo, hi):
    return (0.0 if lo <= 0 <= hi else min(abs(lo), abs(hi))), max(abs(lo), abs(hi))


# ------------------------------------------------------------------ experiment
def run(setting, prim, aux, a_true, a_levels, num, n_random):
    joints = (methods.feasible_joints(prim, aux, a_levels, outcome=OUTCOME, method="grid", num=num)
              + methods.feasible_joints(prim, aux, a_levels, outcome=OUTCOME, method="random",
                                        n_samples=n_random))
    vals = pd.DataFrame(methods.joint_metrics(joints, lambda j: joint_values(j, a_levels)).tolist())
    truth_joint = (prim.assign(a=a_true).groupby(["a", "z", *OUTCOME]).size() / len(prim)
                   ).rename("p").reset_index()
    truth = joint_values(truth_joint, a_levels)
    rows = []
    for m in METRICS:
        kb = methods.kallus_bounds(prim.assign(yhat=prim["yhat1"]), aux, a_levels, m)
        ka = methods.kallus_bounds(prim.assign(yhat=prim["yhat2"]), aux, a_levels, m)
        ab, aa = _abs_iv(*kb), _abs_iv(*ka)
        v = vals.replace([np.inf, -np.inf], np.nan)
        row = {"setting": setting, "metric": m, "n_joints": len(vals),
               "truth_before": truth[f"{m}_before"], "truth_after": truth[f"{m}_after"],
               "truth_d": truth[f"d{m}"], "truth_dabs": truth[f"d|{m}|"],
               "joints_d_lo": v[f"d{m}"].min(), "joints_d_hi": v[f"d{m}"].max(),
               "joints_dabs_lo": v[f"d|{m}|"].min(), "joints_dabs_hi": v[f"d|{m}|"].max(),
               "separate_d_lo": ka[0] - kb[1], "separate_d_hi": ka[1] - kb[0],
               "separate_dabs_lo": aa[0] - ab[1], "separate_dabs_hi": aa[1] - ab[0],
               # coverage check: each model alone over the joints vs its Kallus bounds
               "coverage_before": (v[f"{m}_before"].max() - v[f"{m}_before"].min()) / (kb[1] - kb[0]),
               "coverage_after": (v[f"{m}_after"].max() - v[f"{m}_after"].min()) / (ka[1] - ka[0]),
               "n_undefined": int(v[f"d{m}"].isna().sum())}
        if m == "DD":
            row["exact_d_lo"], row["exact_d_hi"] = exact_dd_change(prim, aux, a_levels)
        rows.append(row)
    t = pd.DataFrame(rows)
    t["joints_certify_help"] = t["joints_dabs_hi"] < 0
    t["separate_certify_help"] = t["separate_dabs_hi"] < 0
    print(f"{setting}: {len(vals)} joints")
    return t


# ------------------------------------------------------------------ figure
BLUE, INK, MUTED, GRID = "#2a78d6", "#1f1f1e", "#6b6a63", "#e4e3dc"


def plot(table, path, xlabel, title=None, rotate=0):
    """For each metric: the change interval from each method (signed and in |.|) per setting."""
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False})
    settings = list(dict.fromkeys(table["setting"]))
    fig, axes = plt.subplots(2, len(METRICS), figsize=(13, 7), sharey="row", constrained_layout=True)
    for row, (key, label) in enumerate([("d", "Δ{m} = {m} after − {m} before"),
                                        ("dabs", "Δ|{m}| = |{m} after| − |{m} before|")]):
        for ax, m in zip(axes[row], METRICS):
            t = table[table["metric"] == m].set_index("setting")
            for k, s in enumerate(settings):
                r = t.loc[s]
                ax.plot([k - 0.12] * 2, [r[f"separate_{key}_lo"], r[f"separate_{key}_hi"]], color=MUTED,
                        lw=2, ls=(0, (2, 1.5)), solid_capstyle="butt")
                ax.plot([k + 0.12] * 2, [r[f"joints_{key}_lo"], r[f"joints_{key}_hi"]], color=BLUE, lw=6,
                        solid_capstyle="butt")
                ax.plot(k + 0.12, r[f"truth_{key}"], "D", color=INK, ms=6, mec="white", mew=1)
                if key == "dabs" and r["joints_certify_help"]:
                    ax.annotate("certified", (k + 0.12, r["joints_dabs_hi"]), xytext=(0, 4),
                                textcoords="offset points", ha="center", fontsize=7.5, color=INK)
            ax.axhline(0, color=INK, lw=0.8)
            ax.set_xticks(range(len(settings)), settings, rotation=rotate, ha="right" if rotate else "center")
            ax.set_xlim(-0.6, len(settings) - 0.4)
            ax.set_xlabel(xlabel)
            ax.set_title(label.format(m=m), loc="left")
            ax.grid(axis="y", color=GRID, lw=0.6)
    axes[0, 0].set_ylabel("change in disparity")
    axes[1, 0].set_ylabel("change in unfairness  (< 0: mitigation helped)")
    handles = [plt.Line2D([], [], color=BLUE, lw=6), plt.Line2D([], [], color=MUTED, lw=2, ls=(0, (2, 1.5))),
               plt.Line2D([], [], marker="D", color=INK, lw=0, ms=6)]
    fig.legend(handles, ["feasible joints (shared A)", "separate bounds, subtracted", "truth (true A)"],
               loc="outside lower center", ncol=3, fontsize=9)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["synthetic", "german"])
    ap.add_argument("--strengths", type=float, nargs="+", default=[0.0, 0.5, 0.8, 0.9, 0.95])
    ap.add_argument("--num", type=int, default=2, help="grid values per free parameter")
    ap.add_argument("--n-random", type=int, default=5000, help="random joints added to the grid")
    ap.add_argument("--out-dir", default="outputs/mitigation")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    pd.set_option("display.width", 250)
    cols = ["setting", "metric", "truth_before", "truth_after", "joints_d_lo", "joints_d_hi",
            "separate_d_lo", "separate_d_hi", "exact_d_lo", "exact_d_hi", "joints_dabs_lo",
            "joints_dabs_hi", "separate_dabs_lo", "separate_dabs_hi", "truth_dabs",
            "coverage_before", "coverage_after", "joints_certify_help", "separate_certify_help"]

    if "synthetic" in args.datasets:
        table = pd.concat([run(f"{s:g}", *synthetic(s), args.num, args.n_random) for s in args.strengths])
        table.to_csv(os.path.join(args.out_dir, "synthetic.csv"), index=False)
        plot(table, os.path.join(args.out_dir, "synthetic.png"), "strength s of Z → A")
        print(table[cols].round(3).to_string(index=False))
    if "german" in args.datasets:
        table = run("german", *german(), args.num, args.n_random)
        table.to_csv(os.path.join(args.out_dir, "german.csv"), index=False)
        plot(table, os.path.join(args.out_dir, "german.png"), "",
             "German credit: before = with age (sex proxy), after = age dropped")
        print(table[cols].round(3).to_string(index=False))
    if "german_methods" in args.datasets:
        prim, aux, a_true, a_levels, info = german_mitigations()
        methods_ = [c[len("yhat_"):] for c in prim.columns if c.startswith("yhat_")]
        table = pd.concat([run(m, prim.drop(columns=[f"yhat_{o}" for o in methods_])
                               .assign(yhat2=prim[f"yhat_{m}"]), aux, a_true, a_levels,
                               args.num, args.n_random) for m in methods_])
        table.to_csv(os.path.join(args.out_dir, "german_methods.csv"), index=False)
        info.to_csv(os.path.join(args.out_dir, "german_methods_models.csv"), index=False)
        plot(table, os.path.join(args.out_dir, "german_methods.png"), "mitigation method",
             "German credit: change from the before model (with age) under each mitigation", rotate=20)
        print(info.round(3).to_string(index=False))
        print(table[cols].round(3).to_string(index=False))
