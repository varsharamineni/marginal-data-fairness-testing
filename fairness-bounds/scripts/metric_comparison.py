"""Which fairness metrics can be bounded: Kallus closed form vs the feasible-joints method.

The Kallus et al. closed form covers DD, TPRD and TNRD. The joints method evaluates any
function of the joint P(A, Z, Yhat, Y), so every metric below gets a range over the
feasible joints. For the Kallus metrics the two are compared; for all metrics the value
under the true A is shown.

Usage:  python scripts/metric_comparison.py [--num-synthetic 3] [--num-german 5]
                                            [--out-dir outputs/metrics]
"""
import argparse
import os
import sys

import matplotlib
import numpy as np
import pandas as pd
import yaml
from scipy.special import rel_entr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fairness import classifier, data, methods  # noqa: E402


# ------------------------------------------------------------------ metrics of a joint
def group_table(joint, a_levels):
    """T[g, yhat, y] = P(A = a_levels[g], Yhat = yhat, Y = y)."""
    t = joint.groupby(["a", "yhat", "y"])["p"].sum()
    return np.array([[[t.get((g, k, c), 0.0) for c in (0, 1)] for k in (0, 1)] for g in a_levels])


def _rates(T):
    P = T.sum(axis=(1, 2))
    return {"pos": T[:, 1, :].sum(1) / P,                               # P(Yhat=1 | A)
            "tpr": T[:, 1, 1] / T[:, :, 1].sum(1),                      # P(Yhat=1 | Y=1, A)
            "tnr": T[:, 0, 0] / T[:, :, 0].sum(1),                      # P(Yhat=0 | Y=0, A)
            "acc": (T[:, 0, 0] + T[:, 1, 1]) / P,                       # P(Yhat=Y | A)
            "ppv": T[:, 1, 1] / T[:, 1, :].sum(1)}                      # P(Y=1 | Yhat=1, A)


def _mi(P):
    """Mutual information of a 2-D joint probability table (nats)."""
    return rel_entr(P, P.sum(1, keepdims=True) * P.sum(0, keepdims=True)).sum()


def _cmi(T):
    """I(A; Yhat | Y) = sum_y P(Y=y) I(A; Yhat | Y=y)."""
    return sum(T[:, :, c].sum() * _mi(T[:, :, c] / T[:, :, c].sum()) for c in (0, 1))


def _kl(T):
    p, q = T[0] / T[0].sum(), T[1] / T[1].sum()
    return rel_entr(p, q).sum()


def _js(T):
    p, q = (T[0] / T[0].sum()).ravel(), (T[1] / T[1].sum()).ravel()
    m = (p + q) / 2
    return 0.5 * rel_entr(p, m).sum() + 0.5 * rel_entr(q, m).sum()


# name: (function of T, has a Kallus closed form, description)
METRICS = {
    "DD":        (lambda T: np.subtract(*_rates(T)["pos"]), True, "P(Ŷ=1|a) − P(Ŷ=1|b)"),
    "TPRD":      (lambda T: np.subtract(*_rates(T)["tpr"]), True, "P(Ŷ=1|Y=1,a) − P(Ŷ=1|Y=1,b)"),
    "TNRD":      (lambda T: np.subtract(*_rates(T)["tnr"]), True, "P(Ŷ=0|Y=0,a) − P(Ŷ=0|Y=0,b)"),
    "DI ratio":  (lambda T: np.divide(*_rates(T)["pos"]), False, "P(Ŷ=1|a) / P(Ŷ=1|b)"),
    "ACC diff":  (lambda T: np.subtract(*_rates(T)["acc"]), False, "P(Ŷ=Y|a) − P(Ŷ=Y|b)"),
    "PPV diff":  (lambda T: np.subtract(*_rates(T)["ppv"]), False, "P(Y=1|Ŷ=1,a) − P(Y=1|Ŷ=1,b)"),
    "EO gap":    (lambda T: max(abs(np.subtract(*_rates(T)["tpr"])), abs(np.subtract(*_rates(T)["tnr"]))),
                  False, "max(|TPRD|, |FPRD|)"),
    "KL":        (_kl, False, "KL(P(Ŷ,Y|a) ‖ P(Ŷ,Y|b))"),
    "JS":        (_js, False, "JS(P(Ŷ,Y|a), P(Ŷ,Y|b))"),
    "I(A;Ŷ)":    (lambda T: _mi(T.sum(2)), False, "mutual information of A and Ŷ"),
    "I(A;Ŷ|Y)":  (_cmi, False, "Σ_y P(Y=y) I(A;Ŷ | Y=y)"),
}


def all_metrics(joint, a_levels):
    T = group_table(joint, a_levels)
    with np.errstate(divide="ignore", invalid="ignore"):
        return [f(T) for f, _, _ in METRICS.values()]


def true_joint(primary, a_true):
    """The joint computed with the true A, in the same format as the feasible joints."""
    counts = primary.assign(a=np.asarray(a_true)).groupby(["a", "z", "yhat", "y"]).size()
    return (counts / len(primary)).rename("p").reset_index()


# ------------------------------------------------------------------ data
def synthetic(n_z=3, n=20_000, seed=0):
    cfg = dict(seed=seed, n_primary=n, n_auxiliary=n, n_z=n_z, p_z=None, p_a_given_z=None,
               y_intercept=0.0, y_coef_a=0.5, y_coef_z=1.0, x1_coef_y=1.5, x1_coef_a=0.5, x2_coef_y=1.0)
    prim, aux = data.generate_synthetic(cfg)
    prim["yhat"] = (prim["x1"] > 0.75).astype(int)
    return prim.drop(columns="a"), aux, prim["a"].to_numpy(), ["a", "b"]


def from_config(path):
    """Primary/auxiliary data as run.py builds them (csv source with a hidden true A)."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    a_levels = list(cfg["a_levels"])
    prim, aux = data.load_csv(cfg["data"]["csv"])
    prim, aux = data.prepare(prim, aux, cfg["columns"], a_levels, true_a=cfg["data"]["csv"]["true_a"])
    if "yhat" not in prim.columns:
        prim, _ = classifier.add_yhat(prim, cfg["columns"]["features"], cfg["classifier"])
    return prim.drop(columns="a_true"), aux, prim["a_true"].to_numpy(), a_levels


# ------------------------------------------------------------------ experiment
def compare(name, prim, aux, a_true, a_levels, num):
    joints = methods.feasible_joints(prim, aux, a_levels, method="grid", num=num)
    vals = pd.DataFrame(methods.joint_metrics(joints, lambda j: all_metrics(j, a_levels)),
                        columns=list(METRICS))
    truth = all_metrics(true_joint(prim, a_true), a_levels)
    rows = []
    for (m, (_, kallus, desc)), t in zip(METRICS.items(), truth):
        v = vals[m].replace([np.inf, -np.inf], np.nan)
        row = {"dataset": name, "metric": m, "definition": desc, "kallus_closed_form": kallus,
               "joint_min": vals[m].min(), "joint_max": vals[m].max(), "truth": t,
               "n_undefined": int(v.isna().sum())}
        if kallus:
            row["kallus_lower"], row["kallus_upper"] = methods.kallus_bounds(prim, aux, a_levels, m)
        rows.append(row)
    print(f"{name}: {len(joints)} feasible joints, Z levels {prim['z'].nunique()}")
    return pd.DataFrame(rows), vals


def plot(name, table, vals, path):
    blue, orange, ink, muted, grid_c = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a63", "#e4e3dc"
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "axes.titlecolor": ink, "xtick.color": muted, "ytick.color": muted,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "legend.frameon": False})
    metrics = list(METRICS)
    ncol = 4
    fig, axes = plt.subplots(int(np.ceil(len(metrics) / ncol)), ncol, figsize=(13, 7.5),
                             constrained_layout=True)
    for ax, m in zip(axes.ravel(), metrics):
        r = table.set_index("metric").loc[m]
        v = vals[m].to_numpy(dtype=float)
        finite = v[np.isfinite(v)]
        ax.hist(finite, bins=40, color=blue, edgecolor="white", linewidth=0.5)
        ax.axvline(r["truth"], color=ink, lw=2)
        if r["kallus_closed_form"]:
            for x in (r["kallus_lower"], r["kallus_upper"]):
                ax.axvline(x, color=orange, lw=1.5, ls="--")
        title = f"{m}   {'Kallus + joints' if r['kallus_closed_form'] else 'joints only'}"
        if np.isinf(v).any():
            title += f"\n{np.isinf(v).sum()} joints give ∞ (max = ∞)"
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.set_yticks([])
        ax.grid(axis="x", color=grid_c, lw=0.6)
    for ax in axes.ravel()[len(metrics):]:
        ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=blue), plt.Line2D([], [], color=ink, lw=2),
               plt.Line2D([], [], color=orange, lw=1.5, ls="--")]
    axes.ravel()[-1].legend(handles, ["feasible joints (count)", "truth (true A)", "Kallus bounds"],
                            loc="center", fontsize=9)
    fig.suptitle(f"{name}: metric values over the feasible joint distributions", x=0.01, ha="left",
                 fontsize=11, color=ink)
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-synthetic", type=int, default=3, help="grid values per free parameter")
    ap.add_argument("--num-german", type=int, default=5)
    ap.add_argument("--out-dir", default="outputs/metrics")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tables = []
    # num chosen so each grid has ~10k joints (synthetic: 3 Z levels, german: 2)
    for name, (prim, aux, a_true, a_levels), num in [
            ("synthetic", synthetic(), args.num_synthetic),
            ("german", from_config("configs/german.yaml"), args.num_german)]:
        table, vals = compare(name, prim, aux, a_true, a_levels, num)
        vals.to_csv(os.path.join(args.out_dir, f"{name}_joint_values.csv"), index_label="joint")
        plot(name, table, vals, os.path.join(args.out_dir, f"{name}.png"))
        tables.append(table)
    table = pd.concat(tables)
    table.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(table.drop(columns="definition").round(4).to_string(index=False))
