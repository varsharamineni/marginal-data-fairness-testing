"""
Simulation pipeline: how well do different methods recover fairness metrics when the
protected attribute is only available in an auxiliary dataset?

  1. scenario     full data (protected attribute, common/proxy variables, primary-only
                  features, label), split into a training set, a primary dataset and an
                  auxiliary dataset (common + protected only)
  2. classifiers  trained on the training set, giving a decision for every primary row
  3. truth        every fairness metric on the primary rows, using the hidden protected attribute
  4. methods      estimate or bound the metrics from the primary data (decision, label,
                  common) and the auxiliary data (common, protected) only:
                    independence  point estimate assuming (decision, label) independent of
                                  the protected attribute given the common variables
                    joint         grid enumeration (grid_joint.py): bounds on every metric
                    kallus        sharp bounds on DP / TPR / FPR disparities
                                  (kallus_bounds.py), binary protected attribute only
  5. compare      coverage, width and miss distance against the truth (results.csv,
                  summary.csv), method-vs-method containment (method_comparison.csv),
                  and interval plots

Extend by adding entries to SCENARIOS, CLASSIFIERS or METHODS. A scenario is a function
seed -> (full data frame, roles); roles names the columns:
    protected  protected attribute (binary for kallus)
    common     list of proxy variables in both datasets (categorical, or binned with --bins)
    features   list of other variables, primary data only (used by classifiers)
    label      true outcome, coded 0/1
    decision   name for the predicted decision column

Command line:
  python pipeline.py                                   # all scenarios, classifiers, methods
  python pipeline.py --scenarios ci_holds feature_dep --classifiers logit gbm --seeds 0 1 2
  python pipeline.py --csv full.csv --protected race --common income_bin \
      --features credit_score age --label repaid       # your own full data as a scenario
"""
import argparse
import contextlib
import io
import itertools
import os

import numpy as np
import pandas as pd

from fairness_metrics import fairness_bounds
from grid_joint import _discretise, from_datasets, solve_joint_distribution
from kallus_bounds import kallus_bounds

ROLES = dict(protected="race", common=["income_bin"], features=["credit_score", "age"],
             label="repaid", decision="yhat")


# ============================================================================ 1. scenarios
def synthetic(seed=0, n=20000, p_group=0.3, n_bins=3, proxy=1.5, feature_dep=0.0,
              label_dep=0.0):
    """
    race ~ Bernoulli(p_group); income_bin depends on race (proxy: larger = stronger proxy,
    group 1 in lower bins); credit_score depends on income_bin and, with feature_dep, on race
    beyond income_bin; repaid depends on income_bin, credit_score, age and, with label_dep,
    on race directly. With feature_dep = label_dep = 0, everything in the primary data is
    independent of race given income_bin, so the independence estimate is unbiased.
    """
    rng = np.random.default_rng(seed)
    a = (rng.random(n) < p_group).astype(int)
    s = np.arange(n_bins) / max(n_bins - 1, 1)
    P = np.exp(-proxy * a[:, None] * s[None, :])
    P /= P.sum(1, keepdims=True)
    c = (P.cumsum(1) > rng.random(n)[:, None]).argmax(1)
    cs = c / max(n_bins - 1, 1)
    credit = 0.8 * cs - feature_dep * a + rng.normal(0, 1, n)
    age = rng.normal(40, 10, n).round()
    eta = -0.2 + 1.0 * cs + 0.9 * credit + 0.02 * (age - 40) - label_dep * a
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int)
    return pd.DataFrame(dict(race=a, income_bin=c, credit_score=credit, age=age, repaid=y)), \
        dict(ROLES)


SCENARIOS = {
    "ci_holds": lambda seed: synthetic(seed),
    "feature_dep": lambda seed: synthetic(seed, feature_dep=0.8),
    "label_dep": lambda seed: synthetic(seed, label_dep=0.8),
    "weak_proxy": lambda seed: synthetic(seed, proxy=0.3, feature_dep=0.8),
    "strong_proxy": lambda seed: synthetic(seed, proxy=4.0, feature_dep=0.8),
}


def csv_scenario(path, roles):
    """Scenario from a full dataset on disk (the seed changes only the split)."""
    full = pd.read_csv(path)
    return lambda seed: (full, roles)


def split(full, roles, seed, train=0.3, primary=0.4, aux_same_rows=False):
    """Random split into training, primary and auxiliary rows. The primary frame keeps the
    protected attribute for computing the truth; methods never see it."""
    idx = np.random.default_rng(seed + 10_000).permutation(len(full))
    n_tr, n_pr = int(train * len(full)), int(primary * len(full))
    tr, pr = full.iloc[idx[:n_tr]], full.iloc[idx[n_tr:n_tr + n_pr]]
    au = pr if aux_same_rows else full.iloc[idx[n_tr + n_pr:]]
    au = au[roles["common"] + [roles["protected"]]]
    return [d.reset_index(drop=True) for d in (tr, pr, au)]


# ============================================================================ 2. classifiers
def _columns(roles, which):
    return sum(([roles[w]] if isinstance(roles[w], str) else list(roles[w]) for w in which), [])


def _design(df, cols, categorical, columns=None):
    X = pd.get_dummies(df[cols].astype({c: str for c in cols if c in categorical}), dtype=float)
    return X if columns is None else X.reindex(columns=columns, fill_value=0.0)


def classifier(model, which):
    """Classifier on the roles in `which` (e.g. ['features', 'common']), thresholded so that
    the approval rate on the training data is `rate`."""
    def fit_predict(train, primary, roles, rate, seed):
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        cols = _columns(roles, which)
        cat = roles["common"] + [roles["protected"]]
        Xtr = _design(train, cols, cat)
        Xpr = _design(primary, cols, cat, Xtr.columns)
        est = (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
               if model == "logistic" else HistGradientBoostingClassifier(random_state=seed))
        est.fit(Xtr, train[roles["label"]].astype(int))
        thr = np.quantile(est.predict_proba(Xtr)[:, 1], 1 - rate)
        return (est.predict_proba(Xpr)[:, 1] >= thr).astype(int)
    return fit_predict


CLASSIFIERS = {
    "logit": classifier("logistic", ["features", "common"]),
    "gbm": classifier("gbm", ["features", "common"]),
    "logit_proxy_only": classifier("logistic", ["common"]),
    "logit_aware": classifier("logistic", ["features", "common", "protected"]),
}


# ============================================================================ 3. truth
def _key(v):
    return tuple(v) if isinstance(v, tuple) and len(v) > 1 else (v[0] if isinstance(v, tuple) else v)


def truth(primary, roles, internal):
    """Every metric on the primary rows, with their real protected attribute."""
    com, prot = roles["common"], roles["protected"]
    s = primary.groupby(internal + com + [prot], observed=True).size()
    ni, nc = len(internal), len(com)
    cells = {(_key(k[:ni]), _key(k[ni:ni + nc]), k[-1]): v for k, v in s.items()}
    srt = lambda vals: sorted(set(vals), key=str)
    I, C, E = (srt(k[j] for k in cells) for j in range(3))
    J = np.zeros((1, len(C), len(I), len(E)))
    for (i, c, e), v in cells.items():
        J[0, C.index(c), I.index(i), E.index(e)] = v
    J /= J.sum()
    mt, _, _ = fairness_bounds(J, I, C, E, internal, roles["decision"], "1", roles["label"], "1")
    return dict(zip(mt.metric, mt.lower))


# ============================================================================ 4. methods
def tables(p_primary, p_aux):
    """I, C, E, r[c, i] = p(i | c) P(c), m[c, e] = p(c, e), as in grid_joint."""
    srt = lambda vals: sorted(vals, key=str)
    I = srt({k[0] for k in p_primary})
    C = srt({k[0] for k in p_aux})
    E = srt({k[1] for k in p_aux})
    m = np.array([[p_aux.get((c, e), 0) for e in E] for c in C], float)
    pr = np.array([[p_primary.get((i, c), 0) for i in I] for c in C], float)
    keep = pr.sum(1) > 0
    C, m, pr = [c for c, k in zip(C, keep) if k], m[keep], pr[keep]
    m = m / m.sum()
    return I, C, E, pr / pr.sum(1, keepdims=True) * m.sum(1, keepdims=True), m


def method_independence(p_primary, p_aux, internal, roles, cfg):
    I, C, E, r, m = tables(p_primary, p_aux)
    J = ((r / r.sum(1, keepdims=True))[:, :, None] * m[:, None, :])[None]
    mt, _, _ = fairness_bounds(J, I, C, E, internal, roles["decision"], "1", roles["label"], "1")
    return mt[["metric", "lower", "upper"]]


def method_joint(p_primary, p_aux, internal, roles, cfg):
    J, (I, C, E) = solve_joint_distribution(p_primary, p_aux, cfg["num"], cfg["grid_method"],
                                            max_joints=cfg["max_joints"], as_dicts=False)
    mt, _, _ = fairness_bounds(J, I, C, E, internal, roles["decision"], "1", roles["label"], "1")
    return mt[["metric", "lower", "upper"]]


def method_kallus(p_primary, p_aux, internal, roles, cfg):
    I, C, E, r, m = tables(p_primary, p_aux)
    return kallus_bounds(r, m, I, E, internal, roles["decision"], roles["label"],
                         extended=cfg.get("kallus_extended", False))


METHODS = {"independence": method_independence, "joint": method_joint, "kallus": method_kallus}


# ============================================================================ 5. run + compare
def run(scenarios, classifiers, methods, seeds, cfg, verbose=False):
    rows = []
    for sname, seed in itertools.product(scenarios, seeds):
        full, roles = SCENARIOS[sname](seed)
        full = _discretise(full.copy(), full.iloc[:0].copy(), roles["common"],
                           cfg["max_categories"], cfg["bins"])[0]
        train, primary, aux = split(full, roles, seed, cfg["train"], cfg["primary"],
                                    cfg["aux_same_rows"])
        dec, lab, com, prot = roles["decision"], roles["label"], roles["common"], roles["protected"]
        for cname in classifiers:
            print(f"{sname} | seed {seed} | {cname}")
            primary[dec] = CLASSIFIERS[cname](train, primary, roles, cfg["rate"], seed)
            quiet = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
            with quiet:
                p_primary, p_aux, internal, _, _ = from_datasets(
                    primary[[dec, lab] + com], aux, [dec, lab], com, [prot])
            true = truth(primary, roles, internal)
            for mname in methods:
                try:
                    with quiet:
                        res = METHODS[mname](p_primary, p_aux, internal, roles, cfg)
                except SystemExit as e:
                    print(f"    {mname} skipped: {e}")
                    continue
                for r in res.itertuples(index=False):
                    rows.append(dict(scenario=sname, seed=seed, classifier=cname, method=mname,
                                     metric=r.metric, lower=r.lower, upper=r.upper,
                                     truth=true.get(r.metric, np.nan)))
    res = pd.DataFrame(rows)
    tol = cfg["tol"]
    res["width"] = res.upper - res.lower
    res["covered"] = (res.lower - tol <= res.truth) & (res.truth <= res.upper + tol)
    res["miss"] = np.maximum(0, np.maximum(res.lower - res.truth, res.truth - res.upper))
    res["midpoint_error"] = (res.lower + res.upper) / 2 - res.truth
    return res


def summarise(res):
    return (res.groupby(["scenario", "method", "metric"])
            .agg(coverage=("covered", "mean"), mean_width=("width", "mean"),
                 mean_miss=("miss", "mean"), max_miss=("miss", "max"),
                 mean_midpoint_error=("midpoint_error", "mean"), runs=("covered", "size"))
            .reset_index())


def compare_methods(res, tol=1e-6):
    """For every pair of methods: is method's interval inside the reference's, and by how much
    do the endpoints differ (positive lower_diff / negative upper_diff = narrower)."""
    key = ["scenario", "seed", "classifier", "metric"]
    w = res.pivot_table(index=key, columns="method", values=["lower", "upper"])
    out = []
    for a, b in itertools.permutations(res.method.unique(), 2):
        if ("lower", a) not in w or ("lower", b) not in w:
            continue
        la, ua, lb, ub = w["lower"][a], w["upper"][a], w["lower"][b], w["upper"][b]
        ok = la.notna() & lb.notna()
        out.append(pd.DataFrame(dict(method=a, reference=b,
                                     inside=(la >= lb - tol) & (ua <= ub + tol),
                                     lower_diff=la - lb, upper_diff=ua - ub))[ok])
    return pd.concat(out).reset_index() if out else pd.DataFrame()


def plot(res, out_dir, seed=None):
    """One figure per scenario x classifier: each metric's interval per method, truth in black."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seed = res.seed.min() if seed is None else seed
    methods = list(res.method.unique())
    colours = dict(zip(methods, plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    paths = []
    for (s, c), d in res[res.seed == seed].groupby(["scenario", "classifier"]):
        metrics = list(dict.fromkeys(d.metric))
        cols = 4
        rows = int(np.ceil(len(metrics) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 1.0 + 0.45 * len(methods) * rows + rows),
                                 squeeze=False)
        for ax, met in zip(axes.flat, metrics):
            dm = d[d.metric == met].set_index("method")
            for k, mth in enumerate(methods):
                if mth in dm.index:
                    lo, hi = dm.loc[mth, "lower"], dm.loc[mth, "upper"]
                    ax.plot([lo, hi], [k, k], lw=5, color=colours[mth], solid_capstyle="butt")
                    ax.plot([lo, hi], [k, k], "|", ms=10, color=colours[mth])
            t = dm.truth.dropna()
            if len(t):
                ax.axvline(t.iloc[0], color="black", lw=1.2)
            ax.set_yticks(range(len(methods)), methods, fontsize=8)
            ax.set_ylim(-0.7, len(methods) - 0.3)
            ax.set_title(met, fontsize=9)
            ax.tick_params(axis="x", labelsize=8)
        for ax in list(axes.flat)[len(metrics):]:
            ax.axis("off")
        fig.suptitle(f"{s} | {c} | seed {seed}   (black line = true value)", fontsize=11)
        fig.tight_layout()
        p = os.path.join(out_dir, f"intervals_{s}_{c}.png")
        fig.savefig(p, dpi=110)
        plt.close(fig)
        paths.append(p)
    return paths


# ============================================================================ command line
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", default=None, help=f"default: all of {list(SCENARIOS)}")
    ap.add_argument("--classifiers", nargs="+", default=list(CLASSIFIERS), choices=list(CLASSIFIERS))
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--csv", help="full dataset to use as a scenario named 'csv'")
    ap.add_argument("--protected")
    ap.add_argument("--common", nargs="+")
    ap.add_argument("--features", nargs="+", default=[])
    ap.add_argument("--label")
    ap.add_argument("--rate", type=float, default=0.5, help="approval rate of every classifier")
    ap.add_argument("--train", type=float, default=0.3, help="share of rows for training")
    ap.add_argument("--primary", type=float, default=0.4, help="share of rows for primary data")
    ap.add_argument("--aux-same-rows", action="store_true",
                    help="auxiliary = the primary rows (removes sampling differences between them)")
    ap.add_argument("--num", type=int, default=4, help="grid values per free entry (joint)")
    ap.add_argument("--grid-method", choices=["sweep", "sequential"], default="sequential")
    ap.add_argument("--max-joints", type=int, default=2_000_000)
    ap.add_argument("--kallus-extended", action="store_true", help="also PPV disparity")
    ap.add_argument("--max-categories", type=int, default=20)
    ap.add_argument("--bins", type=int, default=3, help="bins for numeric common variables")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default="pipeline_results")
    a = ap.parse_args()

    if a.csv:
        if not (a.protected and a.common and a.label):
            raise SystemExit("--csv needs --protected, --common and --label")
        SCENARIOS["csv"] = csv_scenario(a.csv, dict(protected=a.protected, common=a.common,
                                                     features=a.features, label=a.label,
                                                     decision="yhat"))
    scenarios = a.scenarios or (["csv"] if a.csv else list(SCENARIOS))
    cfg = dict(num=a.num, grid_method=a.grid_method, max_joints=a.max_joints, rate=a.rate,
               train=a.train, primary=a.primary, aux_same_rows=a.aux_same_rows, tol=a.tol,
               kallus_extended=a.kallus_extended, max_categories=a.max_categories, bins=a.bins)

    res = run(scenarios, a.classifiers, a.methods, a.seeds, cfg, a.verbose)
    os.makedirs(a.out, exist_ok=True)
    summ, comp = summarise(res), compare_methods(res, a.tol)
    res.to_csv(os.path.join(a.out, "results.csv"), index=False)
    summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    comp.to_csv(os.path.join(a.out, "method_comparison.csv"), index=False)
    if not a.no_plots:
        plot(res, a.out)

    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print("\nmean distance from the true value to the interval (0 = covered; for the "
              "independence point estimate this is |error|):")
        print(res.pivot_table(index="metric", columns=["method", "scenario"], values="miss",
                              aggfunc="mean").round(3).to_string())
        if len(comp):
            print("\nshare of runs where method's interval lies inside reference's:")
            print(comp.groupby(["method", "reference"]).inside.mean().round(3).to_string())
    print(f"\nsaved results.csv, summary.csv, method_comparison.csv and plots to {a.out}/")


if __name__ == "__main__":
    main()