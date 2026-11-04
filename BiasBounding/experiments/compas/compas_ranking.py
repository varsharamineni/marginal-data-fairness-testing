"""
COMPAS ranking experiment, Zehlike et al. (2022) ranking: bounds on ranking fairness with race
withheld.

Ranking. As in Zehlike et al. ("Fair Top-k Ranking with Multiple Protected Groups", IP&M 2022)
and their code: each defendant gets
    score = 0.25 decile_score + 0.5 v_decile_score + 0.25 priors_count,
each column scaled to [0, 1]. By default the score is flipped so the defendants least likely
to recidivate rank first (their alternative-to-prison scenario: a high position is a benefit);
--highest-risk-first ranks the highest score first instead. Gaussian noise (sd 1e-6) breaks
ties, so the ranking is a strict order of all n defendants.

grid_joint needs a discrete internal variable, so the ranking is cut into position blocks
(--edges, default positions 1-500, 501-1000, 1001-1500 and 1501-n). Positions inside a block
are treated as tied. The internal data hold the block and the common variable(s); the external
data hold the common variable(s) and race (Black vs White).

Common variables (--proxies; default charge and age)
  charge          charge degree, felony or misdemeanour (2 bins)
  age             age category: under 25, 25-45, over 45 (3 bins). Age is itself a protected
                  attribute; here it is used only as a proxy for race.
  priors          number of prior offences: 0, 1-3, 4 or more (3 bins)
  priors_charge   prior offences x charge degree (6 bins)
Note: priors_count is also a quarter of the score, so the priors proxies are informative about
rank partly by construction; charge degree and age are not inputs to the score.

Metrics. Group a = 1 is Black and a = 0 White; theta(l, a) = p(block l, race a), blocks
l = 1, ..., L ordered from the top; p(l) and p(a) are the block and race shares (fixed by the
data); n is the number of defendants. Block l occupies positions s_l + 1, ..., e_l with
e_l = n (p(1) + ... + p(l)) and s_l = e_(l-1).
  top-block rate diff   P(block 1 | Black) - P(block 1 | White)
                          = theta(1, 1) / p(1) - theta(1, 0) / p(0)                     linear
                        Default order: below 0, Black defendants are less often in the top
                        (least risky) block.
  exposure ratio        Singh & Joachims (2018). Position k receives attention
                        v_k = 1 / log2(1 + k); a block shares its average attention vbar_l.
                          E_a = sum_l theta(l, a) vbar_l / p(a),   metric E_1 / E_0.
                        Default order: attention is a benefit, so below 1, Black defendants
                        sit further from the top.                                 linear-fractional
  NDKL                  Geyik et al. (2019), over the block boundaries:
                          NDKL = sum_j w_j KL(pi_j || p) / sum_j w_j,  w_j = 1 / log2(1 + e_j),
                        pi_j = racial composition of the top e_j positions.        nonlinear, convex
Reference bounds: exact LP (Charnes-Cooper for the exposure ratio) and, for NDKL, convex
minimisation for the lower bound; NDKL's upper bound is attained at a vertex, which the grid
contains, so the grid's value is exact.

Versions: exact (both tables from all records) and split (records split at random into
internal and external halves; seed = rep). The truth uses all records with race.

Usage
  python compas_ranking_zehlike.py --data path/to/compas-scores-two-years_clean.csv
  python compas_ranking_zehlike.py --data ... --edges 300 750 1500
  python compas_ranking_zehlike.py --data ... --highest-risk-first
Outputs: <out>/compas_bounds.csv and <out>/figures/compas_bounds.png/.pdf
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fs_common import (C_CI, C_EXACT, C_GRID, GRIDC, INK, REF, STYLE, TRUTH, ci_joint,
                       convex_min, enumerate_joints, harmonize, lp_bounds)

PRIOR_EDGES = [-1, 0, 3, np.inf]          # 0, 1-3, 4+ prior offences
PROXY_LABEL = {"charge": "Charge degree", "age": "Age",
               "priors": "Prior offences", "priors_charge": "Prior offences\n$\\times$ charge"}
AGE_CODE = {"Less than 25": 0, "25 - 45": 1, "Greater than 45": 2}
METRICS = ["top-block rate diff", "exposure ratio", "NDKL"]
LABEL = {"top-block rate diff": "Top-block rate difference", "exposure ratio": "Exposure ratio",
         "NDKL": "NDKL"}
PARITY = {"top-block rate diff": 0.0, "exposure ratio": 1.0, "NDKL": 0.0}


def load(path, edges, highest_risk_first=False, seed=0):
    d = pd.read_csv(path)
    d = d[d.race.isin(["African-American", "Caucasian"])].copy()
    d["a"] = (d.race == "African-American").astype(int)
    d["prior_i"] = pd.cut(d.priors_count, PRIOR_EDGES, labels=False)
    d["fel_i"] = (d.c_charge_degree == "F").astype(int)
    d["age_i"] = d.age_cat.map(AGE_CODE)
    d = d.dropna(subset=["prior_i", "age_i", "decile_score", "v_decile_score"]).reset_index(drop=True)
    # Zehlike et al. score: weighted sum of scaled decile, violent decile and prior offences
    scale = lambda x: (x - x.min()) / np.ptp(x)
    risk = 0.25 * scale(d.decile_score) + 0.5 * scale(d.v_decile_score) \
        + 0.25 * scale(d.priors_count)
    d["score"] = risk if highest_risk_first else risk.max() - risk
    d["score"] += np.random.default_rng(seed).normal(0, 1e-6, len(d))     # break ties
    pos = d.score.rank(ascending=False, method="first").to_numpy(int)    # 1 = top
    d["blk_i"] = np.searchsorted(edges, pos, side="left")                # position block
    return d


def tables(d, proxies, n_blocks):
    z, K = {"charge": (d.fel_i, 2),
            "age": (d.age_i, 3),
            "priors": (d.prior_i, 3),
            "priors_charge": (d.prior_i * 2 + d.fel_i, 6)}[proxies]
    z = z.to_numpy(int)
    q = np.zeros((K, n_blocks, 2))
    np.add.at(q, (z, d.blk_i.to_numpy(int), d.a.to_numpy(int)), 1.0)
    return q / q.sum()


# ------------------------------------------------------------------ ranking metrics
class Ranking:
    """Metrics of p(block, race) for a ranking of n defendants, blocks ordered top first.
    The block and race margins are fixed by the data, so block positions are too."""

    def __init__(self, p_level, p_race, n):
        self.p_race = p_race
        ends = np.round(n * np.cumsum(p_level)).astype(int)
        starts = np.concatenate([[0], ends[:-1]])
        H = np.concatenate([[0.0], np.cumsum(1 / np.log2(1 + np.arange(1, n + 1)))])
        size = np.maximum(ends - starts, 1)
        self.block_exp = (H[ends] - H[starts]) / size              # average attention
        self.w = 1 / np.log2(1 + np.maximum(ends, 1))              # NDKL weight per prefix
        self.cum_mass = np.cumsum(p_level)

    def __call__(self, theta):
        """theta [N, block, race] -> dict of metric arrays."""
        theta = np.asarray(theta)
        p = self.p_race
        rate_top = theta[:, 0, :] / p
        exp_g = (theta * self.block_exp[None, :, None]).sum(1) / p
        cum = np.cumsum(theta, axis=1) / self.cum_mass[None, :, None]   # prefix compositions
        with np.errstate(divide="ignore", invalid="ignore"):
            kl = np.where(cum > 0, cum * np.log(cum / p[None, None, :]), 0).sum(2)
        return {"top-block rate diff": rate_top[:, 1] - rate_top[:, 0],
                "exposure ratio": exp_g[:, 1] / exp_g[:, 0],
                "NDKL": (kl * self.w).sum(1) / self.w.sum()}


def run(d, proxies, n_blocks, version, rep, nums, max_joints, n_rank):
    q_true = tables(d, proxies, n_blocks)
    if version == "exact":
        r, m = q_true.sum(2), q_true.sum(1)
    else:
        u = np.random.default_rng(rep).random(len(d))
        qi, qe = tables(d[u < 0.5], proxies, n_blocks), tables(d[u >= 0.5], proxies, n_blocks)
        r, m = harmonize(qi.sum(2), qe.sum(1))
    r, m = harmonize(r, m)
    keep = r.sum(1) > 0                           # drop common values with no records
    r, m = r[keep], m[keep]
    rank = Ranking(r.sum(0), m.sum(0), n_rank)
    J, num, sec = enumerate_joints(r, m, nums, max_joints)
    vals = rank(J.sum(1))
    truth = Ranking(q_true.sum((0, 2)), q_true.sum((0, 1)), n_rank)(q_true.sum(0)[None])
    ci = rank(ci_joint(r, m).sum(0)[None])
    # exact references
    B = r.shape[1]
    p = m.sum(0)
    top = np.zeros((r.shape[0], B, 2)); top[:, 0, 1] = 1 / p[1]; top[:, 0, 0] = -1 / p[0]
    num_w = np.zeros_like(top); den_w = np.zeros_like(top)
    num_w[:, :, 1] = rank.block_exp / p[1]
    den_w[:, :, 0] = rank.block_exp / p[0]
    t0 = time.perf_counter()
    exact = {"top-block rate diff": lp_bounds(r, m, top),
             "exposure ratio": lp_bounds(r, m, num_w, den_w),
             "NDKL": (convex_min(r, m, lambda x: float(rank(x.sum(0)[None])["NDKL"][0])),
                      float(vals["NDKL"].max()))}
    sec_exact = time.perf_counter() - t0
    rows = []
    for k in METRICS:
        rows.append(dict(version=version, proxies=proxies, blocks=n_blocks, rep=rep, metric=k,
                         truth=float(truth[k][0]), ci=float(ci[k][0]),
                         grid_lower=float(np.nanmin(vals[k])), grid_upper=float(np.nanmax(vals[k])),
                         exact_lower=exact[k][0], exact_upper=exact[k][1], num=num,
                         n_joints=len(J), seconds_grid=sec, seconds_exact=sec_exact))
    return rows


def figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update(STYLE)
    res = res[res.version == "split"]
    proxies = [p for p in PROXY_LABEL if p in set(res.proxies)]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), layout="constrained")
    for ax, k in zip(axes, METRICS):
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=GRIDC, lw=0.6)
        ax.axhline(PARITY[k], color=REF, lw=0.8, zorder=1)
        for x, pr in enumerate(proxies):
            row = res[(res.proxies == pr) & (res.metric == k)].iloc[0]
            for off, lo, hi, col in [(-0.1, row.grid_lower, row.grid_upper, C_GRID),
                                     (0.1, row.exact_lower, row.exact_upper, C_EXACT)]:
                mid = (lo + hi) / 2
                ax.errorbar(x + off, mid, yerr=[[mid - lo], [hi - mid]], fmt="none", ecolor=col,
                            elinewidth=1.3, capsize=3, capthick=1.3, zorder=3)
            ax.plot(x - 0.1, row.truth, "*", ms=8.5, color=TRUTH, mec=INK, mew=0.5, zorder=5)
        ax.set_xticks(range(len(proxies)))
        ax.set_xticklabels([PROXY_LABEL[p] for p in proxies])
        ax.set_xlim(-0.5, len(proxies) - 0.4)
        ax.set_title(LABEL[k], loc="left", pad=4)
    h = [Line2D([], [], color=C_GRID, lw=1.3, marker="_", ms=6, label="Feasible set (grid)"),
         Line2D([], [], color=C_EXACT, lw=1.3, marker="_", ms=6, label="Optimisation Methods"),
         Line2D([], [], ls="", marker="*", ms=8.5, color=TRUTH, mec=INK, mew=0.5,
                label="True value")]
    fig.legend(handles=h, loc="outside upper center", ncol=4, frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.expanduser(
    "~/fairness-incomplete-data/real_data/raw/compas-scores-two-years_clean.csv"))
    ap.add_argument("--proxies", nargs="+", default=["charge", "age"],
                    choices=list(PROXY_LABEL))
    ap.add_argument("--edges", type=int, nargs="+", default=[500, 1000, 1500],
                    help="last position of each block except the final one")
    ap.add_argument("--highest-risk-first", action="store_true",
                    help="rank the highest score first instead of the lowest")
    ap.add_argument("--versions", nargs="+", default=["split", "exact"])
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--nums", type=int, nargs="+", default=[2, 3, 4, 5, 10, 20])
    ap.add_argument("--max-joints", type=int, default=2_000_000)
    ap.add_argument("--out", default="results/compas_zehlike")
    a = ap.parse_args()
    d = load(a.data, a.edges, a.highest_risk_first)
    n_blocks = len(a.edges) + 1
    print(f"{len(d):,} defendants ({d.a.sum():,} Black, {(1 - d.a).sum():,} White); "
          f"blocks end at {a.edges + [len(d)]}")
    rows = []
    for proxies in a.proxies:
        for version in a.versions:
            try:
                rows += run(d, proxies, n_blocks, version, a.rep, a.nums, a.max_joints, len(d))
            except SystemExit as e:
                print(f"  {proxies}, {version}: {e}")
                continue
            r0 = rows[-1]
            print(f"  {proxies}, {version}: num={r0['num']}, "
                  f"{r0['n_joints']:,} joints, {r0['seconds_grid']:.1f}s")
    if not rows:
        print("no setting stayed under --max-joints; nothing to save")
        return
    res = pd.DataFrame(rows)
    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)
    res.to_csv(os.path.join(a.out, "compas_bounds.csv"), index=False)
    with pd.option_context("display.width", 220):
        print(res.drop(columns=["rep"]).round(4).to_string(index=False))
    figure(res, os.path.join(a.out, "figures", "compas_bounds"))
    print(f"saved {a.out}/compas_bounds.csv and figures/compas_bounds.png/.pdf")


if __name__ == "__main__":
    main()