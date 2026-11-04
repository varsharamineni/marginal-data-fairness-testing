"""
COMPAS ranking (Zehlike et al. 2022 setting) with race withheld, bounded with grid_joint.

Ranking (as in the paper's D1 and its repository):
  score = 0.25*decile + 0.5*v_decile + 0.25*priors (each scaled to [0,1]), flipped so the
  least likely to recidivate rank first, tiny noise breaks ties; top-k with k = 1500;
  protected group = people of color (PoC, every race but Caucasian).

grid_joint needs a discrete internal variable, so the ranking is cut into position blocks
(default 1-500, 501-1000, 1001-1500, 1501-n). Positions inside a block are treated as tied.
  internal = position block     (ranked data: block x proxy)
  common   = proxy cell         (both datasets)
  external = race, PoC = 1      (auxiliary data: proxy x race)
Every joint from grid_joint gives theta(block, race), and so every metric below.

Metrics (block level; theta(b, a) = p(block b, race a), blocks ordered from the top)
  exposure ratio  position k gets attention v_k = 1/log2(1+k) if k <= k_top, else 0; a block
                  shares its average attention vbar_b. E_a = sum_b theta(b,a) vbar_b / p(a);
                  metric E_PoC / E_White. Top = benefit here, so below 1 = PoC disadvantaged.
                                                                       linear-fractional
  RGF score       FA*IR ranked group fairness at the block boundaries e_j <= k_top:
                    min_j log10 BinomCDF(n * sum_(b<=j) theta(b, PoC); e_j, p_stat).
                  0 = every boundary prefix passes; <= log10(alpha) = fails the test.
                                                                       nonlinear, monotone
Both metrics only improve when PoC mass moves to higher blocks, so in each proxy cell the
bounds are attained at the two "corner" tables (PoC filling blocks from the top, or from the
bottom). grid_joint's sequential method contains both corners, so its bounds are exact; the
script checks this against the closed-form corners ("exact" columns).

Versions: exact (proxy x race from all records) and split (proxy x race from a random half of
the records; the ranked data are always all records). Truth uses everyone's race.

Usage
  python compas_ranking_fair.py                      # downloads the ProPublica file
  python compas_ranking_fair.py --data compas-scores-two-years.csv --edges 300 750 1500
Outputs: <out>/compas_fair_bounds.csv and <out>/compas_fair_bounds.png
Needs grid_joint.py in the same folder or on the Python path.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import binom
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "methods"))
from grid_joint import solve_joint_distribution  

DATA_URL = ("https://raw.githubusercontent.com/propublica/"
            "compas-analysis/master/compas-scores-two-years.csv")
ALPHA = 0.1
METRICS = ["exposure ratio", "RGF score"]
PARITY = {"exposure ratio": 1.0, "RGF score": np.log10(ALPHA)}


# ------------------------------------------------------------------------ data
def load(path, seed=0):
    d = pd.read_csv(path)
    d = d[d.days_b_screening_arrest.between(-30, 30) & (d.is_recid != -1)
          & (d.c_charge_degree != "O") & (d.score_text != "N/A")].copy()
    scale = lambda x: (x - x.min()) / np.ptp(x)
    s = 0.25 * scale(d.decile_score) + 0.5 * scale(d.v_decile_score) \
        + 0.25 * scale(d.priors_count)
    d["score"] = (s.max() - s) + np.random.default_rng(seed).normal(0, 1e-6, len(d))
    d["a"] = (d.race != "Caucasian").astype(int)
    d = d.sort_values("score", ascending=False).reset_index(drop=True)
    d["pos"] = np.arange(1, len(d) + 1)
    return d


def proxy(d, name, rho=None, seed=0):
    """Proxy cells. 'noisy' = race reported correctly with probability rho (2 cells);
    'age' = age category (3 cells); 'age_noisy' = both (6 cells)."""
    if name == "age":
        return d.age_cat.astype(str).values
    flip = np.random.default_rng(seed).random(len(d)) > rho
    guess = np.where(flip, 1 - d.a, d.a).astype(str)
    if name == "noisy":
        return np.char.add("race_guess=", guess)
    return np.char.add(np.char.add(d.age_cat.astype(str).values, "|"), guess)


# ---------------------------------------------------------------- block metrics
class BlockRanking:
    def __init__(self, edges, n, k_top, p_stat):
        self.ends = np.array(list(edges) + [n])
        starts = np.concatenate([[0], self.ends[:-1]])
        k = np.arange(1, n + 1)
        v = np.where(k <= k_top, 1 / np.log2(1 + k), 0.0)
        H = np.concatenate([[0.0], np.cumsum(v)])
        self.vbar = (H[self.ends] - H[starts]) / (self.ends - starts)
        self.n = n
        self.bounds = [j for j, e in enumerate(self.ends) if e <= k_top]
        # log10 BinomCDF(c; e_j, p) for integer c, to interpolate fractional counts
        self.tables = []
        for j in self.bounds:
            e = self.ends[j]
            lc = np.logaddexp.accumulate(binom.logpmf(np.arange(e + 1), e, p_stat))
            self.tables.append((np.arange(e + 1), np.minimum(lc, 0) / np.log(10)))

    def __call__(self, theta):
        """theta [N, block, race(0=White, 1=PoC)] -> dict of metric arrays."""
        theta = np.asarray(theta, float)
        p = theta.sum(1)
        exp_ = (theta * self.vbar[None, :, None]).sum(1) / p
        cum_poc = self.n * np.cumsum(theta[:, :, 1], axis=1)
        rgf = np.min([np.interp(cum_poc[:, j], x, y)
                      for j, (x, y) in zip(self.bounds, self.tables)], axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = exp_[:, 1] / exp_[:, 0]      # inf if no White mass reaches the top-k
        return {"exposure ratio": ratio, "RGF score": rgf}


def corner_tables(row_sums, poc_mass, from_top):
    """Closed-form corner: PoC mass fills blocks from the top (or bottom) of one cell."""
    x, left = np.zeros_like(row_sums), poc_mass
    for b in (range(len(row_sums)) if from_top else reversed(range(len(row_sums)))):
        x[b] = min(row_sums[b], left)
        left -= x[b]
    return np.stack([row_sums - x, x], axis=1)              # [block, race]


# ---------------------------------------------------------------------- one run
def run(d, z, edges, k_top, aux_mask, nums, max_joints):
    blk = np.searchsorted(edges, d.pos.values, side="left")
    B = len(edges) + 1
    p_primary = pd.Series(1.0, index=pd.MultiIndex.from_arrays(
        [[f"b{b}" for b in blk], z])).groupby(level=[0, 1]).sum().to_dict()
    p_aux = pd.Series(1.0, index=pd.MultiIndex.from_arrays(
        [z[aux_mask], d.a.values[aux_mask].astype(str)])).groupby(level=[0, 1]).sum().to_dict()

    J = None
    for num in sorted(nums, reverse=True):                 # largest grid that fits
        try:
            t0 = time.perf_counter()
            J, (I, C, E) = solve_joint_distribution(p_primary, p_aux, num, method="sequential",
                                                    max_joints=max_joints, as_dicts=False)
            sec = time.perf_counter() - t0
            break
        except SystemExit as e:
            print(f"   num={num}: {e}")
    if J is None:
        raise SystemExit("no num fits under --max-joints")
    assert I == [f"b{b}" for b in range(B)] and E == ["0", "1"]

    p_stat = d.a.mean()
    rank = BlockRanking(edges, len(d), k_top, p_stat)
    vals = rank(J.sum(1))

    # closed-form corners on the same margins grid_joint used
    Paux = np.array([[p_aux.get((c, e), 0) for e in E] for c in C]); Paux /= Paux.sum()
    Pprim = np.array([[p_primary.get((i, c), 0) for i in I] for c in C])
    rows_ = Pprim / Pprim.sum(1, keepdims=True) * Paux.sum(1, keepdims=True)
    corners = [sum(corner_tables(rows_[k], Paux[k, 1], top) for k in range(len(C)))
               for top in (True, False)]
    cv = rank(np.array(corners))
    ci = rank((rows_[:, :, None] * (Paux / Paux.sum(1, keepdims=True))[:, None, :]).sum(0)[None])
    t = np.zeros((B, 2)); np.add.at(t, (blk, d.a.values), 1.0)
    truth = rank((t / t.sum())[None])

    return [dict(metric=k, truth=truth[k][0], ci=ci[k][0],
                 grid_lower=np.nanmin(vals[k]), grid_upper=np.nanmax(vals[k]),
                 exact_lower=cv[k].min(), exact_upper=cv[k].max(),
                 num=num, n_joints=len(J), seconds=sec) for k in METRICS]


# ------------------------------------------------------------------------ figure
def figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = res[res.version == "split"]
    labels = list(dict.fromkeys(res.proxy))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), layout="constrained")
    for ax, k in zip(axes, METRICS):
        sub = res[res.metric == k]
        finite = np.concatenate([sub[c].values for c in
                                 ["grid_upper", "exact_upper", "grid_lower", "truth", "ci"]])
        cap = np.nanmax(finite[np.isfinite(finite)]) * 1.15 if k == "exposure ratio" else 0
        ax.axhline(PARITY[k], color="gray", lw=0.8, ls=":")
        for x, pr in enumerate(labels):
            r = res[(res.proxy == pr) & (res.metric == k)].iloc[0]
            for off, lo, hi, col, lab in [(-0.08, r.grid_lower, r.grid_upper, "C0", "grid_joint"),
                                          (0.08, r.exact_lower, r.exact_upper, "C1", "exact corners")]:
                ax.plot([x + off] * 2, [lo, min(hi, cap) if k == "exposure ratio" else hi],
                        color=col, lw=2, label=lab if x == 0 else None)
                if not np.isfinite(hi):          # unbounded above
                    ax.plot(x + off, cap, "^", color=col, ms=6)
            ax.plot(x - 0.08, r.truth, "k*", ms=8, label="true value" if x == 0 else None)
            ax.plot(x + 0.22, r.ci, "o", mfc="white", mec="C3",
                    label="conditional independence" if x == 0 else None)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_title(k + (" (log10)" if k == "RGF score" else ""), loc="left")
    axes[0].legend(fontsize=7, frameon=False)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DATA_URL)
    ap.add_argument("--edges", type=int, nargs="+", default=[500, 1000, 1500],
                    help="last position of each block before the final one")
    ap.add_argument("--k", type=int, default=1500)
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.7, 0.8, 0.9, 0.95])
    ap.add_argument("--versions", nargs="+", default=["split", "exact"])
    ap.add_argument("--nums", type=int, nargs="+", default=[2, 3, 4, 5, 10, 20])
    ap.add_argument("--max-joints", type=int, default=2_000_000)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--out", default="results/compas_fair")
    a = ap.parse_args()

    d = load(a.data)
    print(f"{len(d):,} defendants, {d.a.sum():,} PoC ({d.a.mean():.3f}); "
          f"blocks end at {a.edges + [len(d)]}, k = {a.k}")
    proxies = [("age", proxy(d, "age"))]
    proxies += [(f"race guess, rho={r}", proxy(d, "noisy", r)) for r in a.rhos]
    proxies += [(f"age x race guess, rho={a.rhos[-1]}", proxy(d, "age_noisy", a.rhos[-1]))]

    rows = []
    for name, z in proxies:
        for version in a.versions:
            aux = np.ones(len(d), bool) if version == "exact" else \
                np.random.default_rng(a.rep + 1000).random(len(d)) < 0.5
            print(f"-- {name}, {version}")
            try:
                rows += [dict(proxy=name, version=version, **r)
                         for r in run(d, z, a.edges, a.k, aux, a.nums, a.max_joints)]
            except SystemExit as e:
                print(f"   skipped: {e}")
    res = pd.DataFrame(rows)
    res["truth_inside"] = (res.truth >= res.grid_lower - 1e-9) & (res.truth <= res.grid_upper + 1e-9)
    res["grid_is_exact"] = np.isclose(res.grid_lower, res.exact_lower) & \
        np.isclose(res.grid_upper, res.exact_upper)
    os.makedirs(a.out, exist_ok=True)
    res.to_csv(os.path.join(a.out, "compas_fair_bounds.csv"), index=False)
    with pd.option_context("display.width", 240, "display.max_columns", 30):
        print(res.round(4).to_string(index=False))
    figure(res, os.path.join(a.out, "compas_fair_bounds.png"))
    print(f"saved {a.out}/compas_fair_bounds.csv and .png")


if __name__ == "__main__":
    main()