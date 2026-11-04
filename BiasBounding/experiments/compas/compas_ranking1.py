
"""
COMPAS ranking experiment, Zehlike et al. (2022):
bounds on ranking fairness with race withheld.

Ranking:
    score = 0.25 decile_score
          + 0.5 v_decile_score
          + 0.25 priors_count

Each component is scaled to [0, 1].
By default, defendants least likely to recidivate rank first.

The script calculates:
    1. True individual-position NDKL (all ranking positions).
    2. True coarsened NDKL (ranking blocks).
    3. Grid-based bounds.
    4. Exact optimisation-based bounds.

The individual-position NDKL is an additional benchmark.
The identified bounds remain bounds on the coarsened metric.
"""

import argparse
import os
import time
import sys

from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fs_common import (
    C_CI, C_EXACT, C_GRID, GRIDC, INK, REF, STYLE, TRUTH,
    ci_joint, convex_min, enumerate_joints, harmonize, lp_bounds
)


PRIOR_EDGES = [-1, 0, 3, np.inf]

PROXY_LABEL = {
    "charge": "Charge degree",
    "age": "Age",
    "priors": "Prior offences",
    "priors_charge": "Prior offences\n$\\times$ charge"
}

AGE_CODE = {
    "Less than 25": 0,
    "25 - 45": 1,
    "Greater than 45": 2
}

METRICS = [
    "top-block rate diff",
    "exposure ratio",
    "NDKL"
]

LABEL = {
    "top-block rate diff": "Top-block rate difference",
    "exposure ratio": "Exposure ratio",
    "NDKL": "NDKL"
}

PARITY = {
    "top-block rate diff": 0.0,
    "exposure ratio": 1.0,
    "NDKL": 0.0
}


# ------------------------------------------------------------------
# Load data and construct ranking
# ------------------------------------------------------------------

def load(path, edges, highest_risk_first=False, seed=0):

    d = pd.read_csv(path)

    d = d[
        d.race.isin(["African-American", "Caucasian"])
    ].copy()

    d["a"] = (
        d.race == "African-American"
    ).astype(int)

    d["prior_i"] = pd.cut(
        d.priors_count,
        PRIOR_EDGES,
        labels=False
    )

    d["fel_i"] = (
        d.c_charge_degree == "F"
    ).astype(int)

    d["age_i"] = d.age_cat.map(AGE_CODE)

    d = d.dropna(
        subset=[
            "prior_i",
            "age_i",
            "decile_score",
            "v_decile_score"
        ]
    ).reset_index(drop=True)

    # Zehlike et al. ranking score:
    # weighted sum of scaled decile, violent decile
    # and prior offences.

    scale = lambda x: (
        (x - x.min()) / np.ptp(x)
    )

    risk = (
        0.25 * scale(d.decile_score)
        + 0.5 * scale(d.v_decile_score)
        + 0.25 * scale(d.priors_count)
    )

    d["score"] = (
        risk if highest_risk_first
        else risk.max() - risk
    )

    # Gaussian noise breaks ties.
    d["score"] += np.random.default_rng(seed).normal(
        0, 1e-6, len(d)
    )

    # Exact individual ranking position:
    # position 1 is the top of the ranking.

    pos = d.score.rank(
        ascending=False,
        method="first"
    ).to_numpy(int)

    d["pos_i"] = pos

    # Coarsened ranking block.
    d["blk_i"] = np.searchsorted(
        edges,
        pos,
        side="left"
    )

    return d


# ------------------------------------------------------------------
# Construct joint distribution tables
# ------------------------------------------------------------------

def tables(d, proxies, n_blocks):

    z, K = {
        "charge": (d.fel_i, 2),
        "age": (d.age_i, 3),
        "priors": (d.prior_i, 3),
        "priors_charge": (
            d.prior_i * 2 + d.fel_i,
            6
        )
    }[proxies]

    z = z.to_numpy(int)

    q = np.zeros((K, n_blocks, 2))

    np.add.at(
        q,
        (
            z,
            d.blk_i.to_numpy(int),
            d.a.to_numpy(int)
        ),
        1.0
    )

    return q / q.sum()


# ------------------------------------------------------------------
# Ranking fairness metrics for coarsened rankings
# ------------------------------------------------------------------

class Ranking:

    """
    Metrics of p(block, race) for a ranking of n defendants.

    Blocks are ordered from top to bottom.
    """

    def __init__(self, p_level, p_race, n):

        self.p_race = p_race

        ends = np.round(
            n * np.cumsum(p_level)
        ).astype(int)

        starts = np.concatenate([
            [0],
            ends[:-1]
        ])

        H = np.concatenate([
            [0.0],
            np.cumsum(
                1 / np.log2(
                    1 + np.arange(1, n + 1)
                )
            )
        ])

        size = np.maximum(
            ends - starts,
            1
        )

        # Average attention received by each block.

        self.block_exp = (
            H[ends] - H[starts]
        ) / size

        # NDKL discount weights at block boundaries.

        self.w = (
            1 / np.log2(
                1 + np.maximum(ends, 1)
            )
        )

        self.cum_mass = np.cumsum(p_level)

    def __call__(self, theta):

        theta = np.asarray(theta)

        p = self.p_race

        # Top-block rate difference.

        rate_top = theta[:, 0, :] / p

        # Exposure by racial group.

        exp_g = (
            theta * self.block_exp[None, :, None]
        ).sum(1) / p

        # Racial composition of cumulative prefixes.

        cum = (
            np.cumsum(theta, axis=1)
            / self.cum_mass[None, :, None]
        )

        # KL divergence from overall racial composition.

        with np.errstate(
            divide="ignore",
            invalid="ignore"
        ):

            kl = np.where(
                cum > 0,
                cum * np.log(
                    cum / p[None, None, :]
                ),
                0
            ).sum(2)

        return {
            "top-block rate diff":
                rate_top[:, 1] - rate_top[:, 0],

            "exposure ratio":
                exp_g[:, 1] / exp_g[:, 0],

            "NDKL":
                (kl * self.w).sum(1) / self.w.sum()
        }


# ------------------------------------------------------------------
# NEW: True NDKL using all individual ranking positions
# ------------------------------------------------------------------

def true_ndkl_individual(d):

    """
    Calculate the original NDKL using every individual
    ranking position, rather than coarsened ranking blocks.

    Position 1 is the top of the ranking.

    NDKL = sum_k w_k KL(pi_k || p) / sum_k w_k

    where:
        pi_k = racial composition of the first k defendants
        p    = overall racial composition
        w_k  = 1 / log2(1 + k)
    """

    # Sort using the exact ranking positions.

    d = d.sort_values(
        "pos_i"
    ).reset_index(drop=True)

    n = len(d)

    a = d["a"].to_numpy()

    # Overall racial proportions.

    p_black = np.mean(a == 1)
    p_white = np.mean(a == 0)

    # Cumulative racial proportions at every position.

    k = np.arange(1, n + 1)

    cum_black = (
        np.cumsum(a == 1) / k
    )

    cum_white = (
        np.cumsum(a == 0) / k
    )

    # KL divergence for every ranking prefix.

    kl = np.zeros(n)

    mask_black = cum_black > 0

    kl[mask_black] += (
        cum_black[mask_black]
        * np.log(
            cum_black[mask_black] / p_black
        )
    )

    mask_white = cum_white > 0

    kl[mask_white] += (
        cum_white[mask_white]
        * np.log(
            cum_white[mask_white] / p_white
        )
    )

    # Position-dependent discount weights.

    weights = 1 / np.log2(1 + k)

    # Original individual-position NDKL.

    ndkl = (
        np.sum(weights * kl)
        / np.sum(weights)
    )

    return float(ndkl)


# ------------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------------

def run(
    d,
    proxies,
    n_blocks,
    version,
    rep,
    nums,
    max_joints,
    n_rank
):

    q_true = tables(
        d,
        proxies,
        n_blocks
    )

    if version == "exact":

        r, m = (
            q_true.sum(2),
            q_true.sum(1)
        )

    else:

        u = np.random.default_rng(
            rep
        ).random(len(d))

        qi = tables(
            d[u < 0.5],
            proxies,
            n_blocks
        )

        qe = tables(
            d[u >= 0.5],
            proxies,
            n_blocks
        )

        r, m = harmonize(
            qi.sum(2),
            qe.sum(1)
        )

    r, m = harmonize(r, m)

    keep = r.sum(1) > 0

    r, m = r[keep], m[keep]

    rank = Ranking(
        r.sum(0),
        m.sum(0),
        n_rank
    )

    # Enumerate feasible joint distributions.

    J, num, sec = enumerate_joints(
        r,
        m,
        nums,
        max_joints
    )

    vals = rank(J.sum(1))

    # --------------------------------------------------------------
    # TRUE VALUES
    # --------------------------------------------------------------

    # True values for the four-block ranking.

    truth = Ranking(
        q_true.sum((0, 2)),
        q_true.sum((0, 1)),
        n_rank
    )(q_true.sum(0)[None])

    # NEW: True NDKL using every individual ranking position.

    truth_individual = true_ndkl_individual(d)

    # --------------------------------------------------------------
    # CI ESTIMATE
    # --------------------------------------------------------------

    ci = rank(
        ci_joint(r, m).sum(0)[None]
    )

    # --------------------------------------------------------------
    # EXACT OPTIMISATION BOUNDS
    # --------------------------------------------------------------

    B = r.shape[1]

    p = m.sum(0)

    top = np.zeros(
        (r.shape[0], B, 2)
    )

    top[:, 0, 1] = 1 / p[1]
    top[:, 0, 0] = -1 / p[0]

    num_w = np.zeros_like(top)
    den_w = np.zeros_like(top)

    num_w[:, :, 1] = (
        rank.block_exp / p[1]
    )

    den_w[:, :, 0] = (
        rank.block_exp / p[0]
    )

    t0 = time.perf_counter()

    exact = {

        "top-block rate diff":
            lp_bounds(r, m, top),

        "exposure ratio":
            lp_bounds(
                r, m,
                num_w,
                den_w
            ),

        "NDKL": (
            convex_min(
                r,
                m,
                lambda x: float(
                    rank(
                        x.sum(0)[None]
                    )["NDKL"][0]
                )
            ),

            float(
                vals["NDKL"].max()
            )
        )
    }

    sec_exact = (
        time.perf_counter() - t0
    )

    # --------------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------------

    rows = []

    for k in METRICS:

        rows.append(
            dict(
                version=version,
                proxies=proxies,
                blocks=n_blocks,
                rep=rep,
                metric=k,

                # True coarsened metric.
                truth=float(truth[k][0]),

                # NEW: True individual-position NDKL.
                # Only applicable to NDKL.
                truth_individual=(
                    truth_individual
                    if k == "NDKL"
                    else np.nan
                ),

                ci=float(ci[k][0]),

                grid_lower=float(
                    np.nanmin(vals[k])
                ),

                grid_upper=float(
                    np.nanmax(vals[k])
                ),

                exact_lower=exact[k][0],
                exact_upper=exact[k][1],

                num=num,
                n_joints=len(J),
                seconds_grid=sec,
                seconds_exact=sec_exact
            )
        )

    return rows


# ------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------

def figure(res, path):

    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    from matplotlib.lines import Line2D

    plt.rcParams.update(STYLE)

    res = res[
        res.version == "split"
    ]

    proxies = [
        p for p in PROXY_LABEL
        if p in set(res.proxies)
    ]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(7.0, 2.4),
        layout="constrained"
    )

    for ax, k in zip(axes, METRICS):

        ax.set_axisbelow(True)

        ax.yaxis.grid(
            True,
            color=GRIDC,
            lw=0.6
        )

        ax.axhline(
            PARITY[k],
            color=REF,
            lw=0.8,
            zorder=1
        )

        for x, pr in enumerate(proxies):

            row = res[
                (res.proxies == pr)
                & (res.metric == k)
            ].iloc[0]

            for off, lo, hi, col in [
                (
                    -0.1,
                    row.grid_lower,
                    row.grid_upper,
                    C_GRID
                ),
                (
                    0.1,
                    row.exact_lower,
                    row.exact_upper,
                    C_EXACT
                )
            ]:

                mid = (lo + hi) / 2

                ax.errorbar(
                    x + off,
                    mid,
                    yerr=[
                        [mid - lo],
                        [hi - mid]
                    ],
                    fmt="none",
                    ecolor=col,
                    elinewidth=1.3,
                    capsize=3,
                    capthick=1.3,
                    zorder=3
                )

            ax.plot(
                x - 0.1,
                row.truth,
                "*",
                ms=8.5,
                color=TRUTH,
                mec=INK,
                mew=0.5,
                zorder=5
            )

        ax.set_xticks(
            range(len(proxies))
        )

        ax.set_xticklabels(
            [PROXY_LABEL[p] for p in proxies]
        )

        ax.set_xlim(
            -0.5,
            len(proxies) - 0.4
        )

        ax.set_title(
            LABEL[k],
            loc="left",
            pad=4
        )

    h = [

        Line2D(
            [],
            [],
            color=C_GRID,
            lw=1.3,
            marker="_",
            ms=6,
            label="Feasible set (grid)"
        ),

        Line2D(
            [],
            [],
            color=C_EXACT,
            lw=1.3,
            marker="_",
            ms=6,
            label="Optimisation Methods"
        ),

        Line2D(
            [],
            [],
            ls="",
            marker="*",
            ms=8.5,
            color=TRUTH,
            mec=INK,
            mew=0.5,
            label="True value"
        )
    ]

    fig.legend(
        handles=h,
        loc="outside upper center",
        ncol=4,
        frameon=False
    )

    for ext in ("png", "pdf"):

        fig.savefig(
            f"{path}.{ext}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02
        )

    plt.close(fig)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    ap.add_argument(
        "--data",
        default=os.path.expanduser(
            "~/fairness-incomplete-data/real_data/raw/"
            "compas-scores-two-years_clean.csv"
        )
    )

    ap.add_argument(
        "--proxies",
        nargs="+",
        default=["charge", "age"],
        choices=list(PROXY_LABEL)
    )

    ap.add_argument(
        "--edges",
        type=int,
        nargs="+",
        default=[500, 1000, 1500],
        help="Last position of each block except the final one"
    )

    ap.add_argument(
        "--highest-risk-first",
        action="store_true",
        help="Rank the highest score first instead of the lowest"
    )

    ap.add_argument(
        "--versions",
        nargs="+",
        default=["split", "exact"]
    )

    ap.add_argument(
        "--rep",
        type=int,
        default=0
    )

    ap.add_argument(
        "--nums",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5, 10, 20]
    )

    ap.add_argument(
        "--max-joints",
        type=int,
        default=2_000_000
    )

    ap.add_argument(
        "--out",
        default="results/compas_zehlike"
    )

    a = ap.parse_args()

    d = load(
        a.data,
        a.edges,
        a.highest_risk_first
    )

    n_blocks = len(a.edges) + 1

    print(
        f"{len(d):,} defendants "
        f"({d.a.sum():,} Black, "
        f"{(1 - d.a).sum():,} White); "
        f"blocks end at {a.edges + [len(d)]}"
    )

    # Calculate individual-position NDKL once.
    individual_ndkl = true_ndkl_individual(d)

    print(
        f"True individual-position NDKL: "
        f"{individual_ndkl:.8f}"
    )

    rows = []

    for proxies in a.proxies:

        for version in a.versions:

            try:

                rows += run(
                    d,
                    proxies,
                    n_blocks,
                    version,
                    a.rep,
                    a.nums,
                    a.max_joints,
                    len(d)
                )

            except SystemExit as e:

                print(
                    f"  {proxies}, {version}: {e}"
                )

                continue

            r0 = rows[-1]

            print(
                f"  {proxies}, {version}: "
                f"num={r0['num']}, "
                f"{r0['n_joints']:,} joints, "
                f"{r0['seconds_grid']:.1f}s"
            )

    if not rows:

        print(
            "No setting stayed under --max-joints; "
            "nothing to save"
        )

        return

    res = pd.DataFrame(rows)

    os.makedirs(
        os.path.join(a.out, "figures"),
        exist_ok=True
    )

    res.to_csv(
        os.path.join(
            a.out,
            "compas_bounds.csv"
        ),
        index=False
    )

    with pd.option_context(
        "display.width",
        220
    ):

        print(
            res.drop(
                columns=["rep"]
            ).round(8).to_string(index=False)
        )

    figure(
        res,
        os.path.join(
            a.out,
            "figures",
            "compas_bounds"
        )
    )

    print(
        f"Saved {a.out}/compas_bounds.csv "
        f"and figures/compas_bounds.png/.pdf"
    )


if __name__ == "__main__":
    main()
