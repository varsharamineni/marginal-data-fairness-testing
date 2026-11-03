"""Grid-joint bounds on the paper's HMDA demographic disparities (repo HMDA/small_proxy_*.csv),
next to the paper's Figure 4 intervals (repo HMDA/CI_*.csv). The proxy probabilities in the
repo files are not used: the common variable is binned income and/or county, the primary
data is (outcome, common), the auxiliary data is (common, race) on the same rows."""
import numpy as np, pandas as pd, sys
from grid_joint import from_datasets, solve_joint_distribution

repo = sys.argv[1] if len(sys.argv) > 1 else "repo/HMDA"
num = int(sys.argv[2]) if len(sys.argv) > 2 else 5
out = sys.argv[3] if len(sys.argv) > 3 else "../results/hmda_grid_bounds.csv"
rows = []
EDGES = [0, 25, 50, 75, 100, 150, 250, np.inf]                  # income_bin in hmda_build.py
for proxy, ci, common in [("income", "income", ["income_bin"])]:   # county: too many values to enumerate
    df = pd.read_csv(f"{repo}/small_proxy_{proxy}.csv")
    df["income_bin"] = pd.cut(df.applicant_income_000s, EDGES, right=False).astype(str)
    df["county_id"] = df.county_id.astype(str)                  # categorical, not binned
    ref = pd.read_csv(f"{repo}/CI_{ci}.csv").set_index("race")
    for race, name in [("White", "White-vs-Rest"), ("Black", "Black-vs-Rest"), ("API", "API-vs-Rest")]:
        df["group"] = np.where(df.race == race, race, "Rest")
        try:
            p_primary, p_aux, _, _, _ = from_datasets(df[["outcome"] + common], df[common + ["group"]],
                                                      ["outcome"], common, ["group"])
            J, (I, C, E) = solve_joint_distribution(p_primary, p_aux, num, "sequential", as_dicts=False)
        except SystemExit as e:
            print(f"{ci:<19}{name:<15} skipped: {e}")
            continue
        theta = J.sum(1)                                        # [N, outcome, group]
        rate = theta[:, [str(i) == "1" for i in I], :].sum(1) / theta.sum(1)
        dd = rate[:, E.index(race)] - rate[:, E.index("Rest")]
        truth = (df.outcome[df.race == race] == 1).mean() - (df.outcome[df.race != race] == 1).mean()
        ours = np.array([dd.min(), dd.max(), truth])
        theirs = ref.loc[name, ["PI_lower", "PI_upper", "truth"]].to_numpy(float)
        print(f"{ci:<19}{name:<15} grid {np.round(ours,4)}  repo PI {np.round(theirs,4)}  ({len(J):,} joints)")
        rows.append(dict(proxy=ci, comparison=name, num=num, joints=len(J), grid_lower=ours[0],
                         grid_upper=ours[1], truth=ours[2], repo_PI_lower=theirs[0],
                         repo_PI_upper=theirs[1], repo_truth=theirs[2]))
res = pd.DataFrame(rows)
res.to_csv(out, index=False)
print(f"saved to {out}")

# Interval plot: grid bounds vs the paper's PI per comparison, true value as a black tick
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7.5, 0.9 + 0.9 * len(res)))
for k, r in enumerate(res.itertuples()):
    y = len(res) - 1 - k
    for dy, lo, hi, colour, lab in [(0.15, r.grid_lower, r.grid_upper, "#2a78d6", f"grid joints (num={num})"),
                                    (-0.15, r.repo_PI_lower, r.repo_PI_upper, "#eb6834", "KMZ partial identification set")]:
        ax.plot([lo, hi], [y + dy] * 2, lw=4, color=colour, solid_capstyle="round", label=lab if k == 0 else None)
    ax.plot(r.truth, y, "|", ms=22, mew=2, color="#1a1a19", label="true value" if k == 0 else None)
ax.axvline(0, color="#c3c2b7", lw=1, ls="--", zorder=0)
ax.set_yticks(range(len(res)), res.comparison[::-1])
ax.set_ylim(-0.6, len(res) - 0.4)
ax.set_xlabel("demographic disparity: P(approve | group) - P(approve | rest)")
ax.set_title(f"HMDA, {res.proxy.iloc[0]} proxy: grid joints vs KMZ bounds", loc="left", fontsize=11)
ax.grid(axis="x", color="#e8e7e2", lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(out.rsplit(".", 1)[0] + ".png", dpi=150)
print(f"saved plot to {out.rsplit('.', 1)[0]}.png")
