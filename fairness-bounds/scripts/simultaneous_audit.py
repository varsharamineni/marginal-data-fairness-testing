"""Can the classifier pass two audit rules at the same time, for some feasible joint?

Rules and thresholds are those of audit_rules.py (four-fifths rule, AIF360 fair ranges,
Aequitas ratios, TPRs within 10%, ...). For every pair of rules:

    joints    together: some feasible joint passes both rules (evaluated on the same joint)
    separate  each alone: each rule's separate-bounds interval reaches its pass region.
              This is necessary for "together" but not sufficient; separate bounds cannot
              say whether both can hold at once.

A pair that is possible alone but never together is an incompatibility that only the
joints method detects. On a grid, "never together" means no grid joint passes both; a finer
grid or random joints can still find one (grids of different num are not nested), so check
such pairs with several searches before reporting them. Run on synthetic data, the German data, and a sweep over how
strongly Z predicts A (see audit_rules.sweep_data).

Usage:  python scripts/simultaneous_audit.py [--out-dir outputs/simultaneous_audit]
"""
import argparse
import itertools
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
from audit_rules import RULES, passes, separate_rate_bounds, separate_verdicts, sweep_data  # noqa: E402
from metric_comparison import from_config, group_table, synthetic, true_joint  # noqa: E402


def analyse(name, prim, aux, a_true, a_levels, num):
    joints = methods.feasible_joints(prim, aux, a_levels, method="grid", num=num)
    _, ok = passes(methods.joint_metrics(joints, lambda j: group_table(j, a_levels)))
    ok = ok[list(RULES)]
    _, t_ok = passes(group_table(true_joint(prim, a_true), a_levels)[None])
    sep = separate_verdicts(separate_rate_bounds(prim, aux, a_levels))
    alone_sep = {r: (sep[r][0] != "fail") if r in sep else None for r in RULES}
    rows = []
    for r1, r2 in itertools.combinations(RULES, 2):
        both = (ok[r1] == 1) & (ok[r2] == 1)
        rows.append({
            "setting": name, "rule_1": r1, "rule_2": r2,
            "joints_alone_1": bool((ok[r1] == 1).any()), "joints_alone_2": bool((ok[r2] == 1).any()),
            "joints_together": bool(both.any()), "n_joints_together": int(both.sum()),
            "n_joints": len(ok),
            "separate_alone_both": (None if alone_sep[r1] is None or alone_sep[r2] is None
                                    else alone_sep[r1] and alone_sep[r2]),
            "truth_both": bool(t_ok[r1].iloc[0] == 1 and t_ok[r2].iloc[0] == 1)})
    t = pd.DataFrame(rows)
    t["incompatible"] = t["joints_alone_1"] & t["joints_alone_2"] & ~t["joints_together"]
    print(f"{name}: {len(ok)} joints, {int(t['incompatible'].sum())} pairs possible alone "
          f"but not together")
    return t


# ------------------------------------------------------------------ figure
INK, MUTED = "#1f1f1e", "#6b6a63"
STYLE = {"together": ("#0ca30c", "✓", "white"),       # some joint passes both
         "incompatible": ("#d03b3b", "✗", "white"),   # each alone yes, together no
         "not alone": ("#ecebe6", "–", MUTED)}        # at least one rule can never pass


def plot(tables, path):
    plt.rcParams.update({"font.size": 8, "axes.titlecolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED})
    rules = list(RULES)
    n = len(tables)
    ncol = min(n, 4)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 4.1 * nrow + 0.6), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, t in zip(axes, tables):
        for _, r in t.iterrows():
            i, j = rules.index(r["rule_1"]), rules.index(r["rule_2"])
            state = ("together" if r["joints_together"] else
                     "incompatible" if r["incompatible"] else "not alone")
            face, sym, col = STYLE[state]
            for a, b in ((i, j), (j, i)):
                ax.add_patch(plt.Rectangle((b - 0.46, a - 0.46), 0.92, 0.92, color=face))
                ax.text(b, a, sym, ha="center", va="center", color=col, fontsize=8)
        ax.set_xlim(-0.5, len(rules) - 0.5)
        ax.set_ylim(len(rules) - 0.5, -0.5)
        ax.set_xticks(range(len(rules)), rules, rotation=55, ha="right")
        ax.set_yticks(range(len(rules)), rules)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(t["setting"].iloc[0], loc="left", fontsize=10)
    for ax in axes[n:]:
        ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=STYLE[s][0]) for s in STYLE]
    fig.legend(handles, ["✓ some feasible joint passes both",
                         "✗ each can pass alone, no joint found passing both",
                         "– one rule can never pass"], loc="outside lower center", ncol=3, fontsize=9)
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-synthetic", type=int, default=3)
    ap.add_argument("--num-german", type=int, default=5)
    ap.add_argument("--strengths", type=float, nargs="+", default=[0.5, 0.8, 0.9, 0.95])
    ap.add_argument("--out-dir", default="outputs/simultaneous_audit")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tables = [analyse("synthetic", *synthetic(), args.num_synthetic),
              analyse("german", *from_config("configs/german.yaml"), args.num_german)]
    tables += [analyse(f"sweep s={s}", *sweep_data(s), args.num_synthetic) for s in args.strengths]
    table = pd.concat(tables)
    table.to_csv(os.path.join(args.out_dir, "pairs.csv"), index=False)
    plot(tables, os.path.join(args.out_dir, "pairs.png"))

    pd.set_option("display.width", 250)
    inc = table[table["incompatible"]]
    print("\nPossible alone, never together:")
    print(inc[["setting", "rule_1", "rule_2", "separate_alone_both", "truth_both"]].to_string(index=False)
          if len(inc) else "  none")
    print("\nPairs achievable together (of 36):")
    print(table.groupby("setting", sort=False)["joints_together"].sum().to_string())
