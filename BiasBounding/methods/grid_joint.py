"""
Grid enumeration of joint distributions p(internal, common, external) consistent with
two datasets:

    primary dataset    (e.g. the lender's data):     internal + common variables
    auxiliary dataset  (e.g. census / survey data):  common + external variables

By default the common variables are the columns both files share, the internal variables are
the remaining columns of the primary dataset, and the external variables the remaining columns
of the auxiliary dataset; each group can also be set explicitly. Several columns in a group are
treated as one combined variable, and numeric columns with many values are binned.
P(common) and p(external | common) come from the auxiliary dataset, p(internal | common) from
the primary dataset.

For each common value c, the unknown is a table p(internal, c, external) whose row sums are
p(internal | c) P(c) and column sums p(c, external). Its free entries (all but the last row and
column) are varied over `num` grid values:
  --method sweep       each free entry over its own Frechet-Hoeffding range; invalid
                       combinations are discarded (exact for binary internal and external)
  --method sequential  entries filled one at a time, each over its range given those already
                       set, so every grid point is valid
Valid tables for the different common values are then combined in every possible way.

Command line:
  python grid_joint.py primary.csv auxiliary.csv
  python grid_joint.py primary.csv auxiliary.csv --internal outcome --common income_bin \
         --external race sex --num 3 --decision outcome --positive 1
Python:
  from grid_joint import solve_joint_distribution
  joints = solve_joint_distribution(p_primary, p_aux, num=50)
      # p_primary[(internal, common)], p_aux[(common, external)]: dicts of probabilities/counts
"""
import argparse
import itertools
import math
import os

import numpy as np
import pandas as pd

TOL = 1e-9


# ============================================================================ core
def _name(i, c, e):
    """'p011' (internal, common, external) for single-character values, else a long form."""
    parts = [str(v) for v in (i, c, e)]
    if all(len(p) == 1 for p in parts):
        return "p" + "".join(parts)
    return f"p(internal={i},common={c},external={e})"


def cell_tables(row_sums, col_sums, num, method="sweep", min_value=0.0, max_value=1.0,
                cell_constraints=()):
    """Valid grid tables for one common value: rows = internal values, cols = external values."""
    m, n, total = len(row_sums), len(col_sums), row_sums.sum()
    if method == "sequential":
        states = [(np.zeros((m, n)), row_sums.copy(), col_sums.copy())]
        for i in range(m - 1):
            for j in range(n):
                new = []
                for X, rr, cc in states:
                    if j == n - 1:
                        vals = [rr[i]]
                    else:
                        hi = min(rr[i], cc[j])
                        lo = max(0.0, rr[i] - cc[j + 1:].sum(), cc[j] - rr[i + 1:].sum())
                        vals = [lo] if hi <= lo + 1e-15 else np.linspace(lo, hi, num)
                    for v in vals:
                        X2 = X.copy(); X2[i, j] = v
                        r2 = rr.copy(); r2[i] -= v
                        c2 = cc.copy(); c2[j] -= v
                        new.append((X2, r2, c2))
                states = new
        X = np.array([np.vstack([S[:m - 1], cc[None]]) for S, _, cc in states])
    else:
        free = [(i, j) for i in range(m - 1) for j in range(n - 1)]
        grids = [np.linspace(max(0.0, row_sums[i] + col_sums[j] - total),
                             min(row_sums[i], col_sums[j]), num) for i, j in free]
        pts = np.array(list(itertools.product(*grids))) if free else np.zeros((1, 0))
        X = np.zeros((len(pts), m, n))
        for f, (i, j) in enumerate(free):
            X[:, i, j] = pts[:, f]
        X[:, :m - 1, n - 1] = row_sums[:m - 1] - X[:, :m - 1, :n - 1].sum(2)   # last column
        X[:, m - 1, :] = col_sums - X[:, :m - 1, :].sum(1)                     # last row
    ok = (X >= min_value - TOL).all((1, 2)) & (X <= max_value + TOL).all((1, 2))
    X = np.maximum(X[ok], 0)
    for con in cell_constraints:
        X = X[[bool(con(x)) for x in X]]
    return np.unique(np.round(X, 15), axis=0)


def solve_joint_distribution(p_primary, p_aux, num=50, method="sweep", min_value=0.0,
                             max_value=1.0, cell_constraints=(), joint_constraints=(),
                             max_joints=5_000_000, as_dicts=True):
    """
    p_primary  dict (internal, common) -> probability or count   (primary dataset)
    p_aux      dict (common, external) -> probability or count   (auxiliary dataset)
    cell_constraints   functions f(table) -> bool on one common value's table [internal, external]
    joint_constraints  functions f(joint) -> bool on a full joint [common, internal, external]
    Returns a list of dicts with symbolic keys, or (array [N, common, internal, external],
    (internal_values, common_values, external_values)) if as_dicts=False.
    """
    srt = lambda vals: sorted(vals, key=str)
    I = srt({k[0] for k in p_primary})
    C = srt({k[0] for k in p_aux})
    E = srt({k[1] for k in p_aux})
    Paux = np.array([[p_aux.get((c, e), 0) for e in E] for c in C], float)        # [c, e]
    Pprim = np.array([[p_primary.get((i, c), 0) for i in I] for c in C], float)   # [c, i]
    keep = Pprim.sum(1) > 0
    if (~keep).any():
        print(f"note: dropping common values {[C[k] for k in np.where(~keep)[0]]} "
              "(no records in the primary dataset)")
        C, Paux, Pprim = [c for c, k in zip(C, keep) if k], Paux[keep], Pprim[keep]
    Paux = Paux / Paux.sum()                                                # p(c, e)
    row_sums = Pprim / Pprim.sum(1, keepdims=True) * Paux.sum(1, keepdims=True)  # p(i | c) P(c)

    m, n = len(I), len(E)
    F = len(C) * (m - 1) * (n - 1)
    print(f"|internal|={m}, |common|={len(C)}, |external|={n}: {F} free parameters, "
          f"num={num} -> at most {float(num) ** F:.3g} grid points")
    if float(num) ** ((m - 1) * (n - 1)) > max_joints:
        raise SystemExit("Too many grid points per common value: use fewer internal or "
                         "external values, or a smaller num.")
    tables = [cell_tables(row_sums[k], Paux[k], num, method, min_value, max_value,
                          cell_constraints) for k in range(len(C))]
    total = math.prod(len(t) for t in tables)
    print(f"valid tables per common value: {[len(t) for t in tables]} -> {total:,} joints")
    if total > max_joints:
        raise SystemExit(f"{total:,} joints > max_joints {max_joints:,}: use fewer common, "
                         "internal or external values, or a smaller num.")
    if total == 0:
        raise SystemExit("no valid joints (constraints too strict?)")
    J = np.zeros((1, 0, m, n))
    for t in tables:                                                        # all combinations
        J = np.concatenate([np.repeat(J, len(t), 0), np.tile(t, (len(J), 1, 1))[:, None]], axis=1)
    for con in joint_constraints:
        J = J[[bool(con(x)) for x in J]]
    if not as_dicts:
        return J, (I, C, E)
    return [{_name(i, c, e): float(x[k, a, b]) for k, c in enumerate(C)
             for a, i in enumerate(I) for b, e in enumerate(E)} for x in J]


# ============================================================================ datasets
def _discretise(primary, aux, cols, max_categories, bins):
    """Bin numeric columns with many values, with the same edges in both datasets."""
    for col in cols:
        vals = pd.concat([d[col] for d in (primary, aux) if col in d.columns])
        if pd.api.types.is_numeric_dtype(vals) and vals.nunique() > max_categories:
            edges = np.unique(np.quantile(vals.dropna(), np.linspace(0, 1, bins + 1)))
            for d in (primary, aux):
                if col in d.columns:
                    d[col] = pd.cut(d[col], edges, include_lowest=True).astype(str)
            print(f"binned numeric column '{col}' into {len(edges) - 1} quantile bins")
    return primary, aux


def from_datasets(primary, aux, internal=None, common=None, external=None,
                  weight_primary=None, weight_aux=None, max_categories=20, bins=5):
    """Build p_primary[(internal, common)] and p_aux[(common, external)] from two data frames."""
    w = (weight_primary, weight_aux)
    common = common or [c for c in primary.columns if c in aux.columns and c not in w]
    internal = internal or [c for c in primary.columns if c not in common and c not in w]
    external = external or [c for c in aux.columns if c not in common and c not in w]
    if not common:
        raise SystemExit("the datasets share no columns; pass --common")
    for name, df, cols in (("primary", primary, internal + common),
                           ("auxiliary", aux, common + external)):
        miss = [c for c in cols if c not in df.columns]
        if miss:
            raise SystemExit(f"{name} dataset has no column(s) {miss}; it has {list(df.columns)}")
    print(f"internal = {internal} | common = {common} | external = {external}")
    primary, aux = _discretise(primary.copy(), aux.copy(), internal + common + external,
                               max_categories, bins)

    def counts(df, g1, g2, weight):
        s = df.assign(_w=df[weight] if weight else 1).groupby(g1 + g2, observed=True)["_w"].sum()
        out = {}
        for k, v in s.items():
            k = k if isinstance(k, tuple) else (k,)
            a, b = k[:len(g1)], k[len(g1):]
            out[(a if len(a) > 1 else a[0], b if len(b) > 1 else b[0])] = v
        return out
    p_primary = counts(primary, internal, common, weight_primary)          # (internal, common)
    p_aux = counts(aux, common, external, weight_aux)                      # (common, external)
    return p_primary, p_aux, internal, common, external


# ============================================================================ command line
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("primary", help="CSV with internal + common variables")
    ap.add_argument("auxiliary", help="CSV with common + external variables")
    ap.add_argument("--internal", nargs="+")
    ap.add_argument("--common", nargs="+")
    ap.add_argument("--external", nargs="+")
    ap.add_argument("--weight-primary")
    ap.add_argument("--weight-aux", help="e.g. 'count' for aggregated auxiliary data")
    ap.add_argument("--num", type=int, default=5, help="grid values per free parameter")
    ap.add_argument("--method", choices=["sweep", "sequential"], default="sweep")
    ap.add_argument("--min-value", type=float, default=0.0)
    ap.add_argument("--max-value", type=float, default=1.0)
    ap.add_argument("--max-categories", type=int, default=20, help="bin numeric columns above this")
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--max-joints", type=int, default=5_000_000)
    ap.add_argument("--decision", help="internal column for group rates, e.g. outcome")
    ap.add_argument("--positive", default="1", help="favourable value of --decision")
    ap.add_argument("--out", default="grid_joint_results")
    a = ap.parse_args()

    def read(path, groups, weight):
        cols = None if any(g is None for g in groups) else sum(groups, []) + ([weight] if weight else [])
        return pd.read_csv(path, usecols=cols)
    primary = read(a.primary, [a.internal, a.common], a.weight_primary)
    aux = read(a.auxiliary, [a.common, a.external], a.weight_aux)
    p_primary, p_aux, internal, common, external = from_datasets(
        primary, aux, a.internal, a.common, a.external, a.weight_primary, a.weight_aux,
        a.max_categories, a.bins)
    J, (I, C, E) = solve_joint_distribution(p_primary, p_aux, a.num, a.method, a.min_value,
                                            a.max_value, max_joints=a.max_joints, as_dicts=False)

    os.makedirs(a.out, exist_ok=True)
    names = [_name(i, c, e) for c in C for i in I for e in E]
    flat = J.reshape(len(J), -1)
    np.savez_compressed(os.path.join(a.out, "joints.npz"), joints=J,
                        internal_values=np.array(I, dtype=object),
                        common_values=np.array(C, dtype=object),
                        external_values=np.array(E, dtype=object))
    if len(J) <= 200_000:
        pd.DataFrame(flat, columns=names).to_csv(os.path.join(a.out, "joints.csv"), index=False)
    pd.DataFrame({"entry": names, "min": flat.min(0), "max": flat.max(0)}).to_csv(
        os.path.join(a.out, "entry_ranges.csv"), index=False)
    print(f"saved {len(J):,} joints to {a.out}/")

    if a.decision:                       # bounds on P(decision = positive | external group)
        d = internal.index(a.decision)
        pos = np.array([str(i[d] if isinstance(i, tuple) else i) == a.positive for i in I])
        theta = J.sum(1)                                                   # [N, internal, external]
        rate = theta[:, pos, :].sum(1) / theta.sum(1)
        rows = [{"quantity": f"P({a.decision}={a.positive} | external={e})",
                 "lower": rate[:, k].min(), "upper": rate[:, k].max()} for k, e in enumerate(E)]
        rows += [{"quantity": f"DD({E[g]} - {E[h]})", "lower": (rate[:, g] - rate[:, h]).min(),
                  "upper": (rate[:, g] - rate[:, h]).max()}
                 for g, h in itertools.combinations(range(len(E)), 2)]
        res = pd.DataFrame(rows)
        res.to_csv(os.path.join(a.out, "group_rate_bounds.csv"), index=False)
        print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()