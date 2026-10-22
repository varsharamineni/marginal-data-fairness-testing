"""Step 1: get primary data (Z, Y, [features], [Yhat]) and auxiliary data (Z, A).

Either generate synthetic data or load CSVs, then standardise and validate them.
After `prepare`, the primary frame has columns `z`, `y` (and `yhat` if present,
plus feature columns), and the auxiliary frame has columns `z`, `a`.
"""
import numpy as np
import pandas as pd


# ------------------------------------------------------------------ synthetic
def generate_synthetic(cfg):
    """Return (primary, auxiliary).

    The primary frame keeps the true `a` column so the true disparity can be
    computed for evaluation. run.py removes it before any estimation.
    """
    rng = np.random.default_rng(cfg["seed"])
    K = int(cfg["n_z"])
    p_z = np.full(K, 1.0 / K) if cfg.get("p_z") is None else np.asarray(cfg["p_z"], float)
    p_a = np.linspace(0.1, 0.9, K) if cfg.get("p_a_given_z") is None else np.asarray(cfg["p_a_given_z"], float)
    if len(p_z) != K or len(p_a) != K:
        raise ValueError("p_z and p_a_given_z must have length n_z")
    if not np.isclose(p_z.sum(), 1.0):
        raise ValueError("p_z must sum to 1")

    def draw(n):
        z = rng.choice(K, size=n, p=p_z)
        is_a = rng.random(n) < p_a[z]
        z_c = z / (K - 1) - 0.5 if K > 1 else np.zeros(n)
        logit = cfg["y_intercept"] + cfg["y_coef_a"] * is_a + cfg["y_coef_z"] * z_c
        y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
        x1 = cfg["x1_coef_y"] * y + cfg["x1_coef_a"] * is_a + rng.normal(size=n)
        x2 = cfg["x2_coef_y"] * y + rng.normal(size=n)
        return pd.DataFrame({"z": z, "a": np.where(is_a, "a", "b"), "y": y, "x1": x1, "x2": x2})

    primary = draw(int(cfg["n_primary"]))
    auxiliary = draw(int(cfg["n_auxiliary"]))[["z", "a"]]
    return primary, auxiliary


# ------------------------------------------------------------------ csv
def load_csv(cfg):
    return pd.read_csv(cfg["primary_path"]), pd.read_csv(cfg["auxiliary_path"])


# ------------------------------------------------------------------ processing
def _z_key(df, z_cols):
    if len(z_cols) == 1:
        return df[z_cols[0]].astype(str)
    return df[z_cols].astype(str).agg("|".join, axis=1)


def _check_binary(s, name):
    if not set(pd.unique(s)) <= {0, 1}:
        raise ValueError(f"column '{name}' must be binary 0/1")


def prepare(primary, auxiliary, columns, a_levels, true_a=None):
    """Validate and standardise column names.

    true_a: name of a column in the primary data holding the true protected class
        (synthetic data, or real data where A was hidden for validation). It is kept
        as `a_true` for computing the true disparity and is never used by the methods.
    """
    z_cols = list(columns["z"])
    y, yhat, a = columns["y"], columns["yhat"], columns["a"]
    has_yhat = yhat in primary.columns
    feats = [] if has_yhat else list(columns["features"])

    # dict.fromkeys removes duplicates, so a column can be both a proxy and a feature
    need_p = list(dict.fromkeys(z_cols + [y] + ([yhat] if has_yhat else feats) + ([true_a] if true_a else [])))
    need_a = z_cols + [a]
    for name, df, need in (("primary", primary, need_p), ("auxiliary", auxiliary, need_a)):
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise KeyError(f"{name} data is missing columns: {missing}")

    n0p, n0a = len(primary), len(auxiliary)
    primary = primary[need_p].dropna().copy()
    auxiliary = auxiliary[need_a].dropna().copy()
    if len(primary) < n0p or len(auxiliary) < n0a:
        print(f"Dropped rows with missing values: primary {n0p - len(primary)}, auxiliary {n0a - len(auxiliary)}")

    P = pd.DataFrame({"z": _z_key(primary, z_cols).to_numpy(), "y": primary[y].to_numpy()})
    _check_binary(P["y"], y)
    if has_yhat:
        P["yhat"] = primary[yhat].to_numpy()
        _check_binary(P["yhat"], yhat)
    for f in feats:
        P[f] = primary[f].to_numpy()
    if true_a:
        P["a_true"] = primary[true_a].to_numpy()

    X = pd.DataFrame({"z": _z_key(auxiliary, z_cols).to_numpy(), "a": auxiliary[a].to_numpy()})

    if len(a_levels) != 2:
        raise ValueError("a_levels must contain exactly two levels (binary protected class)")
    extra = set(pd.unique(X["a"])) - set(a_levels)
    if extra:
        raise ValueError(f"auxiliary data has protected-class levels not in a_levels: {extra}")

    unmatched = set(P["z"]) - set(X["z"])
    if unmatched:
        raise ValueError(f"Z levels in primary data but not in auxiliary data "
                         f"(P(A | Z) is not estimable there): {sorted(unmatched)[:10]}")
    return P.reset_index(drop=True), X.reset_index(drop=True)
