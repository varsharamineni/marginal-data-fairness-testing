"""Step 2: create Yhat with a classifier when the primary data does not already have it.

The primary data is split: the classifier is fit on `train_fraction` of the rows
and Yhat is predicted on the remaining rows, which are returned and used for all
fairness computations.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

MODELS = {
    "logistic": lambda seed: LogisticRegression(max_iter=1000),
    "random_forest": lambda seed: RandomForestClassifier(n_estimators=200, random_state=seed),
}


def add_yhat(primary, features, cfg):
    if cfg["model"] not in MODELS:
        raise ValueError(f"classifier.model must be one of {list(MODELS)}")
    rng = np.random.default_rng(cfg["seed"])
    idx = rng.permutation(len(primary))
    n_train = int(round(cfg["train_fraction"] * len(primary)))
    train, held_out = primary.iloc[idx[:n_train]], primary.iloc[idx[n_train:]].copy()

    model = MODELS[cfg["model"]](cfg["seed"]).fit(train[features], train["y"])
    held_out["yhat"] = (model.predict_proba(held_out[features])[:, 1] >= cfg["threshold"]).astype(int)
    accuracy = float((held_out["yhat"] == held_out["y"]).mean())
    return held_out.reset_index(drop=True), {"model": cfg["model"], "n_train": n_train,
                                             "n_held_out": len(held_out), "held_out_accuracy": accuracy}
