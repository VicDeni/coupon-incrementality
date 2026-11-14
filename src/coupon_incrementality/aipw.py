"""Cross-fitted doubly robust (AIPW) building blocks.

DoubleML gives the headline Type A estimate. The later phases need things it does not
expose conveniently: one propensity model shared across several outcomes, per-household
doubly robust scores for testing heterogeneity, and off-policy evaluation. Those all rest
on the same three ingredients, computed here:

    m(x)   = P(targeted | x)                 propensity
    g0(x)  = E[outcome | x, not targeted]    baseline outcome model
    g1(x)  = E[outcome | x, targeted]        treated outcome model

Every prediction for a household comes from models that never saw that household
(cross-fitting, with folds split by household).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import StratifiedGroupKFold

from .config import SEED

CLIP = 0.02  # propensities are clipped to [CLIP, 1 - CLIP]
LGBM = dict(
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=30,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.5,
    reg_lambda=1.0,
    n_jobs=2,
    verbose=-1,
)


def cross_fit(
    frame: pd.DataFrame,
    features: list[str],
    outcomes: list[str],
    treated_model: bool = False,
    n_folds: int = 5,
    seed: int = SEED,
) -> pd.DataFrame:
    """Out-of-fold nuisance predictions.

    Returns a frame aligned with `frame` holding `fold`, `m`, and for each outcome
    `g0_<outcome>` (plus `g1_<outcome>` when `treated_model` is set).
    """
    x, d = frame[features].to_numpy(), frame["d"].to_numpy()
    out = pd.DataFrame(index=frame.index, dtype=float)
    out["fold"] = -1
    out["m"] = np.nan
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (train, test) in enumerate(splitter.split(x, d, groups=frame["household_id"])):
        out.iloc[test, out.columns.get_loc("fold")] = fold
        propensity = LGBMClassifier(**LGBM, random_state=seed).fit(x[train], d[train])
        out.iloc[test, out.columns.get_loc("m")] = propensity.predict_proba(x[test])[:, 1]
        arms = [(0, "g0")] + ([(1, "g1")] if treated_model else [])
        for outcome in outcomes:
            y = frame[outcome].to_numpy()
            for arm, prefix in arms:
                rows = train[d[train] == arm]
                model = LGBMRegressor(**LGBM, random_state=seed).fit(x[rows], y[rows])
                out.loc[frame.index[test], f"{prefix}_{outcome}"] = model.predict(x[test])
    out["m"] = out["m"].clip(CLIP, 1 - CLIP)
    return out


def cluster_se(influence: np.ndarray, clusters: np.ndarray) -> float:
    """Standard error of a mean-zero influence function, clustered."""
    sums = pd.Series(influence).groupby(np.asarray(clusters)).sum().to_numpy()
    return float(np.sqrt((sums**2).sum()) / len(influence))


def att(y, d, m, g0, clusters) -> dict:
    """Average effect on the treated (doubly robust), with a clustered standard error."""
    y, d, m, g0 = (np.asarray(a, dtype=float) for a in (y, d, m, g0))
    weight = d - (1 - d) * m / (1 - m)
    share = d.mean()
    estimate = (weight * (y - g0)).mean() / share
    influence = (weight * (y - g0) - d * estimate) / share
    se = cluster_se(influence, clusters)
    return {
        "estimate": float(estimate),
        "se": se,
        "ci_low": float(estimate - 1.96 * se),
        "ci_high": float(estimate + 1.96 * se),
    }


def dr_scores(y, d, m, g0, g1) -> np.ndarray:
    """Per-row doubly robust score; its conditional mean is the treatment effect."""
    y, d, m, g0, g1 = (np.asarray(a, dtype=float) for a in (y, d, m, g0, g1))
    return g1 - g0 + d * (y - g1) / m - (1 - d) * (y - g0) / (1 - m)


def att_repeated(
    frame: pd.DataFrame, features: list[str], outcomes: list[str], n_rep: int = 3
) -> dict:
    """ATT for several outcomes, repeated over sample splits and median-aggregated.

    The reported standard error adds the split-to-split spread to the sampling variance,
    as in Chernozhukov et al. (2018).
    """
    draws: dict[str, list[dict]] = {o: [] for o in outcomes}
    for rep in range(n_rep):
        fitted = cross_fit(frame, features, outcomes, seed=SEED + rep)
        for outcome in outcomes:
            draws[outcome].append(
                att(
                    frame[outcome],
                    frame["d"],
                    fitted["m"],
                    fitted[f"g0_{outcome}"],
                    frame["household_id"],
                )
            )
    result = {}
    for outcome, reps in draws.items():
        estimates = np.array([r["estimate"] for r in reps])
        variances = np.array([r["se"] ** 2 for r in reps])
        estimate = float(np.median(estimates))
        se = float(np.sqrt(np.median(variances + (estimates - estimate) ** 2)))
        result[outcome] = {
            "estimate": estimate,
            "se": se,
            "ci_low": estimate - 1.96 * se,
            "ci_high": estimate + 1.96 * se,
        }
    return result
