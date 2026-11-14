"""Phase 5 - who actually responds?

A causal forest estimates a coupon effect for every household. Those individual estimates
are noisy, so before believing them we test them, in the spirit of Chernozhukov, Demirer,
Duflo & Fernandez-Val (2018):

  * BLP (best linear predictor): regress each household's doubly robust effect score on
    the forest's prediction. A slope clearly above zero means the forest has found real
    differences; a slope near one means its predictions are also the right size.
  * GATES (sorted group average treatment effects): sort households into fifths by
    predicted effect and measure the actual average effect in each fifth. If the top
    fifth does not beat the bottom fifth, the "segments" are noise.

Both tests are run out of sample: the forest that ranks a household never saw that
household, and neither did the nuisance models behind its effect score.

The tests use doubly robust scores rather than the original paper's known randomisation
probabilities, because this is observational data (Semenova & Chernozhukov, 2021).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from econml.dml import CausalForestDML
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.stats import norm

from . import aipw
from .config import ROOT, SEED, save_result
from .dml import FRAME_PATH
from .features import feature_columns

SCORES_PATH = ROOT / "data" / "household_scores.parquet"
N_GROUPS = 5
# Household traits the forest may use to explain differences in response. Everything
# else in the feature set still serves as a control.
MODIFIERS = {
    "spend_mean_12w": "Weekly spend ($)",
    "trips_mean_12w": "Trips per week",
    "basket_size": "Basket size ($)",
    "deal_share": "Share of spend bought on deal",
    "discount_depth": "Average loyalty discount",
    "coupon_disc_12w": "Other coupon dollars used (12 wk)",
    "private_label_share": "Private-label share",
    "n_categories": "Categories shopped",
    "spend_trend": "Spend trend ($/wk per wk)",
    "share_weeks_shopped": "Share of weeks with a trip",
    "prior_a": "Earlier Type A campaigns",
    "prior_redemptions": "Earlier coupon redemptions",
}


def out_of_fold_cate(frame: pd.DataFrame, folds: pd.Series) -> np.ndarray:
    """Forest predictions for each household from a forest trained on the other folds."""
    modifiers = list(MODIFIERS)
    controls = [c for c in feature_columns(frame) if c not in modifiers]
    predictions = np.full(len(frame), np.nan)
    for fold in sorted(folds.unique()):
        train, test = frame[folds != fold], frame[folds == fold]
        forest = CausalForestDML(
            model_y=LGBMRegressor(**aipw.LGBM, random_state=SEED),
            model_t=LGBMClassifier(**aipw.LGBM, random_state=SEED),
            discrete_treatment=True,
            cv=3,
            n_estimators=1000,
            min_samples_leaf=25,
            max_samples=0.4,
            random_state=SEED,
        )
        forest.fit(
            train.y_change,
            train.d,
            X=train[modifiers],
            W=train[controls],
            groups=train.household_id,
        )
        predictions[(folds == fold).to_numpy()] = forest.effect(test[modifiers])
    return predictions


def _clustered_ols(y, x, clusters):
    return sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": clusters})


def blp(scores: np.ndarray, cate: np.ndarray, clusters) -> dict:
    design = sm.add_constant(pd.DataFrame({"prediction": cate - cate.mean()}))
    fit = _clustered_ols(scores, design, clusters)
    return {
        "average_effect": float(fit.params["const"]),
        "average_effect_se": float(fit.bse["const"]),
        "slope": float(fit.params["prediction"]),
        "slope_se": float(fit.bse["prediction"]),
        "slope_p_value_one_sided": float(norm.sf(fit.tvalues["prediction"])),
    }


def gates(scores: np.ndarray, groups: np.ndarray, clusters) -> dict:
    dummies = pd.get_dummies(groups, dtype=float)
    fit = _clustered_ols(scores, dummies, clusters)
    top, bottom = dummies.columns[-1], dummies.columns[0]
    contrast = np.zeros(len(dummies.columns))
    contrast[-1], contrast[0] = 1, -1
    test = fit.t_test(contrast)
    return {
        "group": [int(g) + 1 for g in dummies.columns],
        "effect": fit.params.tolist(),
        "se": fit.bse.tolist(),
        "top_minus_bottom": float(fit.params[top] - fit.params[bottom]),
        "top_minus_bottom_se": float(test.sd.item()),
        "top_minus_bottom_p_value": float(test.pvalue.item()),
    }


def run() -> dict:
    frame = pd.read_parquet(FRAME_PATH)
    frame = frame[frame.in_support].reset_index(drop=True)
    features = feature_columns(frame)

    nuisance = aipw.cross_fit(frame, features, ["y_change", "coupon_cost"], treated_model=True)
    score = aipw.dr_scores(
        frame.y_change, frame.d, nuisance.m, nuisance.g0_y_change, nuisance.g1_y_change
    )
    # Coupon dollars are zero unless targeted, so the cost "effect" score needs no g0.
    cost_score = (
        nuisance.g1_coupon_cost
        + frame.d * (frame.coupon_cost - nuisance.g1_coupon_cost) / nuisance.m
    )
    cate = out_of_fold_cate(frame, nuisance.fold)

    # Rank within fold: each fold's households were scored by a different forest.
    rank = pd.Series(cate).groupby(nuisance.fold.to_numpy()).rank(pct=True)
    group = np.minimum((rank * N_GROUPS).astype(int), N_GROUPS - 1).to_numpy()

    profile = frame.assign(group=group).groupby("group")[list(MODIFIERS)].mean()
    result = {
        "unit": "dollars of in-store spend per household per campaign week",
        "n_rows": int(len(frame)),
        "predicted_effect": {
            "mean": float(cate.mean()),
            "p10": float(np.quantile(cate, 0.1)),
            "p50": float(np.quantile(cate, 0.5)),
            "p90": float(np.quantile(cate, 0.9)),
        },
        "blp": blp(score, cate, frame.household_id),
        "gates": gates(score, group, frame.household_id),
        "profile": {
            MODIFIERS[c]: {
                "bottom_fifth": float(profile.loc[0, c]),
                "top_fifth": float(profile.loc[N_GROUPS - 1, c]),
            }
            for c in MODIFIERS
        },
    }
    save_result("heterogeneity", result)

    keep = ["household_id", "campaign_id", "d", "n_weeks", "coupon_cost", *MODIFIERS]
    frame[keep].assign(
        fold=nuisance.fold.to_numpy(),
        propensity=nuisance.m.to_numpy(),
        predicted_effect=cate,
        predicted_coupon_cost=nuisance.g1_coupon_cost.to_numpy(),
        effect_score=score,
        cost_score=cost_score.to_numpy(),
        response_group=group + 1,
    ).to_parquet(SCORES_PATH, index=False)
    return result


if __name__ == "__main__":
    out = run()
    print(out["predicted_effect"], out["blp"], sep="\n")
    print(pd.DataFrame({k: out["gates"][k] for k in ("group", "effect", "se")}).round(2).T)
    print(
        {
            k: round(out["gates"][k], 3)
            for k in ("top_minus_bottom", "top_minus_bottom_se", "top_minus_bottom_p_value")
        }
    )
    print(pd.DataFrame(out["profile"]).T.round(2))
