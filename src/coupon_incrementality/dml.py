"""Phase 3a - double machine learning for Type A campaigns.

Type A coupons were targeted on purchase history, and that history is in the data. So we
compare recipients with non-recipients *who had the same history*: gradient-boosted models
predict both who gets targeted and how spending would have changed anyway, and double ML
combines the two so that what is left is a fair comparison.

Unit: household x campaign (campaigns 8, 13 and 18 stacked). Sample splits and standard
errors are clustered by household, because the same household appears up to three times.
The outcome is the change in average weekly spend from the household's own 12-week
pre-campaign average.
"""

from __future__ import annotations

import doubleml as dml
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict

from . import aipw
from .config import PROPENSITY_BOUNDS, ROOT, SEED, TYPE_A_CAMPAIGNS, save_result
from .features import build_frame, feature_columns

FRAME_PATH = ROOT / "data" / "type_a_frame.parquet"
N_FOLDS = 5
N_REP = 5
# Benchmarks for the sensitivity analysis: how much would the estimate move if we had
# *not* observed these feature blocks? An unobserved confounder would have to matter
# about as much to move it the same distance.
BENCHMARKS = {
    "all spending history": lambda c: c.startswith(("spend_", "trips_", "units_", "basket_")),
    "deal and coupon behaviour": lambda c: c.startswith(
        ("deal_", "discount", "coupon_", "retail_")
    ),
    "earlier campaigns": lambda c: c.startswith(("prior_", "overlapping_")),
}


def regressor() -> LGBMRegressor:
    return LGBMRegressor(**aipw.LGBM, random_state=SEED)


def classifier() -> LGBMClassifier:
    return LGBMClassifier(**aipw.LGBM, random_state=SEED)


def add_propensity(frame: pd.DataFrame) -> pd.DataFrame:
    """Out-of-fold probability of being targeted; folds never split a household."""
    frame = frame.copy()
    frame["propensity"] = cross_val_predict(
        classifier(),
        frame[feature_columns(frame)],
        frame.d,
        groups=frame.household_id,
        cv=GroupKFold(n_splits=N_FOLDS),
        method="predict_proba",
    )[:, 1]
    low, high = PROPENSITY_BOUNDS
    frame["in_support"] = frame.propensity.between(low, high)
    return frame


def fit(frame: pd.DataFrame, outcome: str, model: str, cluster: bool = True):
    """Fit one DoubleML model: 'att' / 'ate' (doubly robust) or 'plr' (partially linear)."""
    data = dml.DoubleMLData(
        frame,
        y_col=outcome,
        d_cols="d",
        x_cols=feature_columns(frame),
        cluster_cols="household_id" if cluster else None,
    )
    np.random.seed(SEED)
    if model == "plr":
        estimator = dml.DoubleMLPLR(data, regressor(), classifier(), n_folds=N_FOLDS, n_rep=N_REP)
    else:
        estimator = dml.DoubleMLIRM(
            data,
            regressor(),
            classifier(),
            n_folds=N_FOLDS,
            n_rep=N_REP,
            score="ATTE" if model == "att" else "ATE",
            ps_processor_config=dml.utils.PSProcessorConfig(clipping_threshold=aipw.CLIP),
        )
    return estimator.fit()


def summary(fitted) -> dict:
    low, high = fitted.confint().iloc[0]
    return {
        "estimate": float(fitted.coef[0]),
        "se": float(fitted.se[0]),
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": float(fitted.pval[0]),
    }


def naive(frame: pd.DataFrame) -> dict:
    treated, control = frame[frame.d == 1].y, frame[frame.d == 0].y
    return {
        "recipients": float(treated.mean()),
        "non_recipients": float(control.mean()),
        "difference": float(treated.mean() - control.mean()),
        "lift_pct": float(100 * (treated.mean() / control.mean() - 1)),
    }


def sensitivity(plr, frame: pd.DataFrame) -> dict:
    """How strong would an unobserved confounder need to be to explain the estimate away?

    Uses the omitted-variable-bias bounds of Chernozhukov, Cinelli, Newey, Sharma &
    Syrgkanis (2022), the double-ML generalisation of Cinelli & Hazlett's sensemakr.
    The robustness value (RV) is the share of residual variation in both treatment and
    outcome that a confounder would have to explain to bring the estimate to zero.
    """
    plr.sensitivity_analysis(cf_y=0.03, cf_d=0.03, rho=1.0)
    params = plr.sensitivity_params
    out = {
        "robustness_value_pct": float(100 * params["rv"][0]),
        "robustness_value_ci_pct": float(100 * params["rva"][0]),
        "benchmarks": {},
    }
    features = feature_columns(frame)
    for name, belongs in BENCHMARKS.items():
        block = [c for c in features if belongs(c)]
        np.random.seed(SEED)
        bench = plr.sensitivity_benchmark(benchmarking_set=block)
        out["benchmarks"][name] = {
            "n_features": len(block),
            "cf_y_pct": float(100 * bench["cf_y"].iloc[0]),
            "cf_d_pct": float(100 * bench["cf_d"].iloc[0]),
            "estimate_shift": float(bench["delta_theta"].iloc[0]),
        }
    return out


def run() -> dict:
    frame = add_propensity(build_frame())
    frame.to_parquet(FRAME_PATH, index=False)
    trimmed = frame[frame.in_support].reset_index(drop=True)
    recipients = trimmed[trimmed.d == 1]

    result = {
        "unit": "dollars of in-store spend per household per campaign week",
        "campaigns": list(TYPE_A_CAMPAIGNS),
        "n_rows": len(frame),
        "n_features": len(feature_columns(frame)),
        "n_recipient_rows": int(frame.d.sum()),
        "overlap": {
            "bounds": list(PROPENSITY_BOUNDS),
            "n_trimmed": int((~frame.in_support).sum()),
            "share_trimmed": float(1 - frame.in_support.mean()),
            "recipients_kept": int(len(recipients)),
            "share_recipients_kept": float(len(recipients) / frame.d.sum()),
            "non_recipients_kept": int((trimmed.d == 0).sum()),
        },
        "mean_campaign_weeks": float(recipients.n_weeks.mean()),
        "recipient_mean_spend": float(recipients.y.mean()),
        "coupon_cost_per_recipient": float(recipients.coupon_cost.mean()),
        "naive_all": naive(frame),
        "naive_trimmed": naive(trimmed),
    }

    plr = fit(trimmed, "y_change", "plr")
    result["during"] = {
        "att": summary(fit(trimmed, "y_change", "att")),
        "plr": summary(plr),
        "ate": summary(fit(trimmed, "y_change", "ate")),
        "att_untrimmed": summary(fit(frame, "y_change", "att")),
        "att_levels": summary(fit(trimmed, "y", "att")),
    }
    result["sensitivity"] = sensitivity(plr, trimmed)
    result["by_campaign"] = {
        str(c): summary(fit(trimmed[trimmed.campaign_id == c], "y_change", "att", cluster=False))
        | {"n_recipients": int(recipients[recipients.campaign_id == c].shape[0])}
        for c in TYPE_A_CAMPAIGNS
    }
    # Pull-forward: spend in the four weeks after the campaign ended. Campaign 18 ends one
    # week before the data does, so only campaigns with a full post window are used.
    post = trimmed[trimmed.n_post_weeks == trimmed.n_post_weeks.max()].reset_index(drop=True)
    result["post"] = {
        "campaigns": sorted(int(c) for c in post.campaign_id.unique()),
        "weeks": int(post.n_post_weeks.max()),
        "att": summary(fit(post, "y_post_change", "att")),
    }
    # Cross-check: the hand-written AIPW engine used by later phases should agree.
    result["aipw_cross_check"] = aipw.att_repeated(
        trimmed, feature_columns(trimmed), ["y_change"], n_rep=N_REP
    )["y_change"]
    save_result("dml", result)
    return result


if __name__ == "__main__":
    out = run()
    for key in (
        "overlap",
        "naive_all",
        "during",
        "by_campaign",
        "post",
        "sensitivity",
        "aipw_cross_check",
    ):
        print(key, out[key])
