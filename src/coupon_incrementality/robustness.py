"""Phase 7 - stress tests.

Each test asks a question the main estimates should survive:

  placebo dates       pretend each campaign ran 8 weeks earlier; there should be no effect
  placebo products    products no coupon touches should show no lift (run inside the
                      main triple difference; repeated here for the summary)
  short baseline      triple difference with a 4-week instead of an 8-week baseline
  one campaign only   drop households juggling several campaigns at once
  hidden confounding  how strong would an unmeasured confounder need to be?

Replication on the two-year dunnhumby release lives in `replicate.py`, because that
dataset has to be downloaded by hand.
"""

from __future__ import annotations

import pandas as pd
import statsmodels.formula.api as smf

from . import aipw, sensemakr, triple_diff
from .config import PROPENSITY_BOUNDS, load_result, save_result
from .dml import FRAME_PATH
from .features import build_frame, feature_columns

PLACEBO_SHIFT_WEEKS = 8


def _in_support(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    m = aipw.cross_fit(frame, features, []).m
    return frame[m.between(*PROPENSITY_BOUNDS)].reset_index(drop=True)


def placebo_dates_dml() -> dict:
    """Type A campaigns moved 8 weeks earlier, with features rebuilt as of the fake start."""
    frame = build_frame(shift_weeks=PLACEBO_SHIFT_WEEKS)
    features = feature_columns(frame)
    frame = _in_support(frame, features)
    return aipw.att_repeated(frame, features, ["y_change"], n_rep=3)["y_change"]


def placebo_dates_triple_diff() -> dict:
    """Type B/C campaigns moved 8 weeks earlier, using only weeks before the real start."""
    panel = triple_diff.load_panel(
        f"rel_week < 0 and week - rel_week >= {PLACEBO_SHIFT_WEEKS + 6}"  # 4+ fake-pre weeks
    )
    fake_during = panel.rel_week >= -PLACEBO_SHIFT_WEEKS
    panel["period"] = fake_during.map({True: "during", False: "pre"}).astype("category")
    terms = triple_diff.add_terms(panel, periods=("during",))
    fit = triple_diff.fit_ppml(panel, terms)
    return {"n_campaigns": int(panel.campaign_id.nunique())} | {t: fit[t] for t in terms}


def short_baseline_triple_diff() -> dict:
    """Triple difference using only the four weeks before each campaign as the baseline.

    The placebo-date test shows recipients' covered-product purchases falling back between
    weeks -16..-9 and -8..-1 (they were targeted for having bought those products). If
    that reversion were still under way just before the campaign, a shorter baseline would
    move the estimate.
    """
    panel = triple_diff.load_panel("rel_week >= -4")
    terms = triple_diff.add_terms(panel)
    fit = triple_diff.fit_ppml(panel, terms)
    return {t: fit[t] for t in terms}


def single_campaign_triple_diff() -> dict:
    """Triple difference on households with no other campaign running at the same time."""
    panel = triple_diff.load_panel("rel_week >= -8 and n_overlapping_campaigns = 0")
    terms = triple_diff.add_terms(panel)
    fit = triple_diff.fit_ppml(panel, terms)
    recipients = panel[panel.rec == 1][["campaign_id", "household_id"]].drop_duplicates()
    return {"n_recipient_campaigns": int(len(recipients))} | {t: fit[t] for t in terms}


def single_campaign_dml() -> dict:
    """Type A effect among households with no Type B/C campaign overlapping."""
    frame = pd.read_parquet(FRAME_PATH)
    frame = frame[frame.in_support & (frame.overlapping_campaigns == 0)].reset_index(drop=True)
    out = aipw.att_repeated(frame, feature_columns(frame), ["y_change"], n_rep=3)["y_change"]
    return out | {"n_recipient_rows": int(frame.d.sum())}


def sensemakr_ols() -> dict:
    """Cinelli & Hazlett (2020) sensitivity analysis on a transparent linear model.

    Benchmarks: `k = 1` asks what would happen if an unobserved confounder were as strongly
    related to targeting and to the outcome as an observed one is (pre-campaign spend, or
    how many Type A campaigns the household had already received); k = 2, 3 scale that up.
    """
    frame = pd.read_parquet(FRAME_PATH)
    frame = frame[frame.in_support]
    controls = [
        "spend_mean_12w", "trips_mean_12w", "deal_share", "coupon_disc_12w",
        "private_label_share", "n_categories", "spend_trend", "prior_a", "prior_b",
        "prior_redemptions", "overlapping_campaigns", "campaign_13", "campaign_18",
    ]  # fmt: skip
    benchmarks = {
        "pre-campaign weekly spend": "spend_mean_12w",
        "earlier Type A campaigns": "prior_a",
    }
    outcome_model = smf.ols(f"y_change ~ d + {' + '.join(controls)}", data=frame).fit()
    treatment_model = smf.ols(f"d ~ {' + '.join(controls)}", data=frame).fit()
    estimate, se = outcome_model.params["d"], outcome_model.bse["d"]
    t_stat, dof = outcome_model.tvalues["d"], int(outcome_model.df_resid)
    bounds = {}
    for label, column in benchmarks.items():
        r2_treatment = sensemakr.partial_r2(
            treatment_model.tvalues[column], int(treatment_model.df_resid)
        )
        r2_outcome = sensemakr.partial_r2(outcome_model.tvalues[column], dof)
        bounds[label] = [
            sensemakr.benchmark_bound(estimate, se, dof, r2_treatment, r2_outcome, k=k)
            for k in (1, 2, 3)
        ]
    return {
        "estimate": float(estimate),
        "se": float(se),
        "partial_r2_treatment_outcome_pct": float(100 * sensemakr.partial_r2(t_stat, dof)),
        "robustness_value_pct": float(100 * sensemakr.robustness_value(t_stat, dof)),
        "robustness_value_ci_pct": float(100 * sensemakr.robustness_value(t_stat, dof, alpha=0.05)),
        "bounds": bounds,
    }


def run() -> dict:
    main = load_result("triple_diff")["main"]
    result = {
        "placebo_dates": {
            "shift_weeks": PLACEBO_SHIFT_WEEKS,
            "dml_type_a": placebo_dates_dml(),
            "triple_diff_type_bc": placebo_dates_triple_diff(),
        },
        "placebo_products": {t: main[t] for t in ("placebo_during", "placebo_post")},
        "short_baseline_triple_diff": short_baseline_triple_diff(),
        "single_campaign": {
            "dml_type_a": single_campaign_dml(),
            "triple_diff_type_bc": single_campaign_triple_diff(),
        },
        "hidden_confounding": {
            "double_ml": load_result("dml")["sensitivity"],
            "sensemakr_ols": sensemakr_ols(),
        },
    }
    save_result("robustness", result)
    return result


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=1, default=float))
