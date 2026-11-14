"""The triple-difference estimator recovers a known percentage lift."""

import numpy as np
import pandas as pd
import pytest

from coupon_incrementality import triple_diff

TRUE_LIFT = 0.25  # +25% on covered products for recipients during the campaign


@pytest.fixture(scope="module")
def panel():
    """One campaign; recipients are heavier shoppers who favour the covered products."""
    rng = np.random.default_rng(0)
    rows = []
    group_size = {"covered": 1.0, "sibling": 6.0, "placebo": 8.0, "other": 40.0}
    for household in range(700):
        rec = int(household < 250)
        level = rng.gamma(4, 0.25) * (3.0 if rec else 1.0)
        taste = {
            g: rng.gamma(3, 1 / 3) * (4.0 if rec and g == "covered" else 1.0) for g in group_size
        }
        for week in range(1, 23):
            period = "pre" if week <= 8 else ("during" if week <= 14 else "post")
            shock = rng.gamma(5, 0.2)  # a household-week shock that scales the whole basket
            for group, size in group_size.items():
                mean = size * level * taste[group] * shock * (1 + 0.1 * np.sin(week))
                if rec and group == "covered" and period == "during":
                    mean *= 1 + TRUE_LIFT
                rows.append((1, household, week, group, period, rec, rng.poisson(mean)))
    frame = pd.DataFrame(
        rows,
        columns=["campaign_id", "household_id", "week", "product_group", "period", "rec", "spend"],
    )
    frame["spend"] = frame.spend.astype(float)
    frame["display_share"] = rng.uniform(0, 0.1, len(frame))
    frame["mailer_share"] = rng.uniform(0, 0.1, len(frame))
    frame["rel_week"] = np.where(
        frame.week <= 8, frame.week - 9, np.where(frame.week <= 14, 0, frame.week - 14)
    )
    for name, keys in {
        "hw": ["household_id", "week"], "gw": ["product_group", "week"],
        "hg": ["household_id", "product_group"], "hc": ["household_id"], "wc": ["week"],
    }.items():  # fmt: skip
        frame[name] = frame.groupby(keys).ngroup()
    return frame


@pytest.fixture(scope="module")
def fitted(panel):
    panel = panel.copy()
    terms = triple_diff.add_terms(panel)
    return panel, triple_diff.fit_ppml(panel, terms)


def test_recovers_the_true_lift(fitted):
    _, fit = fitted
    covered = fit["covered_during"]
    assert covered["lift_pct_low"] < 100 * TRUE_LIFT < covered["lift_pct_high"]
    assert covered["lift_pct"] == pytest.approx(100 * TRUE_LIFT, abs=6)


def test_untouched_products_and_periods_show_nothing(fitted):
    _, fit = fitted
    for term in ("covered_post", "sibling_during", "placebo_during", "placebo_post"):
        assert fit[term]["lift_pct_low"] < 0 < fit[term]["lift_pct_high"], term


def test_naive_comparison_overstates_the_lift(panel):
    assert triple_diff.naive(panel)["lift_pct"] > 500


def test_dollar_conversion(fitted):
    panel, fit = fitted
    dollars = triple_diff.to_dollars(panel, fit)
    block = dollars["covered_during"]
    lift = fit["covered_during"]["lift_pct"] / 100
    expected = block["observed_weekly_spend"] * (1 - 1 / (1 + lift)) * dollars["weeks_during"]
    assert block["dollars_per_recipient"] == pytest.approx(expected)
    assert dollars["weeks_during"] == 6 and dollars["n_recipient_campaigns"] == 250


def test_informative_drops_only_all_zero_cells():
    frame = pd.DataFrame(
        {
            "hw": [1, 1, 2, 2, 3, 3],
            "hg": [1, 2, 1, 2, 3, 4],
            "spend": [0.0, 0.0, 1.0, 2.0, 0.0, 5.0],
        }
    )
    kept = triple_diff.informative(frame)
    # household-week 1 never bought anything; cell hg=3 is always zero
    assert kept.hw.tolist() == [2, 2, 3] and kept.hg.tolist() == [1, 2, 4]


def test_event_study_is_flat_before_and_positive_during(panel):
    out = triple_diff.event_study(panel)
    assert out["pretrend_test"]["p_value"] > 0.01
    during = out["coef"][out["rel_week"].index(0)]
    assert np.expm1(during) == pytest.approx(TRUE_LIFT, abs=0.1)
