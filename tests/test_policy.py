"""Targeting policies respect the budget and the evaluation arithmetic adds up."""

import numpy as np
import pandas as pd
import pytest

from coupon_incrementality import policy


@pytest.fixture
def scores():
    rng = np.random.default_rng(0)
    n = 600
    frame = pd.DataFrame(
        {
            "household_id": rng.integers(0, 250, size=n),
            "campaign_id": rng.choice([8, 13, 18], size=n),
            "d": rng.binomial(1, 0.55, size=n),
            "n_weeks": 7,
            "spend_mean_12w": rng.gamma(4, 12, size=n),
            "predicted_effect": rng.normal(2, 2, size=n),
            "predicted_coupon_cost": rng.uniform(0, 1, size=n),
            "cost_score": rng.uniform(0, 1, size=n),
        }
    )
    frame["effect_score"] = frame.predicted_effect + rng.normal(0, 10, size=n)
    return frame


def test_budgeted_policies_send_exactly_the_actual_number_of_mailers(scores):
    weights = policy.policy_weights(scores, margin=0.275, mail_cost=0.5)
    budget = scores.d.sum()
    for name in ("Retailer's actual targeting", "Top spenders", "Random", "Causal forest"):
        assert weights[name].sum() == pytest.approx(budget), name
    assert weights["Mail everyone"].sum() == len(scores)


def test_budget_holds_within_each_campaign(scores):
    weights = policy.policy_weights(scores, margin=0.275, mail_cost=0.5)
    per_campaign = scores.groupby("campaign_id").d.sum()
    for name in ("Top spenders", "Causal forest"):
        sent = pd.Series(weights[name]).groupby(scores.campaign_id.to_numpy()).sum()
        assert (sent == per_campaign).all(), name


def test_top_spenders_picks_the_biggest_spenders(scores):
    weights = policy.policy_weights(scores, margin=0.275, mail_cost=0.5)["Top spenders"]
    for _, part in scores.assign(w=weights).groupby("campaign_id"):
        assert part[part.w == 1].spend_mean_12w.min() >= part[part.w == 0].spend_mean_12w.max()


def test_profit_score_formula(scores):
    profit = policy.profit_score(scores, margin=0.3, mail_cost=0.5)
    expected = 0.3 * 7 * scores.effect_score - scores.cost_score - 0.5
    assert np.allclose(profit, expected)


def test_actual_policy_has_zero_gain_over_itself(scores):
    out = policy.evaluate(scores, margin=0.275, mail_cost=0.5)
    gain = out["Retailer's actual targeting"]["gain_vs_actual"]
    assert gain["estimate"] == 0 and gain["se"] == 0


def test_gain_is_the_difference_in_profit(scores):
    out = policy.evaluate(scores, margin=0.275, mail_cost=0.5)
    actual = out["Retailer's actual targeting"]["profit"]["estimate"]
    for name, result in out.items():
        assert result["gain_vs_actual"]["estimate"] == pytest.approx(
            result["profit"]["estimate"] - actual
        ), name


def test_a_forest_with_perfect_foresight_beats_random():
    rng = np.random.default_rng(1)
    n = 2000
    effect = rng.normal(2, 3, size=n)
    scores = pd.DataFrame(
        {
            "household_id": np.arange(n), "campaign_id": 8, "d": rng.binomial(1, 0.5, size=n),
            "n_weeks": 7, "spend_mean_12w": rng.gamma(4, 12, size=n),
            "predicted_effect": effect, "predicted_coupon_cost": 0.5, "cost_score": 0.5,
            "effect_score": effect + rng.normal(0, 5, size=n),
        }
    )  # fmt: skip
    out = policy.evaluate(scores, margin=0.275, mail_cost=0.5)
    assert out["Causal forest"]["profit"]["estimate"] > out["Random"]["profit"]["estimate"]


def test_budget_curve_ends_at_mail_everyone(scores):
    curve = policy.budget_curve(scores, margin=0.275, mail_cost=0.5)
    everyone = policy.profit_score(scores, 0.275, 0.5).sum()
    for name in ("Causal forest", "Top spenders", "Random"):
        assert curve[name][-1] == pytest.approx(everyone)
