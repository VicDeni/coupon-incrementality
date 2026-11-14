"""Phase 6 - a targeting policy under a budget.

Profit from mailing a campaign to a household
    = gross margin on the spend the coupon adds
    - coupon dollars the household redeems
    - the cost of printing and mailing.

Five policies choose who gets the mailer. Each is scored with doubly robust off-policy
evaluation: a household's contribution is its doubly robust profit score, which is an
unbiased estimate of the profit effect of mailing it whether or not it was mailed in
reality. Scores and rankings both come from models that never saw the household, so the
winning policy is not graded on the data it learned from.

"Same budget" means the same number of mailers per campaign as the retailer really sent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    DISTRIBUTION_COST,
    MARGIN_BASE,
    MARGIN_HIGH,
    MARGIN_LOW,
    ROOT,
    SEED,
    save_result,
)
from .heterogeneity import SCORES_PATH

APP_DATA = ROOT / "app" / "data" / "household_scores.csv"


def profit_score(scores: pd.DataFrame, margin: float, mail_cost: float) -> pd.Series:
    """Doubly robust estimate of the profit from mailing each household."""
    return margin * scores.n_weeks * scores.effect_score - scores.cost_score - mail_cost


def predicted_profit(scores: pd.DataFrame, margin: float, mail_cost: float) -> pd.Series:
    """What the model expects mailing each household to earn (used only to rank)."""
    return (
        margin * scores.n_weeks * scores.predicted_effect - scores.predicted_coupon_cost - mail_cost
    )


def top_n_per_campaign(scores: pd.DataFrame, ranking: pd.Series, n: pd.Series) -> np.ndarray:
    """1.0 for the n highest-ranked households in each campaign, else 0.0."""
    position = ranking.groupby(scores.campaign_id).rank(ascending=False, method="first")
    return (position <= scores.campaign_id.map(n)).to_numpy(dtype=float)


def policy_weights(scores: pd.DataFrame, margin: float, mail_cost: float) -> dict[str, np.ndarray]:
    """Probability that each policy mails each household."""
    budget = scores.groupby("campaign_id").d.sum()
    campaign_size = scores.groupby("campaign_id").d.size()
    return {
        "Mail everyone": np.ones(len(scores)),
        "Retailer's actual targeting": scores.d.to_numpy(dtype=float),
        "Top spenders": top_n_per_campaign(scores, scores.spend_mean_12w, budget),
        "Random": scores.campaign_id.map(budget / campaign_size).to_numpy(),
        "Causal forest": top_n_per_campaign(
            scores, predicted_profit(scores, margin, mail_cost), budget
        ),
    }


def total(values: np.ndarray, clusters) -> dict:
    """Sum of per-row values with a household-clustered standard error."""
    sums = pd.Series(values).groupby(np.asarray(clusters)).sum()
    n = len(sums)
    se = float(np.sqrt(n / (n - 1) * ((sums - sums.mean()) ** 2).sum()))
    estimate = float(values.sum())
    return {
        "estimate": estimate,
        "se": se,
        "ci_low": estimate - 1.96 * se,
        "ci_high": estimate + 1.96 * se,
    }


def evaluate(scores: pd.DataFrame, margin: float, mail_cost: float) -> dict:
    profit = profit_score(scores, margin, mail_cost).to_numpy()
    gross = (margin * scores.n_weeks * scores.effect_score).to_numpy()
    cost = (scores.cost_score + mail_cost).to_numpy()
    weights = policy_weights(scores, margin, mail_cost)
    actual = weights["Retailer's actual targeting"]
    out = {}
    for name, w in weights.items():
        out[name] = {
            "mailers": float(w.sum()),
            "profit": total(w * profit, scores.household_id),
            "profit_per_1000_mailers": float(1000 * (w * profit).sum() / w.sum()),
            "gross_profit_per_cost_dollar": float((w * gross).sum() / (w * cost).sum()),
            "gain_vs_actual": total((w - actual) * profit, scores.household_id),
        }
    return out


def budget_curve(scores: pd.DataFrame, margin: float, mail_cost: float) -> dict:
    """Profit as the mailing list grows, under three ways of ordering households."""
    profit = profit_score(scores, margin, mail_cost).to_numpy()
    rng = np.random.default_rng(SEED)
    orderings = {
        "Causal forest": predicted_profit(scores, margin, mail_cost).to_numpy(),
        "Top spenders": scores.spend_mean_12w.to_numpy(),
        "Random": rng.random(len(scores)),
    }
    shares = np.linspace(0.05, 1.0, 20)
    curve = {"share_mailed": shares.tolist()}
    for name, ranking in orderings.items():
        cumulative = np.cumsum(profit[np.argsort(-ranking)])
        curve[name] = [float(cumulative[int(round(s * len(scores))) - 1]) for s in shares]
    return curve


def run() -> dict:
    scores = pd.read_parquet(SCORES_PATH)
    result = {
        "unit": "dollars of profit vs. mailing nobody, summed over the evaluated households",
        "n_rows": int(len(scores)),
        "mailers_at_actual_budget": int(scores.d.sum()),
        "assumptions": {"margin": MARGIN_BASE, "mail_cost": DISTRIBUTION_COST},
        "base": evaluate(scores, MARGIN_BASE, DISTRIBUTION_COST),
        "margin_low": evaluate(scores, MARGIN_LOW, DISTRIBUTION_COST),
        "margin_high": evaluate(scores, MARGIN_HIGH, DISTRIBUTION_COST),
        "budget_curve": budget_curve(scores, MARGIN_BASE, DISTRIBUTION_COST),
    }
    save_result("policy", result)

    # Small, anonymous extract for the Streamlit simulator (committed to the repo so the
    # app runs without rebuilding the pipeline).
    APP_DATA.parent.mkdir(parents=True, exist_ok=True)
    scores.drop(columns=["fold", "propensity"]).round(4).to_csv(APP_DATA, index=False)
    return result


if __name__ == "__main__":
    out = run()
    table = pd.DataFrame(
        {
            name: {
                "mailers": p["mailers"],
                "profit": p["profit"]["estimate"],
                "se": p["profit"]["se"],
                "per_cost_$": p["gross_profit_per_cost_dollar"],
                "gain_vs_actual": p["gain_vs_actual"]["estimate"],
                "gain_se": p["gain_vs_actual"]["se"],
            }
            for name, p in out["base"].items()
        }
    ).T
    print(table.round(2))
