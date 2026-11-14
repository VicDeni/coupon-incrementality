"""Coupon budget simulator.

Pick a mailing budget, a gross margin and a mailing cost; see who to mail and what it is
expected to earn. Everything is computed from `data/household_scores.csv`, the
out-of-sample scores written by the pipeline's policy phase, so the app runs on its own.

    streamlit run app/streamlit_app.py
"""

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

DATA = Path(__file__).parent / "data" / "household_scores.csv"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
RULES = {
    "Causal forest": "Highest predicted incremental profit first",
    "Top spenders": "Highest pre-campaign weekly spend first",
    "Random": "A random draw of households",
}
PROFILE = {
    "spend_mean_12w": "Weekly spend ($)",
    "trips_mean_12w": "Trips per week",
    "deal_share": "Share of spend bought on deal",
    "coupon_disc_12w": "Other coupon dollars used (12 wk)",
    "prior_redemptions": "Earlier coupon redemptions",
}


def profit_score(scores: pd.DataFrame, margin: float, mail_cost: float) -> pd.Series:
    """Doubly robust estimate of the profit from mailing each household."""
    return margin * scores.n_weeks * scores.effect_score - scores.cost_score - mail_cost


def predicted_profit(scores: pd.DataFrame, margin: float, mail_cost: float) -> pd.Series:
    """What the model expects mailing each household to earn (used only to rank)."""
    return (
        margin * scores.n_weeks * scores.predicted_effect - scores.predicted_coupon_cost - mail_cost
    )


def rankings(scores: pd.DataFrame, margin: float, mail_cost: float) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "Causal forest": predicted_profit(scores, margin, mail_cost).to_numpy(),
        "Top spenders": scores.spend_mean_12w.to_numpy(),
        "Random": rng.random(len(scores)),
    }


def total_with_interval(values: np.ndarray, households: pd.Series) -> tuple[float, float, float]:
    """Sum with a 95% interval, clustered by household."""
    sums = pd.Series(values).groupby(households.to_numpy()).sum()
    n = len(sums)
    se = float(np.sqrt(n / (n - 1) * ((sums - sums.mean()) ** 2).sum()))
    estimate = float(values.sum())
    return estimate, estimate - 1.96 * se, estimate + 1.96 * se


def dollars(value: float) -> str:
    return f"{'−' if value < 0 else ''}${abs(value):,.0f}"


@st.cache_data
def load() -> pd.DataFrame:
    return pd.read_csv(DATA)


st.set_page_config(page_title="Coupon budget simulator", layout="wide")
scores = load()

st.title("Coupon budget simulator")
st.caption(
    "Three personalised (Type A) coupon campaigns, "
    f"{scores.household_id.nunique():,} households, {len(scores):,} possible mailers. "
    "Profit is incremental: what mailing adds compared with mailing nobody."
)

with st.sidebar:
    st.header("Your assumptions")
    budget = st.slider(
        "Mailers to send", 250, len(scores), int(scores.d.sum()), step=50,
        help="The retailer actually sent this many in the evaluated sample by default.",
    )  # fmt: skip
    margin = st.slider("Gross margin", 0.20, 0.35, 0.275, step=0.005, format="%.3f")
    mail_cost = st.slider("Cost to print and mail one campaign ($)", 0.0, 1.5, 0.5, step=0.05)
    rule = st.radio("Who gets a mailer?", list(RULES), captions=list(RULES.values()))

profit = profit_score(scores, margin, mail_cost).to_numpy()
order = {
    name: np.argsort(-ranking) for name, ranking in rankings(scores, margin, mail_cost).items()
}
chosen = np.zeros(len(scores), dtype=bool)
chosen[order[rule][:budget]] = True

estimate, low, high = total_with_interval(chosen * profit, scores.household_id)
gross = (margin * scores.n_weeks * scores.effect_score).to_numpy()[chosen].sum()
cost = (scores.cost_score.to_numpy()[chosen] + mail_cost).sum()

# The retailer's own list, and a same-size list under the chosen rule, for comparison.
actual = scores.d.to_numpy(dtype=bool)
same_size = np.zeros(len(scores), dtype=bool)
same_size[order[rule][: actual.sum()]] = True
gain, gain_low, gain_high = total_with_interval(
    (same_size.astype(float) - actual) * profit, scores.household_id
)

a, b, c, d = st.columns(4)
a.metric("Expected incremental profit", dollars(estimate))
a.caption(f"95% interval {dollars(low)} to {dollars(high)}")
b.metric("Gross profit per $1 of coupon spend", f"${gross / cost:,.2f}")
b.caption("Break-even is $1.00")
c.metric("Profit per mailer", f"${estimate / budget:,.2f}")
c.caption(f"{budget:,} mailers, {budget / len(scores):.0%} of households")
d.metric("vs. the retailer's actual list", dollars(gain))
d.caption(
    f"Same {actual.sum():,} mailers; 95% interval {dollars(gain_low)} to {dollars(gain_high)}"
)

if gain_low < 0 < gain_high:
    st.info(
        "The interval for the gain over the retailer's list includes zero: with about 2,500 "
        "households, the data cannot prove this rule beats the retailer's own targeting. "
        "Treat the ranking as a hypothesis to A/B test, not a settled result."
    )

left, right = st.columns([3, 2])

with left:
    st.subheader("Profit as the mailing list grows")
    shares = np.arange(250, len(scores) + 1, 50)
    curve = pd.concat(
        [
            pd.DataFrame(
                {"Mailers": shares, "Profit": np.cumsum(profit[idx])[shares - 1], "Rule": name}
            )
            for name, idx in order.items()
        ]
    )
    palette = alt.Scale(domain=list(RULES), range=[BLUE, ORANGE, AQUA])
    lines = (
        alt.Chart(curve)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("Mailers:Q", axis=alt.Axis(format=",d", grid=False)),
            y=alt.Y("Profit:Q", title="Expected incremental profit", axis=alt.Axis(format="$,.0f")),
            color=alt.Color("Rule:N", scale=palette, legend=alt.Legend(orient="top", title=None)),
            strokeWidth=alt.condition(alt.datum.Rule == rule, alt.value(3.5), alt.value(1.5)),
            tooltip=[
                "Rule",
                alt.Tooltip("Mailers:Q", format=","),
                alt.Tooltip("Profit:Q", format="$,.0f"),
            ],
        )
    )
    marker = (
        alt.Chart(pd.DataFrame({"Mailers": [budget]})).mark_rule(color=GREY).encode(x="Mailers:Q")
    )
    st.altair_chart(lines + marker, use_container_width=True)
    st.caption(
        "Curves are doubly robust estimates on households the models never saw. They are "
        "noisy: differences between rules are within the margin of error."
    )

with right:
    st.subheader("Who gets a mailer")
    profile = pd.DataFrame(
        {
            "Mailed": scores[chosen][list(PROFILE)].mean(),
            "Not mailed": scores[~chosen][list(PROFILE)].mean(),
        }
    ).rename(index=PROFILE)
    st.dataframe(profile.style.format("{:,.2f}"), use_container_width=True)
    overlap = (chosen & actual).sum() / max(chosen.sum(), 1)
    st.caption(f"{overlap:.0%} of these households were also on the retailer's actual list.")

with st.expander("How this works, and what it cannot tell you"):
    st.markdown(
        """
- **Effect scores** come from a causal forest and doubly robust (AIPW) scoring fitted to
  one year of real purchases. A household's score is always produced by models trained
  without that household.
- **Profit** = gross margin × the spend the campaign adds − coupon dollars redeemed −
  mailing cost. Margin and mailing cost are not in the data; they are your assumptions.
- **Limits.** Only households with a comparable counterpart are included (those neither
  almost certain nor almost certain not to be targeted). Estimates assume the retailer
  targeted on purchase history that is visible in the data.
"""
    )
