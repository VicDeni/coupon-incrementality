"""Household x campaign modelling frame.

One row per household per campaign, with:
  * treatment  d       : the household received the campaign
  * outcomes   y       : average weekly in-store spend while the campaign ran
               y_post  : average weekly spend in the weeks after it ended
               coupon_cost : dollars of this campaign's coupons the household redeemed
  * controls           : ~250 features describing the household's purchases in the
                         12 weeks before the campaign started

Every control is measured strictly before the campaign's first week, so none of them
can be affected by the treatment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FIRST_FULL_WEEK, HISTORY_WEEKS, LAST_WEEK, POST_WEEKS, TYPE_A_CAMPAIGNS
from .db import query

ID_COLS = ["household_id", "campaign_id"]
OUTCOME_COLS = [
    "d", "y", "y_post", "y_change", "y_post_change", "coupon_cost", "n_weeks", "n_post_weeks",
]  # fmt: skip
N_TOP_CATEGORIES = 100


def _campaigns(campaign_ids: tuple[int, ...], shift_weeks: int = 0) -> str:
    """SQL for the campaigns to model. `shift_weeks` moves them earlier (placebo test)."""
    ids = ", ".join(str(c) for c in campaign_ids)
    return f"""
        select campaign_id, start_week - {shift_weeks} as start_week,
               end_week - {shift_weeks} as end_week
        from stg_campaign_descriptions
        where campaign_id in ({ids})
    """


def _outcomes(campaigns: str) -> pd.DataFrame:
    return query(f"""
        with c as ({campaigns})
        select
            h.household_id,
            c.campaign_id,
            cast(hc.household_id is not null as integer) as d,
            avg(w.spend) filter (where w.week between c.start_week and c.end_week) as y,
            avg(w.spend) filter (
                where w.week between c.end_week + 1 and c.end_week + {POST_WEEKS}
            ) as y_post,
            coalesce(max(hc.coupon_value), 0) as coupon_cost,
            count(*) filter (where w.week between c.start_week and c.end_week) as n_weeks,
            count(*) filter (
                where w.week between c.end_week + 1 and c.end_week + {POST_WEEKS}
            ) as n_post_weeks
        from int_households as h
        cross join c
        inner join fct_household_week as w
            on w.household_id = h.household_id and w.week <= {LAST_WEEK}
        left join fct_household_campaign as hc
            on hc.household_id = h.household_id and hc.campaign_id = c.campaign_id
        group by all
    """)


def _weekly_history(campaigns: str) -> pd.DataFrame:
    """Spend and trips in each of the 12 weeks before the start, plus summary stats."""
    long = query(f"""
        with c as ({campaigns})
        select
            w.household_id, c.campaign_id, c.start_week - w.week as lag,
            w.spend, w.trips, w.units, w.fuel_spend, w.retail_disc, w.coupon_disc,
            w.coupon_match_disc, w.discounted_spend
        from fct_household_week as w
        inner join c on w.week between c.start_week - {HISTORY_WEEKS} and c.start_week - 1
    """)
    wide = long.pivot(index=ID_COLS, columns="lag", values=["spend", "trips"])
    wide.columns = [f"{name}_lag{lag:02d}" for name, lag in wide.columns]

    grouped = long.groupby(ID_COLS)
    recent = long[long.lag <= 4].groupby(ID_COLS)
    stats = pd.DataFrame(
        {
            "spend_mean_12w": grouped.spend.mean(),
            "spend_std_12w": grouped.spend.std(),
            "spend_max_12w": grouped.spend.max(),
            "spend_mean_4w": recent.spend.mean(),
            "trips_mean_12w": grouped.trips.mean(),
            "trips_mean_4w": recent.trips.mean(),
            "units_mean_12w": grouped.units.mean(),
            "fuel_mean_12w": grouped.fuel_spend.mean(),
            "share_weeks_shopped": grouped.trips.apply(lambda s: (s > 0).mean()),
            "retail_disc_12w": grouped.retail_disc.sum(),
            "coupon_disc_12w": grouped.coupon_disc.sum(),
            "coupon_match_disc_12w": grouped.coupon_match_disc.sum(),
            "discounted_spend_12w": grouped.discounted_spend.sum(),
        }
    )
    total = grouped.spend.sum()
    trips = grouped.trips.sum()
    stats["basket_size"] = total / trips.clip(lower=1)
    # share of the shelf price the household avoided through loyalty-card discounts
    stats["discount_depth"] = stats.retail_disc_12w / (total + stats.retail_disc_12w).clip(lower=1)
    # share of spend that was bought on any kind of deal
    stats["deal_share"] = (stats.discounted_spend_12w / total.clip(lower=1)).clip(upper=1)
    # linear trend in weekly spend ($ per week); lag runs backwards in time
    lags = np.arange(HISTORY_WEEKS, 0, -1)
    spend_matrix = wide[[f"spend_lag{lag:02d}" for lag in lags]].to_numpy()
    time = np.arange(HISTORY_WEEKS) - (HISTORY_WEEKS - 1) / 2
    stats["spend_trend"] = (spend_matrix @ time) / (time @ time)
    return wide.join(stats)


def _long_history(campaigns: str) -> pd.DataFrame:
    return query(f"""
        with c as ({campaigns})
        select
            w.household_id, c.campaign_id,
            avg(w.spend) as spend_mean_all, avg(w.trips) as trips_mean_all,
            sum(w.coupon_disc) as coupon_disc_all, count(*) as history_weeks
        from fct_household_week as w
        inner join c on w.week between {FIRST_FULL_WEEK} and c.start_week - 1
        group by all
    """).set_index(ID_COLS)


def _basket_mix(campaigns: str) -> pd.DataFrame:
    """Where the money went: shares by department and by the 100 biggest categories."""
    long = query(f"""
        with c as ({campaigns})
        select t.household_id, c.campaign_id, p.department, p.product_category,
               p.brand = 'Private' as is_private, sum(t.sales_value) as spend,
               count(distinct t.product_id) as n_products
        from stg_transactions as t
        inner join stg_products as p on t.product_id = p.product_id
        inner join c on t.week between c.start_week - {HISTORY_WEEKS} and c.start_week - 1
        where p.department <> 'FUEL'
        group by all
    """)
    total = long.groupby(ID_COLS).spend.sum()

    def shares(column: str, keep: list[str], prefix: str) -> pd.DataFrame:
        part = long[long[column].isin(keep)]
        wide = part.pivot_table(index=ID_COLS, columns=column, values="spend", aggfunc="sum")
        wide = wide.reindex(columns=keep).div(total, axis=0)
        wide.columns = [f"{prefix}_{_slug(c)}" for c in wide.columns]
        return wide

    departments = sorted(long.department.unique())
    top_categories = (
        long[long.product_category != "UNKNOWN"]
        .groupby("product_category")
        .spend.sum()
        .nlargest(N_TOP_CATEGORIES)
        .index.tolist()
    )
    extra = pd.DataFrame(
        {
            "private_label_share": long[long.is_private].groupby(ID_COLS).spend.sum() / total,
            "n_categories": long.groupby(ID_COLS).product_category.nunique(),
            "n_departments": long.groupby(ID_COLS).department.nunique(),
            "n_products": long.groupby(ID_COLS).n_products.sum(),
        }
    )
    return pd.concat(
        [
            shares("department", departments, "dept"),
            shares("product_category", top_categories, "cat"),
            extra,
        ],
        axis=1,
    )


def _program_history(campaigns: str) -> pd.DataFrame:
    """Earlier campaigns, earlier redemptions, and campaigns overlapping this one."""
    return query(f"""
        with c as ({campaigns})
        select
            h.household_id, c.campaign_id,
            count(*) filter (where o.campaign_type = 'A' and o.start_week < c.start_week)
                as prior_a,
            count(*) filter (where o.campaign_type = 'B' and o.start_week < c.start_week)
                as prior_b,
            count(*) filter (where o.campaign_type = 'C' and o.start_week < c.start_week)
                as prior_c,
            coalesce(sum(o.n_redemptions) filter (where o.start_week < c.start_week), 0)
                as prior_redemptions,
            count(*) filter (
                where o.start_week <= c.end_week and o.end_week >= c.start_week
            ) as overlapping_campaigns
        from int_households as h
        cross join c
        left join fct_household_campaign as o
            on o.household_id = h.household_id and o.campaign_id <> c.campaign_id
        group by all
    """).set_index(ID_COLS)


def _demographics() -> pd.DataFrame:
    demo = query("""
        select household_id, has_demographics, age, income, home_ownership, marital_status,
               household_size, kids_count
        from dim_household
    """).set_index("household_id")
    dummies = pd.get_dummies(demo.drop(columns="has_demographics"), prefix_sep="_", dtype=float)
    dummies.columns = [f"demo_{_slug(c)}" for c in dummies.columns]
    dummies["demo_available"] = demo.has_demographics.astype(float)
    return dummies


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(text)).strip("_").lower()


def build_frame(
    campaign_ids: tuple[int, ...] = TYPE_A_CAMPAIGNS, shift_weeks: int = 0
) -> pd.DataFrame:
    """Return the modelling frame; feature columns are everything not in ID/OUTCOME cols."""
    campaigns = _campaigns(campaign_ids, shift_weeks)
    frame = _outcomes(campaigns).set_index(ID_COLS)
    for build in (_weekly_history, _long_history, _basket_mix, _program_history):
        frame = frame.join(build(campaigns))
    frame = frame.reset_index().merge(_demographics(), on="household_id", how="left")
    for campaign in campaign_ids:
        frame[f"campaign_{campaign}"] = (frame.campaign_id == campaign).astype(float)
    features = feature_columns(frame)
    frame[features] = frame[features].fillna(0.0).astype(float)
    # Outcomes as changes from the household's own pre-campaign average. The models then
    # predict a change rather than a level, which they do far more reliably for the
    # heaviest shoppers (who have few comparable non-recipients).
    frame["y_change"] = frame.y - frame.spend_mean_12w
    frame["y_post_change"] = frame.y_post - frame.spend_mean_12w
    return frame.sort_values(ID_COLS).reset_index(drop=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Pre-campaign controls: every column that is not an id, an outcome or a model output."""
    excluded = set(ID_COLS + OUTCOME_COLS) | {"propensity", "in_support"}
    return [c for c in frame.columns if c not in excluded and not c.startswith("out_")]
