"""Phase 4 (halo) and a second opinion on Phase 3b - double ML for Type B/C campaigns.

The triple difference cannot see a store-wide halo: its household x week fixed effect
absorbs anything that moves the whole basket. And a plain difference-in-differences on
rest-of-store spend fails, because recipients' spending was already drifting down
relative to everyone else before the campaigns began (see `triple_diff.halo_did`).

So halo is measured the same way as the Type A effect: compare recipients with
non-recipients who had the same purchase history. One propensity model is shared across
three outcomes - spend on covered products, on sibling products (same category, not
covered) and on the rest of the store - which also gives an independent check on the
triple difference's covered-product estimate, under different assumptions.

Unit: household x campaign, stacked over the Type B/C campaigns with 12 weeks of history.
"""

from __future__ import annotations

import pandas as pd

from . import aipw
from .config import FIRST_FULL_WEEK, HISTORY_WEEKS, PROPENSITY_BOUNDS, save_result
from .db import query
from .features import build_frame, feature_columns

GROUPS = ("covered", "sibling", "rest")
N_REP = 2


def eligible_campaigns() -> tuple[int, ...]:
    ids = query(f"""
        select campaign_id
        from stg_campaign_descriptions
        where campaign_id in (select distinct campaign_id from fct_household_group_week)
          and start_week - {HISTORY_WEEKS} >= {FIRST_FULL_WEEK}
        order by 1
    """).campaign_id
    return tuple(int(c) for c in ids)


def group_spend(campaign_ids: tuple[int, ...]) -> pd.DataFrame:
    """Average weekly spend by product group before, during and after each campaign."""
    long = query(f"""
        select
            campaign_id, household_id, period,
            case when product_group in ('other', 'placebo') then 'rest' else product_group end
                as grp,
            sum(spend) / count(distinct week) as weekly_spend,
            count(distinct week) as n_weeks
        from fct_household_group_week
        where rel_week >= -8 and campaign_id in {campaign_ids}
        group by all
    """)
    wide = long.pivot(
        index=["household_id", "campaign_id"], columns=["grp", "period"], values="weekly_spend"
    )
    wide.columns = [f"{group}_{period}" for group, period in wide.columns]
    weeks = long.pivot_table(
        index=["household_id", "campaign_id"], columns="period", values="n_weeks", aggfunc="max"
    ).fillna(0)
    wide["weeks_during"], wide["weeks_post"] = weeks["during"], weeks.get("post", 0)
    return wide.reset_index()


def build() -> pd.DataFrame:
    campaigns = eligible_campaigns()
    frame = build_frame(campaigns).merge(group_spend(campaigns), on=["household_id", "campaign_id"])
    for group in GROUPS:
        # outcomes (prefixed out_, so they are never used as features)
        frame[f"out_{group}"] = frame[f"{group}_during"] - frame[f"{group}_pre"]
        frame[f"out_{group}_post"] = frame[f"{group}_post"] - frame[f"{group}_pre"]
    renames = {f"{g}_{p}": f"out_level_{g}_{p}" for g in GROUPS for p in ("during", "post")}
    renames |= {"weeks_during": "out_weeks_during", "weeks_post": "out_weeks_post"}
    return frame.rename(columns=renames)


def run() -> dict:
    frame = build()
    features = feature_columns(frame)
    fitted = aipw.cross_fit(frame, features, [])
    # Recipients who were all but certain to be targeted have no comparable non-recipients.
    frame = frame[fitted.m <= PROPENSITY_BOUNDS[1]].reset_index(drop=True)
    recipients = frame[frame.d == 1]

    during = aipw.att_repeated(frame, features, [f"out_{g}" for g in GROUPS], n_rep=N_REP)
    has_post = frame[frame.out_weeks_post >= 4].reset_index(drop=True)
    post = aipw.att_repeated(has_post, features, [f"out_{g}_post" for g in GROUPS], n_rep=N_REP)

    def per_recipient(estimate: dict, weeks: float) -> dict:
        return {k: v * weeks for k, v in estimate.items()} | {"weekly": estimate}

    weeks_during = float(recipients.out_weeks_during.mean())
    weeks_post = float(has_post[has_post.d == 1].out_weeks_post.mean())
    result = {
        "unit": "dollars per recipient household per campaign (and 'weekly': dollars per week)",
        "campaigns": [int(c) for c in sorted(frame.campaign_id.unique())],
        "n_rows": int(len(frame)),
        "n_recipient_rows": int(len(recipients)),
        "n_features": len(features),
        "weeks_during": weeks_during,
        "weeks_post": weeks_post,
        "coupon_cost_per_recipient": float(recipients.coupon_cost.mean()),
        "during": {g: per_recipient(during[f"out_{g}"], weeks_during) for g in GROUPS},
        "post": {g: per_recipient(post[f"out_{g}_post"], weeks_post) for g in GROUPS},
        "recipient_weekly_spend": {
            g: float(recipients[f"out_level_{g}_during"].mean()) for g in GROUPS
        },
    }
    save_result("dml_bc", result)
    return result


if __name__ == "__main__":
    out = run()
    for key in ("n_rows", "n_recipient_rows", "weeks_during", "recipient_weekly_spend"):
        print(key, out[key])
    for period in ("during", "post"):
        print(
            period,
            {g: {k: round(v, 3) for k, v in out[period][g]["weekly"].items()} for g in GROUPS},
        )
