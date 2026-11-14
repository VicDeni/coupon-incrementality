"""Phase 2 - the naive baseline: what a marketing dashboard would report.

Three flavours, none of which adjusts for who was chosen to receive coupons:

  1. recipients vs. non-recipients   average weekly spend while the campaign ran
  2. redeemers vs. non-redeemers     same, split by who actually used a coupon
  3. redemption attribution          every dollar of a redeemed item counted as incremental

The first is the starting bar of the waterfall, because it answers the same question as
the causal estimates (what did *sending* the campaign do?).
"""

from __future__ import annotations

from .config import LAST_WEEK, MARGIN_BASE, TYPE_A_CAMPAIGNS, save_result
from .db import query


def _weighted_mean(values, weights) -> float:
    """Weighted mean that skips campaigns where the value is undefined (no redeemers)."""
    known = values.notna()
    return float((values[known] * weights[known]).sum() / weights[known].sum())


def campaign_comparison() -> dict:
    """Pooled over analysed campaigns, weighted by the number of recipients."""
    frame = query(f"""
        with analysed as (
            select campaign_id, campaign_type, start_week, end_week
            from stg_campaign_descriptions
            where campaign_id in {TYPE_A_CAMPAIGNS}
               or campaign_id in (select distinct campaign_id from fct_household_group_week)
        ),
        spend as (
            select
                c.campaign_id, c.campaign_type, w.household_id, avg(w.spend) as weekly_spend,
                max(cast(hc.household_id is not null as integer)) as received,
                max(cast(coalesce(hc.n_redemptions, 0) > 0 as integer)) as redeemed
            from analysed as c
            inner join fct_household_week as w
                on w.week between c.start_week and least(c.end_week, {LAST_WEEK})
            left join fct_household_campaign as hc
                on hc.campaign_id = c.campaign_id and hc.household_id = w.household_id
            group by all
        )
        select
            campaign_id, campaign_type,
            sum(received) as n_recipients,
            sum(redeemed) as n_redeemers,
            avg(weekly_spend) filter (where received = 1) as recipients,
            avg(weekly_spend) filter (where received = 0) as non_recipients,
            avg(weekly_spend) filter (where redeemed = 1) as redeemers,
            avg(weekly_spend) filter (where redeemed = 0) as non_redeemers
        from spend
        group by all
    """)
    out = {}
    for label, part in [
        ("A", frame[frame.campaign_type == "A"]),
        ("BC", frame[frame.campaign_type != "A"]),
    ]:
        w, r = part.n_recipients, part.n_redeemers
        recipients = _weighted_mean(part.recipients, w)
        non_recipients = _weighted_mean(part.non_recipients, w)
        redeemers = _weighted_mean(part.redeemers, r)
        non_redeemers = _weighted_mean(part.non_redeemers, r)
        out[label] = {
            "n_campaigns": int(len(part)),
            "n_recipient_campaigns": int(w.sum()),
            "n_redeemer_campaigns": int(r.sum()),
            "recipients_weekly_spend": recipients,
            "non_recipients_weekly_spend": non_recipients,
            "recipient_lift_pct": 100 * (recipients / non_recipients - 1),
            "redeemers_weekly_spend": redeemers,
            "non_redeemers_weekly_spend": non_redeemers,
            "redeemer_lift_pct": 100 * (redeemers / non_redeemers - 1),
        }
    return out


def redemption_attribution() -> dict:
    """Dashboard logic: a redeemed coupon 'caused' the sale of the item it was used on."""
    row = query("""
        select count(*) as n, sum(coupon_value) as coupon_dollars, sum(matched_sales) as sales
        from int_redemption_values
        where is_matched
    """).iloc[0]
    return {
        "matched_redemptions": int(row.n),
        "coupon_dollars": float(row.coupon_dollars),
        "attributed_sales": float(row.sales),
        "sales_per_coupon_dollar": float(row.sales / row.coupon_dollars),
        "gross_profit_per_coupon_dollar": float(MARGIN_BASE * row.sales / row.coupon_dollars),
    }


def run() -> dict:
    result = {
        "unit": "dollars of in-store spend per household per campaign week",
        "comparison": campaign_comparison(),
        "redemption_attribution": redemption_attribution(),
    }
    save_result("naive", result)
    return result


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=1))
