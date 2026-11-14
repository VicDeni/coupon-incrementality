"""Checks against the built warehouse. Skipped until `make data` has been run."""

import pytest

from coupon_incrementality.config import WAREHOUSE
from coupon_incrementality.db import query

pytestmark = [
    pytest.mark.warehouse,
    pytest.mark.skipif(not WAREHOUSE.exists(), reason="warehouse not built"),
]


def test_headline_counts_match_the_published_dataset():
    row = query("""
        select
            (select count(*) from stg_transactions) as transactions,
            (select count(*) from stg_promotions) as promotions,
            (select count(*) from int_households) as households,
            (select count(*) from stg_campaign_descriptions) as campaigns,
            (select count(*) from stg_coupon_redemptions) as redemptions
    """).iloc[0]
    assert row.transactions == 1_469_307
    assert row.promotions == 20_940_529
    assert row.households == 2_469
    assert row.campaigns == 27
    assert row.redemptions == 2_102


def test_features_only_use_weeks_before_the_campaign():
    from coupon_incrementality.features import build_frame, feature_columns

    frame = build_frame((8,))
    start = query("select start_week from stg_campaign_descriptions where campaign_id = 8").iloc[
        0, 0
    ]
    spend = (
        query(f"""
        select household_id, spend from fct_household_week where week = {start} - 1
    """)
        .set_index("household_id")
        .spend
    )
    lag1 = frame.set_index("household_id").spend_lag01
    assert (lag1 - spend.reindex(lag1.index)).abs().max() < 1e-9
    assert not {"y", "y_change", "d", "coupon_cost"} & set(feature_columns(frame))


def test_recipients_never_redeem_before_receiving():
    early = query("""
        select count(*) as n
        from int_redemption_values as r
        inner join stg_campaign_descriptions as d on r.campaign_id = d.campaign_id
        where r.redemption_date < d.start_date
    """).n[0]
    assert early == 0


def test_non_recipients_have_no_campaign_coupon_cost():
    from coupon_incrementality.features import build_frame

    frame = build_frame()
    assert frame[frame.d == 0].coupon_cost.eq(0).all()
