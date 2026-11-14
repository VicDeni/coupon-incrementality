-- Dollar value of each campaign-coupon redemption.
--
-- The redemption table says who redeemed which coupon on which day, but not what it was
-- worth. We recover the value from the basket: coupon discounts (manufacturer coupon plus
-- the retailer's match) on products that coupon covers, bought by that household within a
-- day of the redemption date. Redemptions we cannot match get the average matched value of
-- their campaign type.
with redemptions as (
    select
        r.*,
        d.campaign_type,
        count(*) over (
            partition by r.household_id, r.coupon_upc, r.campaign_id, r.redemption_date
        ) as n_same
    from {{ ref('stg_coupon_redemptions') }} as r
    inner join {{ ref('stg_campaign_descriptions') }} as d on r.campaign_id = d.campaign_id
),

matched as (
    select
        r.household_id,
        r.coupon_upc,
        r.campaign_id,
        r.redemption_date,
        sum(t.coupon_disc + t.coupon_match_disc) as matched_value,
        sum(t.sales_value) as matched_sales
    from (select distinct household_id, coupon_upc, campaign_id, redemption_date from redemptions) as r
    inner join {{ ref('stg_coupons') }} as c
        on r.coupon_upc = c.coupon_upc and r.campaign_id = c.campaign_id
    inner join {{ ref('stg_transactions') }} as t
        on r.household_id = t.household_id
        and c.product_id = t.product_id
        -- timestamps are UTC; the stores are roughly five hours behind
        and abs(date_diff('day', r.redemption_date, cast(t.transaction_ts - interval 5 hour as date))) <= 1
    where t.coupon_disc + t.coupon_match_disc > 0
    group by 1, 2, 3, 4
),

joined as (
    select
        r.household_id,
        r.coupon_upc,
        r.campaign_id,
        r.campaign_type,
        r.redemption_date,
        r.week,
        m.matched_value / r.n_same as matched_value,
        m.matched_sales / r.n_same as matched_sales
    from redemptions as r
    left join matched as m
        on r.household_id = m.household_id
        and r.coupon_upc = m.coupon_upc
        and r.campaign_id = m.campaign_id
        and r.redemption_date = m.redemption_date
)

select
    household_id,
    coupon_upc,
    campaign_id,
    campaign_type,
    redemption_date,
    week,
    matched_value is not null as is_matched,
    -- retailer revenue on the items the coupon was used on (null when unmatched)
    matched_sales,
    coalesce(matched_value, avg(matched_value) over (partition by campaign_type)) as coupon_value
from joined
