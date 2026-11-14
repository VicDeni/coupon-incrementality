-- Household x campaign received, with what the household redeemed.
with redeemed as (
    select
        household_id,
        campaign_id,
        count(*) as n_redemptions,
        sum(coupon_value) as coupon_value
    from {{ ref('int_redemption_values') }}
    group by 1, 2
)

select
    hc.household_id,
    hc.campaign_id,
    hc.campaign_type,
    hc.start_date,
    hc.end_date,
    hc.start_week,
    hc.end_week,
    coalesce(r.n_redemptions, 0) as n_redemptions,
    coalesce(r.coupon_value, 0) as coupon_value
from {{ ref('int_household_campaigns') }} as hc
left join redeemed as r on hc.household_id = r.household_id and hc.campaign_id = r.campaign_id
