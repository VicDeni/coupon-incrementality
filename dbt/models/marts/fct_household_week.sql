-- Household x week panel (balanced: a week with no shopping is a row of zeros).
-- `spend` excludes fuel.
with weeks as (
    select unnest(range({{ var('first_week') }}, {{ var('last_week') }} + 1)) as week
),

grid as (
    select h.household_id, w.week
    from {{ ref('int_households') }} as h
    cross join weeks as w
),

purchases as (
    select
        t.household_id,
        t.week,
        sum(case when p.department <> 'FUEL' then t.sales_value else 0 end) as spend,
        sum(case when p.department = 'FUEL' then t.sales_value else 0 end)  as fuel_spend,
        count(distinct t.basket_id)                                         as trips,
        sum(case when p.department <> 'FUEL' then t.quantity else 0 end)    as units,
        sum(t.retail_disc)                                                  as retail_disc,
        sum(t.coupon_disc)                                                  as coupon_disc,
        sum(t.coupon_match_disc)                                            as coupon_match_disc,
        sum(case when p.department <> 'FUEL' and (t.retail_disc > 0 or t.coupon_disc > 0)
                 then t.sales_value else 0 end)                              as discounted_spend
    from {{ ref('stg_transactions') }} as t
    inner join {{ ref('stg_products') }} as p on t.product_id = p.product_id
    group by 1, 2
),

active as (
    select
        g.household_id,
        g.week,
        count(*) filter (where hc.campaign_type = 'A') as active_a,
        count(*) filter (where hc.campaign_type = 'B') as active_b,
        count(*) filter (where hc.campaign_type = 'C') as active_c
    from grid as g
    inner join {{ ref('int_household_campaigns') }} as hc
        on g.household_id = hc.household_id and g.week between hc.start_week and hc.end_week
    group by 1, 2
),

redeemed as (
    select household_id, week, count(*) as n_redemptions, sum(coupon_value) as campaign_coupon_value
    from {{ ref('int_redemption_values') }}
    group by 1, 2
)

select
    g.household_id,
    g.week,
    date '{{ var("week_one_start") }}' + cast(7 * (g.week - 1) as integer) as week_start,
    coalesce(p.spend, 0)              as spend,
    coalesce(p.fuel_spend, 0)         as fuel_spend,
    coalesce(p.trips, 0)              as trips,
    coalesce(p.units, 0)              as units,
    coalesce(p.retail_disc, 0)        as retail_disc,
    coalesce(p.coupon_disc, 0)        as coupon_disc,
    coalesce(p.coupon_match_disc, 0)  as coupon_match_disc,
    coalesce(p.discounted_spend, 0)   as discounted_spend,
    coalesce(a.active_a, 0)           as active_a,
    coalesce(a.active_b, 0)           as active_b,
    coalesce(a.active_c, 0)           as active_c,
    coalesce(r.n_redemptions, 0)      as n_redemptions,
    coalesce(r.campaign_coupon_value, 0) as campaign_coupon_value
from grid as g
left join purchases as p on g.household_id = p.household_id and g.week = p.week
left join active as a on g.household_id = a.household_id and g.week = a.week
left join redeemed as r on g.household_id = r.household_id and g.week = r.week
