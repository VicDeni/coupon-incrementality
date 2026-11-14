-- For each Type B / Type C campaign, sort every product into one of four groups:
--   covered : the campaign's coupons can be used on it
--   sibling : not covered, but in the same product category as a covered product
--             (where cannibalisation would show up)
--   placebo : a fixed pseudo-random eighth of the remaining products; no coupon touches
--             them, so a "coupon effect" here would signal a broken design
--   other   : everything else in the store (the reference group)
-- Fuel is excluded everywhere: a grocery margin does not apply to it.
with eligible_campaigns as (
    select *
    from {{ ref('stg_campaign_descriptions') }}
    where campaign_type in ('B', 'C')
      -- need a pre-period inside the data ...
      and start_week - {{ var('min_pre_weeks') }} >= {{ var('first_week') }} + 1
      -- ... and at least three campaign weeks inside the data
      and start_week + 2 <= {{ var('last_week') }}
),

covered as (
    select distinct c.campaign_id, c.product_id
    from {{ ref('stg_coupons') }} as c
    inner join eligible_campaigns as e on c.campaign_id = e.campaign_id
),

covered_categories as (
    select distinct cv.campaign_id, p.product_category
    from covered as cv
    inner join {{ ref('stg_products') }} as p on cv.product_id = p.product_id
    where p.product_category <> 'UNKNOWN'
)

select
    e.campaign_id,
    p.product_id,
    case
        when cv.product_id is not null then 'covered'
        when cc.product_category is not null then 'sibling'
        when left(md5(cast(p.product_id as varchar)), 1) in ('0', '1') then 'placebo'
        else 'other'
    end as product_group
from eligible_campaigns as e
cross join {{ ref('stg_products') }} as p
left join covered as cv on e.campaign_id = cv.campaign_id and p.product_id = cv.product_id
left join covered_categories as cc
    on e.campaign_id = cc.campaign_id and p.product_category = cc.product_category
where p.department <> 'FUEL'
