-- One row per household: program exposure and (where available) demographics.
with exposure as (
    select
        household_id,
        count(*)                                        as n_campaigns,
        count(*) filter (where campaign_type = 'A')     as n_campaigns_a,
        count(*) filter (where campaign_type = 'B')     as n_campaigns_b,
        count(*) filter (where campaign_type = 'C')     as n_campaigns_c,
        min(start_week)                                 as first_campaign_week
    from {{ ref('int_household_campaigns') }}
    group by 1
)

select
    h.household_id,
    h.primary_store_id,
    h.first_week,
    h.last_week,
    coalesce(e.n_campaigns, 0)   as n_campaigns,
    coalesce(e.n_campaigns_a, 0) as n_campaigns_a,
    coalesce(e.n_campaigns_b, 0) as n_campaigns_b,
    coalesce(e.n_campaigns_c, 0) as n_campaigns_c,
    -- week of the household's first campaign; null if it never received one.
    -- Values below 2 mean the first campaign started before the data does.
    e.first_campaign_week,
    d.household_id is not null   as has_demographics,
    d.age,
    d.income,
    d.home_ownership,
    d.marital_status,
    d.household_size,
    d.household_comp,
    d.kids_count
from {{ ref('int_households') }} as h
left join exposure as e on h.household_id = e.household_id
left join {{ ref('stg_demographics') }} as d on h.household_id = d.household_id
