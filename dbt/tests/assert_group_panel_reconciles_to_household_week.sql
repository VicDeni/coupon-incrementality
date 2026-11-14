-- Within a campaign block, the four product groups add up to the household's in-store spend.
with grouped as (
    select campaign_id, household_id, week, sum(spend) as spend
    from {{ ref('fct_household_group_week') }}
    group by 1, 2, 3
)

select g.campaign_id, g.household_id, g.week, g.spend, w.spend as expected
from grouped as g
inner join {{ ref('fct_household_week') }} as w
    on g.household_id = w.household_id and g.week = w.week
where abs(g.spend - w.spend) > 0.01
