-- Every household that appears in the transaction log, with its primary store
-- (the store where it made the most baskets) for matching in-store promotions.
with baskets as (
    select household_id, store_id, count(distinct basket_id) as n_baskets
    from {{ ref('stg_transactions') }}
    group by 1, 2
),

ranked as (
    select
        *,
        row_number() over (partition by household_id order by n_baskets desc, store_id) as rn
    from baskets
),

span as (
    select household_id, min(week) as first_week, max(week) as last_week
    from {{ ref('stg_transactions') }}
    group by 1
)

select
    s.household_id,
    r.store_id as primary_store_id,
    s.first_week,
    s.last_week
from span as s
inner join ranked as r on s.household_id = r.household_id and r.rn = 1
