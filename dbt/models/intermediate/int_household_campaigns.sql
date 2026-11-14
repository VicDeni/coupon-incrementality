-- Household x campaign: who received what, and when.
select
    c.household_id,
    c.campaign_id,
    d.campaign_type,
    d.start_date,
    d.end_date,
    d.start_week,
    d.end_week
from {{ ref('stg_campaigns') }} as c
inner join {{ ref('stg_campaign_descriptions') }} as d on c.campaign_id = d.campaign_id
-- 2 recipients never appear in the transaction log; they cannot be analysed.
inner join {{ ref('int_households') }} as h on c.household_id = h.household_id
