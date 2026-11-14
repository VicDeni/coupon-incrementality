-- A campaign cannot end before it starts.
select *
from {{ ref('stg_campaign_descriptions') }}
where end_date < start_date
