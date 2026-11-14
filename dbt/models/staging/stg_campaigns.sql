select
    cast(campaign_id as integer)  as campaign_id,
    cast(household_id as integer) as household_id
from {{ source('raw', 'campaigns') }}
