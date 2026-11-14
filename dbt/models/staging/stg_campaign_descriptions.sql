select
    cast(campaign_id as integer)    as campaign_id,
    replace(campaign_type, 'Type ', '') as campaign_type,
    cast(start_date as date)        as start_date,
    cast(end_date as date)          as end_date,
    {{ week_of('start_date') }}     as start_week,
    {{ week_of('end_date') }}       as end_week
from {{ source('raw', 'campaign_descriptions') }}
