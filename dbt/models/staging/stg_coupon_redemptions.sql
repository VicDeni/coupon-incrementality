select
    cast(household_id as integer)  as household_id,
    cast(coupon_upc as bigint)     as coupon_upc,
    cast(campaign_id as integer)   as campaign_id,
    cast(redemption_date as date)  as redemption_date,
    {{ week_of('redemption_date') }} as week
from {{ source('raw', 'coupon_redemptions') }}
