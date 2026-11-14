-- Product x store x week. A location code of 0 means "not on display" / "not in the mailer".
select
    cast(product_id as integer) as product_id,
    cast(store_id as integer)   as store_id,
    cast(week as integer)       as week,
    cast(display_location as varchar) <> '0' as on_display,
    cast(mailer_location as varchar) <> '0'  as in_mailer
from {{ source('raw', 'promotions') }}
