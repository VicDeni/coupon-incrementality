select
    cast(household_id as integer) as household_id,
    cast(age as varchar)            as age,
    cast(income as varchar)         as income,
    cast(home_ownership as varchar) as home_ownership,
    cast(marital_status as varchar) as marital_status,
    cast(household_size as varchar) as household_size,
    cast(household_comp as varchar) as household_comp,
    cast(kids_count as varchar)     as kids_count
from {{ source('raw', 'demographics') }}
