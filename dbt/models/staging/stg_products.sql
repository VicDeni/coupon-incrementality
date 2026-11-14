-- Product catalog. A few thousand basket lines (about $170 of sales in total) reference
-- product ids that are missing from the catalog; they are kept as 'UNKNOWN' products so
-- that no transaction is silently dropped by a join.
with catalog as (
    select
        cast(product_id as integer)      as product_id,
        cast(manufacturer_id as integer) as manufacturer_id,
        coalesce(department, 'UNKNOWN')  as department,
        brand,
        coalesce(product_category, 'UNKNOWN') as product_category,
        product_type,
        package_size
    from {{ source('raw', 'products') }}
),

uncatalogued as (
    select distinct cast(product_id as integer) as product_id
    from {{ source('raw', 'transactions') }}
    where cast(product_id as integer) not in (select product_id from catalog)
)

select * from catalog
union all
select
    product_id,
    null        as manufacturer_id,
    'UNKNOWN'   as department,
    null        as brand,
    'UNKNOWN'   as product_category,
    null        as product_type,
    null        as package_size
from uncatalogued
