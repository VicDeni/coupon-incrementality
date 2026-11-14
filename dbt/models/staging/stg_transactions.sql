-- One row per basket line. Discounts are stored as positive dollars taken off the shelf price.
-- `sales_value` is what the retailer receives (manufacturer coupons are reimbursed, so they
-- are already inside it).
select
    cast(household_id as integer)            as household_id,
    cast(store_id as integer)                as store_id,
    cast(basket_id as bigint)                as basket_id,
    cast(product_id as integer)              as product_id,
    cast(quantity as integer)                as quantity,
    cast(sales_value as double)              as sales_value,
    cast(retail_disc as double)              as retail_disc,
    cast(coupon_disc as double)              as coupon_disc,
    cast(coupon_match_disc as double)        as coupon_match_disc,
    cast(week as integer)                    as week,
    cast(transaction_timestamp as timestamp) as transaction_ts
from {{ source('raw', 'transactions') }}
