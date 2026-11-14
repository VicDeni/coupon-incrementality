-- Every redemption in the source keeps a row and a value.
select 'row count mismatch' as problem
where (select count(*) from {{ ref('int_redemption_values') }})
   <> (select count(*) from {{ ref('stg_coupon_redemptions') }})
