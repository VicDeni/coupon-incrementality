-- Nothing is lost or double counted between the raw log and the household x week panel.
with raw as (
    select sum(sales_value) as total from {{ ref('stg_transactions') }}
),

panel as (
    select sum(spend) + sum(fuel_spend) as total from {{ ref('fct_household_week') }}
)

select raw.total as raw_total, panel.total as panel_total
from raw
cross join panel
where abs(raw.total - panel.total) > 0.01
