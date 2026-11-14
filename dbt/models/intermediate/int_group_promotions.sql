-- In-store promotion intensity for each campaign's product groups, by store and week:
-- the sales-weighted share of the group's products that were on display / in the mailer.
-- Used as a control so a display that happens to coincide with a campaign is not
-- credited to the coupon.
with product_weight as (
    select product_id, sum(sales_value) as sales
    from {{ ref('stg_transactions') }}
    group by 1
),

group_weight as (
    select g.campaign_id, g.product_group, sum(w.sales) as group_sales
    from {{ ref('int_campaign_product_groups') }} as g
    inner join product_weight as w on g.product_id = w.product_id
    group by 1, 2
),

windows as (
    select
        campaign_id,
        greatest(start_week - {{ var('pre_window_weeks') }}, {{ var('first_week') }} + 1) as window_start,
        least(end_week + {{ var('post_window_weeks') }}, {{ var('last_week') }}) as window_end
    from {{ ref('stg_campaign_descriptions') }}
),

promo as (
    select
        g.campaign_id,
        g.product_group,
        pr.store_id,
        pr.week,
        sum(case when pr.on_display then w.sales else 0 end) as display_sales,
        sum(case when pr.in_mailer then w.sales else 0 end) as mailer_sales
    from {{ ref('stg_promotions') }} as pr
    inner join {{ ref('int_campaign_product_groups') }} as g on pr.product_id = g.product_id
    inner join windows as win
        on g.campaign_id = win.campaign_id and pr.week between win.window_start and win.window_end
    inner join product_weight as w on pr.product_id = w.product_id
    group by 1, 2, 3, 4
)

select
    p.campaign_id,
    p.product_group,
    p.store_id,
    p.week,
    p.display_sales / gw.group_sales as display_share,
    p.mailer_sales / gw.group_sales as mailer_share
from promo as p
inner join group_weight as gw
    on p.campaign_id = gw.campaign_id and p.product_group = gw.product_group
