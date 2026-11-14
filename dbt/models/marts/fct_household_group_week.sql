-- Stacked triple-difference panel: one block per Type B / Type C campaign.
-- Within a block: every household x product group (covered / sibling / placebo / other)
-- x week, from 16 weeks before the campaign to 8 weeks after it. Balanced, zeros included.
with campaigns as (
    select
        campaign_id,
        campaign_type,
        start_week,
        end_week,
        greatest(start_week - {{ var('pre_window_weeks') }}, {{ var('first_week') }} + 1) as window_start,
        least(end_week + {{ var('post_window_weeks') }}, {{ var('last_week') }}) as window_end
    from {{ ref('stg_campaign_descriptions') }}
    where campaign_id in (select distinct campaign_id from {{ ref('int_campaign_product_groups') }})
),

grid as (
    select
        c.campaign_id,
        c.campaign_type,
        c.start_week,
        c.end_week,
        h.household_id,
        h.primary_store_id,
        g.product_group,
        w.week
    from campaigns as c
    cross join {{ ref('int_households') }} as h
    cross join (select unnest(['covered', 'sibling', 'placebo', 'other']) as product_group) as g
    cross join (select unnest(range({{ var('first_week') }}, {{ var('last_week') }} + 1)) as week) as w
    where w.week between c.window_start and c.window_end
),

-- other campaigns the household received that overlap this campaign's run
concurrent as (
    select a.campaign_id, b.household_id, count(*) as n_overlapping_campaigns
    from campaigns as a
    inner join {{ ref('int_household_campaigns') }} as b
        on a.campaign_id <> b.campaign_id
        and b.start_week <= a.end_week
        and b.end_week >= a.start_week
    group by 1, 2
),

purchases as (
    select
        g.campaign_id,
        t.household_id,
        g.product_group,
        t.week,
        sum(t.sales_value) as spend,
        sum(t.quantity)    as units
    from {{ ref('stg_transactions') }} as t
    inner join {{ ref('int_campaign_product_groups') }} as g on t.product_id = g.product_id
    inner join campaigns as c
        on g.campaign_id = c.campaign_id and t.week between c.window_start and c.window_end
    group by 1, 2, 3, 4
)

select
    g.campaign_id,
    g.campaign_type,
    g.household_id,
    g.product_group,
    g.week,
    hc.household_id is not null as is_recipient,
    coalesce(ov.n_overlapping_campaigns, 0) as n_overlapping_campaigns,
    case
        when g.week < g.start_week then 'pre'
        when g.week <= g.end_week then 'during'
        else 'post'
    end as period,
    -- weeks relative to the campaign: negative before the start, 0 while it runs,
    -- positive after it ends
    case
        when g.week < g.start_week then g.week - g.start_week
        when g.week <= g.end_week then 0
        else g.week - g.end_week
    end as rel_week,
    coalesce(p.spend, 0)          as spend,
    coalesce(p.units, 0)          as units,
    coalesce(pr.display_share, 0) as display_share,
    coalesce(pr.mailer_share, 0)  as mailer_share
from grid as g
left join purchases as p
    on g.campaign_id = p.campaign_id
    and g.household_id = p.household_id
    and g.product_group = p.product_group
    and g.week = p.week
left join {{ ref('int_household_campaigns') }} as hc
    on g.campaign_id = hc.campaign_id and g.household_id = hc.household_id
left join concurrent as ov
    on g.campaign_id = ov.campaign_id and g.household_id = ov.household_id
left join {{ ref('int_group_promotions') }} as pr
    on g.campaign_id = pr.campaign_id
    and g.product_group = pr.product_group
    and g.primary_store_id = pr.store_id
    and g.week = pr.week
