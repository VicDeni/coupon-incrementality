-- Every household has exactly one row for every week.
select household_id, count(*) as n_weeks
from {{ ref('fct_household_week') }}
group by 1
having count(*) <> {{ var('last_week') }} - {{ var('first_week') }} + 1
