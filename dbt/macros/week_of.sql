{# Dataset week number (1-53) that contains a calendar date. Weeks run Monday-Sunday. #}
{% macro week_of(date_expr) -%}
    (1 + date_diff('day', date '{{ var("week_one_start") }}', cast({{ date_expr }} as date)) // 7)
{%- endmacro %}
