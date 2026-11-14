"""Phase 7 - replication on the original dunnhumby release (two years, ~2,500 households).

The dunnhumby "The Complete Journey" files are free but sit behind a click-through form
(https://www.dunnhumby.com/source-files/), so they cannot be fetched automatically.
Download and unzip them, then run:

    python -m coupon_incrementality.replicate path/to/dunnhumby_csvs

This converts the CSVs to the same parquet layout the main pipeline reads, builds a
second warehouse with the same dbt project, and reruns the triple difference on it.
Results are written to reports/results/replication/.

Every output column is aliased explicitly, because DuckDB keeps the CSV's own
capitalisation otherwise. The column mapping follows the dunnhumby user guide.
Differences handled here:
  * discounts are stored as negative numbers            -> flipped to positive
  * dates are day numbers 1-711, weeks 1-102            -> mapped onto a calendar that
                                                           starts on Monday 2000-01-03
  * campaign types are spelled "TypeA"                  -> "Type A"
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import duckdb

from .config import ROOT

EPOCH = "2000-01-03"  # an arbitrary Monday: day 1 of the study
LAST_WEEK = 102
_DATE = "date '{epoch}' + cast({day} - 1 as integer)"

CONVERSIONS = {
    "transactions": """
        select
            household_key as household_id, store_id as store_id, basket_id as basket_id,
            product_id as product_id, quantity as quantity, sales_value as sales_value,
            abs(retail_disc) as retail_disc, abs(coupon_disc) as coupon_disc,
            abs(coupon_match_disc) as coupon_match_disc, week_no as week,
            -- the main pipeline expects UTC timestamps five hours ahead of store time
            cast({date} as timestamp)
                + to_minutes(cast(60 * (cast(trans_time as integer) // 100)
                                  + cast(trans_time as integer) % 100 + 300 as integer))
                as transaction_timestamp
        from read_csv_auto('{src}/transaction_data.csv')
    """,
    "promotions": """
        select product_id as product_id, store_id as store_id,
               cast(display as varchar) as display_location,
               cast(mailer as varchar) as mailer_location, week_no as week
        from read_csv_auto('{src}/causal_data.csv')
    """,
    "campaigns": """
        select campaign as campaign_id, household_key as household_id
        from read_csv_auto('{src}/campaign_table.csv')
    """,
    "campaign_descriptions": """
        select campaign as campaign_id, 'Type ' || right(description, 1) as campaign_type,
               {start_date} as start_date, {end_date} as end_date
        from read_csv_auto('{src}/campaign_desc.csv')
    """,
    "coupons": """
        select coupon_upc as coupon_upc, product_id as product_id, campaign as campaign_id
        from read_csv_auto('{src}/coupon.csv')
    """,
    "coupon_redemptions": """
        select household_key as household_id, coupon_upc as coupon_upc,
               campaign as campaign_id,
               {date} as redemption_date
        from read_csv_auto('{src}/coupon_redempt.csv')
    """,
    "products": """
        select product_id as product_id, manufacturer as manufacturer_id,
               department as department, brand as brand,
               commodity_desc as product_category, sub_commodity_desc as product_type,
               curr_size_of_product as package_size
        from read_csv_auto('{src}/product.csv')
    """,
    "demographics": """
        select household_key as household_id, age_desc as age, income_desc as income,
               homeowner_desc as home_ownership, marital_status_code as marital_status,
               household_size_desc as household_size, hh_comp_desc as household_comp,
               kid_category_desc as kids_count
        from read_csv_auto('{src}/hh_demographic.csv')
    """,
}


def convert(source: Path, target: Path) -> None:
    """Write the eight dunnhumby CSVs as parquet files in the main pipeline's layout."""
    target.mkdir(parents=True, exist_ok=True)
    dates = {
        "date": _DATE.format(epoch=EPOCH, day="day"),
        "start_date": _DATE.format(epoch=EPOCH, day="start_day"),
        "end_date": _DATE.format(epoch=EPOCH, day="end_day"),
    }
    with duckdb.connect() as con:
        for table, sql in CONVERSIONS.items():
            query = sql.format(src=source, **dates)
            con.execute(f"copy ({query}) to '{target / (table + '.parquet')}' (format parquet)")


def main(source: str) -> None:
    raw = ROOT / "data" / "replication" / "raw"
    warehouse = ROOT / "data" / "replication" / "warehouse.duckdb"
    convert(Path(source), raw)
    env = os.environ | {
        "COUPON_RAW_DIR": str(raw),
        "COUPON_WAREHOUSE": str(warehouse),
        "COUPON_RESULTS_DIR": str(ROOT / "reports" / "results" / "replication"),
        "DBT_PROFILES_DIR": ".",
    }
    dbt_vars = f"{{week_one_start: '{EPOCH}', last_week: {LAST_WEEK}}}"
    subprocess.run(["dbt", "build", "--vars", dbt_vars], cwd=ROOT / "dbt", env=env, check=True)
    subprocess.run([sys.executable, "-m", "coupon_incrementality.triple_diff"], env=env, check=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
