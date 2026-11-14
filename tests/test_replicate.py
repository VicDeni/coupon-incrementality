"""The dunnhumby converter produces the same table layout the main pipeline reads."""

import duckdb
import pandas as pd
import pytest

from coupon_incrementality import replicate
from coupon_incrementality.download import TABLES

FIXTURES = {
    "transaction_data.csv": (
        "household_key,BASKET_ID,DAY,PRODUCT_ID,QUANTITY,SALES_VALUE,STORE_ID,RETAIL_DISC,"
        "TRANS_TIME,WEEK_NO,COUPON_DISC,COUPON_MATCH_DISC\n"
        "2375,26984851472,1,1004906,1,1.39,364,-0.60,1631,1,0.0,0.0\n"
        "2375,26984851472,8,1033142,1,0.82,364,0.00,0905,2,-1.00,-0.50\n"
    ),
    "causal_data.csv": "PRODUCT_ID,STORE_ID,WEEK_NO,display,mailer\n1004906,364,1,0,A\n",
    "campaign_table.csv": "DESCRIPTION,household_key,CAMPAIGN\nTypeA,2375,26\n",
    "campaign_desc.csv": (
        "DESCRIPTION,CAMPAIGN,START_DAY,END_DAY\nTypeB,24,659,719\nTypeA,26,8,40\n"
    ),
    "coupon.csv": "COUPON_UPC,PRODUCT_ID,CAMPAIGN\n10000089061,27160,4\n",
    "coupon_redempt.csv": "household_key,DAY,COUPON_UPC,CAMPAIGN\n1,421,10000085364,8\n",
    "product.csv": (
        "PRODUCT_ID,MANUFACTURER,DEPARTMENT,BRAND,COMMODITY_DESC,SUB_COMMODITY_DESC,"
        "CURR_SIZE_OF_PRODUCT\n25671,2,GROCERY,National,FRZN ICE,ICE - CRUSHED/CUBED,22 LB\n"
    ),
    "hh_demographic.csv": (
        "AGE_DESC,MARITAL_STATUS_CODE,INCOME_DESC,HOMEOWNER_DESC,HH_COMP_DESC,"
        "HOUSEHOLD_SIZE_DESC,KID_CATEGORY_DESC,household_key\n"
        "65+,A,35-49K,Homeowner,2 Adults No Kids,2,None/Unknown,1\n"
    ),
}

# Columns of the 84.51 tables, as extracted by `download.py`.
EXPECTED_COLUMNS = {
    "transactions": {
        "household_id", "store_id", "basket_id", "product_id", "quantity", "sales_value",
        "retail_disc", "coupon_disc", "coupon_match_disc", "week", "transaction_timestamp",
    },
    "promotions": {"product_id", "store_id", "display_location", "mailer_location", "week"},
    "campaigns": {"campaign_id", "household_id"},
    "campaign_descriptions": {"campaign_id", "campaign_type", "start_date", "end_date"},
    "coupons": {"coupon_upc", "product_id", "campaign_id"},
    "coupon_redemptions": {"household_id", "coupon_upc", "campaign_id", "redemption_date"},
    "products": {
        "product_id", "manufacturer_id", "department", "brand", "product_category",
        "product_type", "package_size",
    },
    "demographics": {
        "household_id", "age", "income", "home_ownership", "marital_status",
        "household_size", "household_comp", "kids_count",
    },
}  # fmt: skip


@pytest.fixture
def converted(tmp_path):
    source = tmp_path / "csv"
    source.mkdir()
    for name, text in FIXTURES.items():
        (source / name).write_text(text)
    replicate.convert(source, tmp_path / "raw")
    return tmp_path / "raw"


def read(folder, table):
    return duckdb.connect().execute(f"select * from '{folder}/{table}.parquet'").df()


def test_every_table_has_the_main_pipeline_layout(converted):
    assert set(EXPECTED_COLUMNS) == set(TABLES)
    for table, columns in EXPECTED_COLUMNS.items():
        assert set(read(converted, table).columns) == columns, table


def test_discounts_become_positive(converted):
    transactions = read(converted, "transactions")
    assert (transactions[["retail_disc", "coupon_disc", "coupon_match_disc"]] >= 0).all().all()
    assert transactions.coupon_disc.max() == 1.0


def test_days_map_to_the_right_week_and_time(converted):
    transactions = read(converted, "transactions").sort_values("week")
    local = transactions.transaction_timestamp - pd.Timedelta(hours=5)
    assert local.dt.strftime("%Y-%m-%d %H:%M").tolist() == ["2000-01-03 16:31", "2000-01-10 09:05"]
    campaigns = read(converted, "campaign_descriptions").set_index("campaign_id")
    assert str(campaigns.loc[26, "start_date"])[:10] == "2000-01-10"  # day 8 opens week 2
    assert campaigns.campaign_type.tolist() == ["Type B", "Type A"]
