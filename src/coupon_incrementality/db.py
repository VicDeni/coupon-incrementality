"""Read access to the DuckDB warehouse that dbt builds."""

from __future__ import annotations

import duckdb
import pandas as pd

from .config import WAREHOUSE


def query(sql: str) -> pd.DataFrame:
    if not WAREHOUSE.exists():
        raise FileNotFoundError(f"{WAREHOUSE} not found. Run `make data` first.")
    with duckdb.connect(str(WAREHOUSE), read_only=True) as con:
        return con.execute(sql).df()
