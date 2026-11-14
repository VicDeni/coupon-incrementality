"""Paths, assumptions and constants shared by every phase."""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("COUPON_ROOT", Path(__file__).resolve().parents[2]))
RAW_DIR = ROOT / "data" / "raw"
WAREHOUSE = Path(os.environ.get("COUPON_WAREHOUSE", ROOT / "data" / "warehouse.duckdb"))
RESULTS_DIR = Path(os.environ.get("COUPON_RESULTS_DIR", ROOT / "reports" / "results"))
FIGURES_DIR = ROOT / "reports" / "figures"

SEED = 20261114

# --- Assumptions that are not in the data -------------------------------------------------
# Grocery gross margin. Results are reported at the midpoint and across the whole range.
MARGIN_LOW, MARGIN_BASE, MARGIN_HIGH = 0.25, 0.275, 0.30
# Cost of printing and mailing one campaign to one household, in dollars.
DISTRIBUTION_COST = 0.50

# --- Study design -------------------------------------------------------------------------
FIRST_FULL_WEEK = 2  # week 1 holds only two days
LAST_WEEK = 53
# Type A campaigns with at least 12 weeks of purchase history before they start.
TYPE_A_CAMPAIGNS = (8, 13, 18)
HISTORY_WEEKS = 12
POST_WEEKS = 4
# Households whose estimated chance of being targeted falls outside this band have no
# comparable counterpart and are dropped.
PROPENSITY_BOUNDS = (0.05, 0.95)


def save_result(name: str, payload: dict) -> Path:
    """Write one phase's numbers to reports/results/<name>.json."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    return path


def load_result(name: str) -> dict:
    return json.loads((RESULTS_DIR / f"{name}.json").read_text())
