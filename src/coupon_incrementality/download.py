"""Fetch the Complete Journey tables.

84.51 distributes the dataset inside the `completejourney-py` package on PyPI as parquet
files. We download that exact wheel, check its hash, and extract the tables to data/raw/.
The wheel is treated as a data archive: nothing in it is imported or executed.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from .config import RAW_DIR

PACKAGE = "completejourney-py==0.1.0"
SHA256 = "7b8aa74688b37522ef2ef6f0623dd8b394636655c60cfc9adae18f0dff96c80b"
TABLES = (
    "transactions",
    "promotions",
    "campaigns",
    "campaign_descriptions",
    "coupons",
    "coupon_redemptions",
    "products",
    "demographics",
)


def main() -> None:
    if all((RAW_DIR / f"{t}.parquet").exists() for t in TABLES):
        print(f"Raw tables already in {RAW_DIR}")
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [sys.executable, "-m", "pip", "download", PACKAGE, "--no-deps", "-q", "-d", tmp],
            check=True,
        )
        wheel = next(Path(tmp).glob("*.whl"))
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
        if digest != SHA256:
            raise RuntimeError(f"Unexpected hash for {wheel.name}: {digest}")
        with zipfile.ZipFile(wheel) as archive:
            for table in TABLES:
                data = archive.read(f"completejourney_py/data/{table}.parquet")
                (RAW_DIR / f"{table}.parquet").write_bytes(data)
    print(f"Extracted {len(TABLES)} tables to {RAW_DIR}")


if __name__ == "__main__":
    main()
