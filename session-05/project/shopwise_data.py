"""ShopWise's records: the mock backend, loaded as five plain dictionaries — project version v2.

The data lives once, in data/backend.yml, and every snapshot from v2 reads it
through this loader (ADR-0006). In a real system this file is where you would open a
database connection or call an API. Keep it boring: everything that shapes the data
for the model belongs in tools.py.

    uv run python shopwise_data.py      # the whole store on one screen
"""

from __future__ import annotations

import yaml

from config import settings

# Read once, at import. A malformed fixture fails here, naming the file, not halfway
# through an agent run.
_BACKEND = yaml.safe_load((settings.DATA_DIR / "backend.yml").read_text(encoding="utf-8"))

CUSTOMERS: dict[str, dict] = _BACKEND["customers"]   # keyed by email, what a customer types
PRODUCTS: dict[str, dict] = _BACKEND["products"]     # keyed by SKU
ORDERS: dict[str, dict] = _BACKEND["orders"]         # keyed by order id
SHIPMENTS: dict[str, dict] = _BACKEND["shipments"]   # keyed by ORDER id, not tracking id
REFUNDS: dict[str, dict] = _BACKEND["refunds"]       # keyed by refund id


if __name__ == "__main__":
    for name, table in [("CUSTOMERS", CUSTOMERS), ("PRODUCTS", PRODUCTS), ("ORDERS", ORDERS),
                        ("SHIPMENTS", SHIPMENTS), ("REFUNDS", REFUNDS)]:
        print(f"\n{name} ({len(table)})")
        for key, row in table.items():
            print(f"  {key:22} {row}")
