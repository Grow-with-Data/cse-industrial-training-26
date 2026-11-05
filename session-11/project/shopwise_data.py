"""ShopWise's records: the mock backend, loaded as five plain dictionaries — project version v8.

v8: REFUNDS also reads settings.REFUND_LEDGER, the refunds issued so far. Until
v7 an issued refund lived only in this process's REFUNDS dict and vanished on
exit. That was harmless while nothing could survive an exit. From v8 a paused
refund survives a restart, so the record of a PAID refund has to as well:
otherwise "was it already paid?" gets a different answer after a crash, and the
same refund is paid twice. In a real system the database is the ledger; here it
is one JSON line per refund, appended by tools.issue_refund.

The data lives once, in labs/data/backend.yml, and every snapshot from v2 reads it
through this loader (ADR-0006). In a real system this file is where you would open a
database connection or call an API. Keep it boring: everything that shapes the data
for the model belongs in tools.py.

    uv run python shopwise_data.py      # the whole store on one screen
"""

from __future__ import annotations

import json

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


def load_ledger() -> None:
    """Replay the ledger over REFUNDS: each line is one refund's latest record."""
    if settings.REFUND_LEDGER.exists():
        for line in settings.REFUND_LEDGER.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                REFUNDS[entry.pop("refund_id")] = entry


def record_refund(refund_id: str) -> None:
    """Append one refund's current record to the ledger (tools.issue_refund calls this)."""
    settings.REFUND_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with settings.REFUND_LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"refund_id": refund_id, **REFUNDS[refund_id]}) + "\n")


def reset_ledger() -> None:
    """Forget every issued refund: delete the ledger and reload the fixture's refunds.
    For notebooks and demos, so a story can be told twice from the same start."""
    settings.REFUND_LEDGER.unlink(missing_ok=True)
    REFUNDS.clear()
    REFUNDS.update(yaml.safe_load((settings.DATA_DIR / "backend.yml").read_text(encoding="utf-8"))["refunds"])


load_ledger()


if __name__ == "__main__":
    for name, table in [("CUSTOMERS", CUSTOMERS), ("PRODUCTS", PRODUCTS), ("ORDERS", ORDERS),
                        ("SHIPMENTS", SHIPMENTS), ("REFUNDS", REFUNDS)]:
        print(f"\n{name} ({len(table)})")
        for key, row in table.items():
            print(f"  {key:22} {row}")
