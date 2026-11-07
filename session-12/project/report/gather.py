"""[ADV] The Shop Owner's Daily Brief — the three gather steps.

    uv run python report/gather.py              # the brief's facts for 6 August 2026
    uv run python report/gather.py 2026-08-04

Each function is pure Python over the fixtures: no model, no key, the same
numbers every time. They are the brief's first three nodes. Building the rest —
three writer nodes that turn each result into one paragraph, and an `assemble`
node that dates and joins them — is the [ADV] exercise in session 10's
exercises.ipynb, and it is the textbook WORKFLOW: the steps never vary, so
nothing in it should be an agent.

Every function returns raw data, not sentences. Formatting is the writer
nodes' job (session 10, section 5: store raw data, not formatted text).

"Category" is the word for what a ticket is about. Products have a product TYPE, and
sales are grouped by that.
"""
from __future__ import annotations

import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

if __name__ == "__main__":                      # run as a script: import from project/
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from core.config import settings  # noqa: E402
from shopwise_data import ORDERS, PRODUCTS, REFUNDS  # noqa: E402

REPORT_DAY = "2026-08-06"        # the last day in data/sales.yml


def load_tickets() -> list[dict]:
    """tickets.yml as plain dicts: the hand-labelled categories are all the brief needs."""
    return yaml.safe_load((settings.DATA_DIR / "tickets.yml").read_text(encoding="utf-8"))


def load_sales() -> list[dict]:
    return yaml.safe_load((settings.DATA_DIR / "sales.yml").read_text(encoding="utf-8"))


def sales_by_product_type(day: str = REPORT_DAY) -> dict:
    """Units and revenue per product type on `day`, beside the average day of
    the seven before it — so a writer can say "up" or "down", not just "how much"."""
    sales = load_sales()
    week = {str(date.fromisoformat(day) - timedelta(days=n)) for n in range(1, 8)}

    today: dict[str, dict] = {}
    last_week: Counter = Counter()
    for line in sales:
        product_type = PRODUCTS[line["sku"]]["product_type"]
        revenue = line["qty"] * line["unit_price"]
        if line["date"] == day:
            row = today.setdefault(product_type, {"units": 0, "revenue": 0.0})
            row["units"] += line["qty"]
            row["revenue"] += revenue
        elif line["date"] in week:
            last_week[product_type] += revenue

    types = sorted({p["product_type"] for p in PRODUCTS.values()})
    return {
        "day": day,
        "by_type": {t: {**today.get(t, {"units": 0, "revenue": 0.0}),
                        "avg_revenue_last_7_days": round(last_week[t] / 7, 2)}
                    for t in types},
        "total_revenue": round(sum(r["revenue"] for r in today.values()), 2),
    }


def open_refunds(as_of: str = REPORT_DAY) -> list[dict]:
    """Every refund nobody has finished, and how many days it has waited."""
    waiting = []
    for refund_id, refund in REFUNDS.items():
        if refund["status"] in ("processed", "rejected"):
            continue
        age = (date.fromisoformat(as_of) - date.fromisoformat(refund["requested_date"])).days
        waiting.append({"refund_id": refund_id, "order_id": refund["order_id"],
                        "customer_id": ORDERS[refund["order_id"]]["customer_id"],
                        "amount": refund["amount"], "status": refund["status"],
                        "days_waiting": age})
    return sorted(waiting, key=lambda r: r["days_waiting"], reverse=True)


def ticket_volume(day: str = REPORT_DAY) -> dict:
    """Tickets received on `day`, counted by category.

    The fixture's hand-labelled categories stand in for triage's output. In
    production this would read what the triage node wrote, which is one more
    reason triage is a node of its own and not a step inside an agent."""
    counts = Counter(t["category"] for t in load_tickets() if str(t["received"]) == day)
    return {"day": day, "total": sum(counts.values()), "by_category": dict(counts.most_common())}


if __name__ == "__main__":
    import json
    day = sys.argv[1] if len(sys.argv) > 1 else REPORT_DAY
    print(json.dumps({"sales": sales_by_product_type(day), "refunds": open_refunds(day),
                      "tickets": ticket_volume(day)}, indent=2))
