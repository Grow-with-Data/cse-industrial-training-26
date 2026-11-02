"""ShopWise's return rules, written as code — project version v0.2.

The model reads the ticket and reports what the customer claims (TicketFacts).
This file decides what those claims mean under the returns policy. Arithmetic and
policy belong in code: models are unreliable at the first and have no authority
over the second.

The numbers are from session 2's returns_policy.md, sections 1.1 and 1.2.
"""

from __future__ import annotations

from typing import Literal

from schemas import TicketFacts

RETURN_DAYS = 30           # 1.1: most items, counted from delivery
OPENED_AUDIO_DAYS = 15     # 1.2: opened headphones, earbuds and speakers


def return_window(facts: TicketFacts) -> Literal["inside", "outside", "unknown"]:
    """Is the item inside the return window, going by what the customer told us?

    "unknown" is an honest answer and a common one: most tickets don't say how
    long ago the item arrived. This is a hint for the person handling the ticket,
    not a decision to send the customer; session 5 checks the real delivery date.
    """
    if facts.days_since_delivery is None:
        return "unknown"

    limit = RETURN_DAYS
    if facts.product_kind == "audio" and facts.opened:
        limit = OPENED_AUDIO_DAYS

    if facts.days_since_delivery <= limit:
        return "inside"
    return "outside"
