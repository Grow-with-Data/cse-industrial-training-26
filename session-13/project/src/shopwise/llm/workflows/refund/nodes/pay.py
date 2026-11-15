"""pay: issue what was approved, once."""
from __future__ import annotations

from shopwise.llm.workflows.inbox.nodes import settle
from shopwise.llm.workflows.refund.state import RefundState


def pay(state: RefundState) -> dict:
    return {"outcome": settle(state["decision"])}
