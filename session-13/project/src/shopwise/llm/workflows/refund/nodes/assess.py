"""assess: work the refund out in code; needs_manager routes on the result."""
from __future__ import annotations

from shopwise.llm.middleware import Customer
from shopwise.llm.workflows.inbox.nodes import progress, refund_case
from shopwise.llm.workflows.refund.state import RefundState


def assess(state: RefundState) -> dict:
    customer = Customer(email=state["email"], customer_id=state["customer_id"])
    progress("refunds", f"Checking refund eligibility for {state['order_id']}…")
    return refund_case(state["order_id"], state["ticket"], customer, state["reason"])


def needs_manager(state: RefundState) -> str:
    """A proposal waits for a manager; a hard refusal has nothing to decide."""
    return "approval" if state.get("proposal") else "__end__"
