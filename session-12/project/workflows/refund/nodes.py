"""The refund graph's three nodes. Every one is v8 code, called, not copied.

    assess     v8's refund_case: the facts, the hard rules, a proposal or a refusal
    approval   v8's ask_manager: interrupt() until a manager sends a valid answer
    pay        v8's settle: issue_refund once (idempotent), or record the decline

The pause and the payment are still in different nodes, for v8's reason: the
manager's decision is checkpointed before any money moves.
"""
from __future__ import annotations

from middleware import Customer
from workflows.inbox.nodes import ask_manager, progress, refund_case, settle
from workflows.refund.state import RefundState


def assess(state: RefundState) -> dict:
    customer = Customer(email=state["email"], customer_id=state["customer_id"])
    progress("refunds", f"Checking refund eligibility for {state['order_id']}…")
    return refund_case(state["order_id"], state["ticket"], customer, state["reason"])


def needs_manager(state: RefundState) -> str:
    """A proposal waits for a manager; a hard refusal has nothing to decide."""
    return "approval" if state.get("proposal") else "__end__"


def approval(state: RefundState) -> dict:
    """RULE 1 still applies: on resume this node runs again from its first line,
    so the progress line fires once per resume. Nothing here moves money."""
    proposal = state["proposal"] | {"position": state["position"]}
    progress("refunds", f"Asking a manager about the refund on {proposal['order_id']}…")
    return {"decision": ask_manager(proposal) | {"proposal": proposal}}


def pay(state: RefundState) -> dict:
    return {"outcome": settle(state["decision"])}
