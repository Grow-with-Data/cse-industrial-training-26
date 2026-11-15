"""approval: pause for a manager (v8's ask_manager)."""
from __future__ import annotations

from shopwise.llm.workflows.inbox.nodes import ask_manager, progress
from shopwise.llm.workflows.refund.state import RefundState


def approval(state: RefundState) -> dict:
    """RULE 1 still applies: on resume this node runs again from its first line,
    so the progress line fires once per resume. Nothing here moves money."""
    proposal = state["proposal"] | {"position": state["position"]}
    progress("refunds", f"Asking a manager about the refund on {proposal['order_id']}…")
    return {"decision": ask_manager(proposal) | {"proposal": proposal}}
