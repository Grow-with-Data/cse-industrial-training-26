"""approval: interrupt() until a manager decides each refund proposal (v8)."""
from __future__ import annotations

from shopwise.llm.workflows.inbox.state import InboxState
from shopwise.llm.workflows.inbox.nodes._common import progress
from shopwise.llm.workflows.inbox.nodes._refunds import ask_manager, proposals


def approval(state: InboxState) -> dict:
    """The pause: one manager decision per refund proposal. No proposals, no pause.

    RULE 1 IN PRACTICE. On every resume this node runs again from its first
    line, so the progress line below fires once per resume: watch for it twice in
    the notebook. It is a message, so twice is harmless. Nothing in this node
    moves money: the refund node does that, in the NEXT step, so by the time it
    runs every decision is already saved in a checkpoint."""
    pending = proposals(state)
    decisions = []
    for proposal in pending:
        progress("approval", f"Asking a manager about the refund on {proposal['order_id']}…")
        decisions.append(ask_manager(proposal) | {"proposal": proposal})
    return {"decisions": decisions}
