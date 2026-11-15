"""refund: pay what was approved, once. The only node that moves money (v8)."""
from __future__ import annotations

from shopwise.llm.workflows.inbox.state import InboxState
from shopwise.llm.workflows.inbox.nodes._refunds import settle


def refund(state: InboxState) -> dict:
    """Pay what the manager approved. The only node in the graph that moves money.

    Its own node, so the pause and the payment are in different steps: the
    decision is checkpointed before any money moves. If the process dies inside
    this node, the next run on the thread starts here again with the saved
    decisions, and issue_refund's idempotency check finds the refund already
    paid instead of paying twice (durable execution, notebook section 9)."""
    return {"refunds": [settle(d) for d in state.get("decisions", [])]}
