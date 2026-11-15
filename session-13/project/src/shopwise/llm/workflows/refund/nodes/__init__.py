"""The refund graph's three nodes. Every one is v8 code, called, not copied.

    assess     v8's refund_case: the facts, the hard rules, a proposal or a refusal
    approval   v8's ask_manager: interrupt() until a manager sends a valid answer
    pay        v8's settle: issue_refund once (idempotent), or record the decline

The pause and the payment are still in different nodes, for v8's reason: the
manager's decision is checkpointed before any money moves.

v10: one node per file; this file re-exports them.
"""
from __future__ import annotations

from shopwise.llm.workflows.refund.nodes.assess import assess, needs_manager
from shopwise.llm.workflows.refund.nodes.approval import approval
from shopwise.llm.workflows.refund.nodes.pay import pay

__all__ = [
    "assess",
    "needs_manager",
    "approval",
    "pay",
]
