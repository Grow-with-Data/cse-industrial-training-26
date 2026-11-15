"""The inbox graph's nodes. Each one is a plain function: state in, updates out.

    intake      v8: the turn's message joins the conversation; per-turn keys reset
    summarize   v8: past SUMMARIZE_AFTER messages, older turns become one summary
    rewrite     v8: a follow-up becomes a standalone question (kb/query.py)
    triage      one structured call on the turn: S03's TriageResult
    split       one structured call: the turn's sub-issues, each with a category
    desks       orders · billing · policy · general, one sub-issue each
                v8: billing works out a refund request and PROPOSES it
    approval    v8: interrupt(): pause until a manager decides each proposal
    refund      v8: pay what was approved, once. The only node that moves money
    compose     one call: every desk's finding, in order, as one reply

recall and remember, the customer-memory nodes, are in llm/workflows/memory/nodes/.

Three kinds of node, and the graph does not care which is which:

    model node    triage, split, compose, summarize, rewrite: one LLM call, no tools
    agent node    orders, general, and billing's non-refund questions: a
                  create_agent with its own tools, invoked like any function
    code first    policy, and billing's refunds: facts fetched in Python, then
                  the policy graph answers, or a proposal waits for a manager

v8: NODES TALK WHILE THEY WORK
    A node can say what it is doing before it returns, through LangGraph's
    stream writer: emit({"type": "progress", "text": "Checking ORD-5001…"}). The
    caller sees it as a `custom` stream event, and workflows/events.py turns it
    into a typed `progress` or `tool` event. Nothing is printed here: the node
    does not know whether a terminal, a web page or nobody is listening.

v10: ONE NODE PER FILE
    This was one 568-line nodes.py. Each node now has its own file; helpers two
    or more nodes share live in the underscore files (_common, _desks, _refunds).
    This file re-exports every name, so `from shopwise.llm.workflows.inbox.nodes
    import compose` works exactly as `from workflows.inbox.nodes import compose` did.
"""
from __future__ import annotations

from shopwise.llm.workflows.inbox.nodes._common import as_block, emit, progress, reading, who
from shopwise.llm.workflows.inbox.nodes.intake import intake
from shopwise.llm.workflows.inbox.nodes.summarize import summarize
from shopwise.llm.workflows.inbox.nodes.rewrite import rewrite
from shopwise.llm.workflows.inbox.nodes.triage import triage_system, triage_node
from shopwise.llm.workflows.inbox.nodes.split import DESK_FOR, route_sub_issues, split
from shopwise.llm.workflows.inbox.nodes._desks import DESKS, finding, order_facts, order_of_refund, run_desk
from shopwise.llm.workflows.inbox.nodes.orders_desk import orders_desk
from shopwise.llm.workflows.inbox.nodes.general_desk import general_desk
from shopwise.llm.workflows.inbox.nodes.billing_desk import billing_desk
from shopwise.llm.workflows.inbox.nodes.policy_desk import policy_desk
from shopwise.llm.workflows.inbox.nodes._refunds import ask_manager, check_answer, proposals, refund_case, settle
from shopwise.llm.workflows.inbox.nodes.approval import approval
from shopwise.llm.workflows.inbox.nodes.refund import refund
from shopwise.llm.workflows.inbox.nodes.compose import describe, render_findings, compose

__all__ = [
    "as_block",
    "emit",
    "progress",
    "reading",
    "who",
    "intake",
    "summarize",
    "rewrite",
    "triage_system",
    "triage_node",
    "DESK_FOR",
    "route_sub_issues",
    "split",
    "DESKS",
    "order_facts",
    "run_desk",
    "finding",
    "order_of_refund",
    "orders_desk",
    "general_desk",
    "billing_desk",
    "policy_desk",
    "ask_manager",
    "check_answer",
    "proposals",
    "refund_case",
    "settle",
    "approval",
    "refund",
    "describe",
    "render_findings",
    "compose",
]
