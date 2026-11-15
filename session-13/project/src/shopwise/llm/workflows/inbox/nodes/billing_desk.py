"""The billing desk: refund requests become proposals; other billing questions go to an agent."""
from __future__ import annotations

from langgraph.runtime import Runtime
from shopwise.llm.middleware import ShopContext
from shopwise.llm.workflows.inbox.state import DeskTask
from shopwise.llm.workflows.inbox.nodes._common import progress, who
from shopwise.llm.workflows.inbox.nodes._desks import finding, order_of_refund, run_desk
from shopwise.llm.workflows.inbox.nodes._refunds import refund_case


def billing_desk(task: DeskTask, runtime: Runtime[ShopContext]) -> dict:
    """v8: a refund request for a named order is worked out in code and becomes a
    PROPOSAL for a manager (section 6 below). The desk does not pause and does
    not pay: the approval node pauses, the refund node pays. Every other billing
    question still goes to the desk's agent."""
    sub = task["sub_issue"]
    order_id = sub.order_id or order_of_refund(sub.question)
    if sub.category == "refund" and order_id:
        progress("billing", f"Checking refund eligibility for {order_id}…")
        case = refund_case(order_id, task["ticket"], who(task["ticket"], runtime), sub.question)
        return finding("billing", task, note=case["note"], facts=case["facts"],
                       proposal=case["proposal"])
    return run_desk("billing", task, runtime)
