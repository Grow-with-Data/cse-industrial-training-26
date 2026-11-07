"""The refunds specialist: calls the refund graph inside a node, and translates.

workflows/refund/ has its own state (RefundState), so this node is the adapter:
supervisor state in, RefundState for the graph, a finding and an outcome back
out. Its interrupt() pauses the whole supervisor run, and a resume comes back
through here (workflows/refund/graph.py explains how).

A handoff from orders arrives in state["handoff"]: the order and a three-line
summary, which this node reads instead of the supervisor's task, then clears.
"""
from __future__ import annotations

from langgraph.runtime import Runtime

from middleware import ShopContext
from workflows.inbox.nodes import order_of_refund, who
from workflows.refund import build_refund
from workflows.specialists.common import report


# Compiled once, at import, as a MODULE-LEVEL object, not behind a function.
# LangGraph finds a subgraph that a node calls by looking at what the node's
# function refers to; a compiled graph it can see is one get_state(config,
# subgraphs=True) can show, mid-pause. Hidden behind lru_cache, the pause and
# the resume still work, but the child's state is invisible from the parent.
refund_graph = build_refund()


def refunds(state: dict, runtime: Runtime[ShopContext]) -> dict:
    task, handoff, customer = state.get("task") or {}, state.get("handoff"), who(state["ticket"], runtime)
    order_id = (handoff or {}).get("order_id") or task.get("order_id") or order_of_refund(task.get("question", ""))
    reason = handoff["summary"] if handoff else task.get("question", "")
    task = {**task, "question": task.get("question") or reason, "position": task.get("position", 0)}
    if not order_id:
        return report("refunds", task, note="No order or refund id was named, so there is "
                      "nothing to work out yet. Ask the customer which order.") | {"handoff": None}
    out = refund_graph.invoke({"ticket": state["ticket"], "order_id": order_id, "reason": reason,
                                 "position": task["position"], "email": customer.email,
                                 "customer_id": customer.customer_id})
    update = report("refunds", task, out["note"], out.get("facts"), out.get("proposal"))
    return update | {"handoff": None, "refunds": [out["outcome"]] if out.get("outcome") else []}
