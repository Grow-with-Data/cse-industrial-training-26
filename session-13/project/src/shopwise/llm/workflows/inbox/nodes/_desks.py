"""What the four desks share: their agents, the brief they get, the finding they return."""
from __future__ import annotations

import re
from datetime import date
from functools import lru_cache
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.runtime import Runtime
from shopwise.llm.middleware import Customer, ShopContext, desk_middleware, owner_of
from shopwise.llm.models import stamp
from shopwise.llm.llm_data.shopwise_data import CUSTOMERS, REFUNDS
from shopwise.llm.tools import check_refund_status, get_order, get_product, get_shipment, list_orders, search_policy
from shopwise.llm.workflows.corrective_rag import build_corrective
from shopwise.llm.workflows.inbox.state import DeskTask
from shopwise.llm.workflows.inbox.nodes._common import DESK_PROMPT, _model, emit, who


# ── 4. the agent desks ───────────────────────────────────────────────────────
# Least privilege by construction: a desk cannot misuse a tool it was never
# given. No agent holds issue_refund: from v8 only the billing desk's refund
# path calls it, in code, after a manager has decided.
DESKS = {
    "orders": {
        "remit": "You handle where orders are and when they ship or arrive.",
        "tools": [get_order, list_orders, get_shipment, search_policy],
    },
    "billing": {
        "remit": "You handle charges and refunds: what was charged, and the status of "
                 "refunds already requested.",
        "tools": [get_order, check_refund_status],
    },
    "general": {
        "remit": "You handle product questions, complaints and anything no other desk owns.",
        "tools": [get_order, list_orders, get_shipment, get_product,
                  check_refund_status, search_policy],
    },
}


@lru_cache
def desk_agent(desk: str):
    """One small agent per desk, built once, inside v7's rails."""
    return create_agent(
        model=_model(),
        tools=DESKS[desk]["tools"],
        system_prompt=DESK_PROMPT.format(desk=desk, remit=DESKS[desk]["remit"]),
        middleware=desk_middleware(),
        context_schema=Customer,
    )


def brief(task: DeskTask, customer: Customer) -> str:
    """The desk's one message. Who the customer is comes from the runtime context."""
    sub = task["sub_issue"]
    name = CUSTOMERS[customer.email]["name"]
    return (f"Customer: {name}, customer_id {customer.customer_id}\n"
            f"Order named for this issue: {sub.order_id or 'none'}\n"
            f"Issue: {sub.question}")


def finding(desk: str, task: DeskTask, note: str, facts: dict | None = None,
            proposal: dict | None = None) -> dict:
    """What every desk returns: raw data, one dict, inside a one-item list.
    `proposal` is v8's: a refund the billing desk wants a manager to decide."""
    sub = task["sub_issue"]
    return {"findings": [{"position": task["position"], "desk": desk,
                          "category": sub.category, "question": sub.question,
                          "facts": facts, "note": note, "proposal": proposal}]}


def run_desk(desk: str, task: DeskTask, runtime: Runtime[ShopContext]) -> dict:
    """Run a desk's agent on one sub-issue, reporting each tool call as it happens.

    v7 called .invoke() and saw nothing until the agent finished. v8 streams the
    agent's updates INSIDE the node and emits a `tool` event per call and per
    result, so the customer sees "checking get_shipment…" while it happens. The
    agent's own tokens never reach the customer: they are a nested graph's, and
    only compose's tokens are the reply (workflows/events.py)."""
    customer = who(task["ticket"], runtime)
    reply = ""
    for update in desk_agent(desk).stream(
            {"messages": [HumanMessage(brief(task, customer))]},
            config={"run_name": f"{desk}_desk", **stamp("desk_prompt", DESK_PROMPT)},
            context=customer, stream_mode="updates"):
        for delta in update.values():
            messages = delta.get("messages", []) if isinstance(delta, dict) else []
            for m in messages if isinstance(messages, list) else [messages]:
                if m.type == "ai":
                    for call in m.tool_calls:
                        emit({"type": "tool", "node": desk, "name": call["name"], "status": "started"})
                    if not m.tool_calls:
                        reply = m.text
                elif m.type == "tool":
                    emit({"type": "tool", "node": desk, "name": m.name, "status": "finished"})
    return finding(desk, task, note=reply)


REFUND_ID = re.compile(r"\bREF-\d+\b", re.IGNORECASE)


def order_of_refund(text: str) -> str | None:
    """Customers name the refund ("REF-9001") more often than its order. Find the
    order in code, from the refund record; the authorize check still decides
    whether this customer may touch it."""
    match = REFUND_ID.search(text)
    refund = REFUNDS.get(match.group(0).upper()) if match else None
    return refund["order_id"] if refund else None


def order_facts(order_id: str, as_of: str, customer: Customer) -> dict:
    """The facts a policy rule gets applied to, fetched by code, every time.

    Calling a tool's `.invoke` directly goes AROUND the agent's middleware, so
    the authorize rail does not run. The same check is applied by hand, with the
    same function: a path that skips the rails must carry the rule itself.
    days_since_delivery is counted here: models are unreliable at date arithmetic.
    """
    owner = owner_of("get_order", {"order_id": order_id})
    if owner is not None and owner != customer.customer_id:
        return {"error": "not authorised: that order belongs to another customer"}

    order = get_order.invoke({"order_id": order_id})
    if "error" in order:
        return order

    facts = {"order_id": order["order_id"], "status": order["status"],
             "order_date": order["order_date"], "delivered_date": order["delivered_date"],
             "items": []}
    for item in order["items"]:
        product = get_product.invoke({"sku_or_name": item["sku"]})
        facts["items"].append({"name": item["name"], "product_type": product.get("product_type"),
                               "warranty_months": product.get("warranty_months"),
                               "in_stock": bool(product.get("stock"))})
    if order["delivered_date"]:
        delivered = date.fromisoformat(order["delivered_date"])
        facts["days_since_delivery"] = (date.fromisoformat(as_of) - delivered).days
        facts["as_of"] = as_of
    shipment = get_shipment.invoke({"order_id": order_id})
    if "error" not in shipment:
        facts["shipment"] = shipment
    return facts


@lru_cache
def policy_graph():
    """Built once, with the shape settings choose (RAG_FUSION, RAG_HYDE, RAG_GRADE)."""
    return build_corrective()
