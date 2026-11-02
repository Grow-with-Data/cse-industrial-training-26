"""The inbox graph's nodes. Each one is a plain function: state in, updates out.

    triage    one structured call on the whole ticket: S03's TriageResult
    split     one structured call: the ticket's sub-issues, each with a category
    desks     orders · billing · policy · general, one sub-issue each
    compose   one call: every desk's finding, in order, as one reply

Three kinds of node, and the graph does not care which is which:

    model node    triage, split, compose: one LLM call, no tools
    agent node    orders, billing, general: a create_agent with its own tools,
                  invoked like any function
    code first    policy: fetches the order's facts in Python, then calls the
                  policy graph in workflows/corrective_rag/

Routing is not a node. `route_sub_issues` reads each sub-issue's category and
looks the desk up in DESK_FOR: a dict lookup, zero model calls.
"""
from __future__ import annotations

import json
import re
from datetime import date
from functools import lru_cache

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.runtime import Runtime
from langgraph.types import Send

from middleware import Customer, customer_for, desk_middleware, owner_of
from schemas import TicketSplit, TriageResult
from shopwise_data import CUSTOMERS
from tools import (check_refund_status, get_order, get_product, get_shipment,
                   list_orders, search_policy)
from triage import PROMPTS, stamp, triage_system
from utils import ask_json, build_model
from workflows.corrective_rag import build_corrective
from workflows.inbox.state import DeskTask, InboxState

SPLIT_PROMPT = PROMPTS["split_prompt"]
DESK_PROMPT = PROMPTS["desk_prompt"]
COMPOSE_PROMPT = PROMPTS["compose_prompt"]


@lru_cache
def _model():
    return build_model()


def as_block(ticket: dict) -> str:
    """The ticket wrapped in tags, so the model reads it as data, not
    instructions. Ticket.as_block's shape, for a plain dict: the multi-issue
    tickets carry no category, so they are not schemas.Ticket objects."""
    return f'<ticket id="{ticket["id"]}">\nSubject: {ticket["subject"]}\n{ticket["body"]}\n</ticket>'


def who(ticket: dict, runtime: Runtime[Customer]) -> Customer:
    """The logged-in customer. inbox.py passes it as runtime context, built from
    the ticket's sender. LangGraph Studio starts runs without one, so the sender
    is the fallback: same email, same Customer, same authorize rail."""
    return runtime.context or customer_for(ticket["from"])


# ── 1. triage and split ──────────────────────────────────────────────────────

def triage_node(state: InboxState) -> dict:
    """Session 3's triage, unchanged in substance, now as a node: the same prompt
    and the same TriageResult. It runs once, on the whole ticket: its category
    files the ticket and its urgency orders the queue. Answering uses the
    sub-issues' categories instead."""
    result, _ = ask_json(_model(), TriageResult,
                         [("system", triage_system()), ("human", as_block(state["ticket"]))])
    return {"triage": result}


def split(state: InboxState) -> dict:
    """Every distinct issue in the ticket, each with its own category."""
    result, _ = ask_json(_model(), TicketSplit,
                         [("system", SPLIT_PROMPT), ("human", as_block(state["ticket"]))])
    return {"sub_issues": result.sub_issues}


# ── 2. routing: a dict, not a model ──────────────────────────────────────────
# Every one of the nine ticket categories lands on a desk. `general` is the
# catch-all, and it is named honestly: it answers what no specialist owns.
DESK_FOR = {
    "order_status": "orders",
    "shipping": "orders",
    "billing": "billing",
    "refund": "billing",
    "return": "policy",
    "warranty": "policy",
    "product_question": "general",
    "complaint": "general",
    "other": "general",
}


def route_sub_issues(state: InboxState) -> list[Send]:
    """One Send per sub-issue, straight to its desk. How many desks run is
    decided here, at run time, by what the customer wrote."""
    sends = []
    for position, sub in enumerate(state["sub_issues"]):
        task = {"ticket": state["ticket"], "sub_issue": sub, "position": position}
        sends.append(Send(DESK_FOR[sub.category], task))
    return sends


# ── 3. the agent desks ───────────────────────────────────────────────────────
# Least privilege by construction: a desk cannot misuse a tool it was never
# given. Nothing here can issue a refund: v7 has no pause for a manager, and
# session 11 adds one. lookup_customer is not needed: the desk is told who the
# customer is, by code, in its brief.
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
    """One small agent per desk, built once. Same rails as v6's agent minus the
    two that need a conversation (middleware.desk_middleware)."""
    return create_agent(
        model=_model(),
        tools=DESKS[desk]["tools"],
        system_prompt=DESK_PROMPT.format(desk=desk, remit=DESKS[desk]["remit"]),
        middleware=desk_middleware(),
        context_schema=Customer,
    )


def brief(task: DeskTask, customer: Customer) -> str:
    """The desk's one message. Who the customer is comes from the runtime
    context, written in by code, so a desk never has to guess or look it up."""
    sub = task["sub_issue"]
    name = CUSTOMERS[customer.email]["name"]
    return (f"Customer: {name}, customer_id {customer.customer_id}\n"
            f"Order named for this issue: {sub.order_id or 'none'}\n"
            f"Issue: {sub.question}")


def finding(desk: str, task: DeskTask, note: str, facts: dict | None = None) -> dict:
    """What every desk returns: raw data, one dict, inside a one-item list.
    `findings` has an append reducer, so a desk returns a list to append."""
    sub = task["sub_issue"]
    return {"findings": [{"position": task["position"], "desk": desk,
                          "category": sub.category, "question": sub.question,
                          "facts": facts, "note": note}]}


def run_desk(desk: str, task: DeskTask, runtime: Runtime[Customer]) -> dict:
    """Invoke a desk's agent on one sub-issue. An agent is just a node."""
    customer = who(task["ticket"], runtime)
    result = desk_agent(desk).invoke(
        {"messages": [HumanMessage(brief(task, customer))]},
        config={"run_name": f"{desk}_desk", **stamp("desk_prompt", DESK_PROMPT)},
        context=customer)
    return finding(desk, task, note=result["messages"][-1].text)


def orders_desk(task: DeskTask, runtime: Runtime[Customer]) -> dict:
    return run_desk("orders", task, runtime)


def billing_desk(task: DeskTask, runtime: Runtime[Customer]) -> dict:
    return run_desk("billing", task, runtime)


def general_desk(task: DeskTask, runtime: Runtime[Customer]) -> dict:
    return run_desk("general", task, runtime)


# ── 4. the policy desk: facts in code, then the rule ─────────────────────────

def order_facts(order_id: str, as_of: str, customer: Customer) -> dict:
    """The facts a policy rule gets applied to, fetched by code, every time.

    v6 left this to the agent, and on some runs it answered "can I return it?"
    without ever looking the order up. Here there is no choice to get wrong.

    Calling a tool's `.invoke` directly goes AROUND the agent's middleware, so
    the authorize rail does not run. The same check is applied by hand, with
    the same function: a path that skips the rails must carry the rule itself.

    days_since_delivery is counted here, from the ticket's received date.
    Models are unreliable at date arithmetic.
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
                               "warranty_months": product.get("warranty_months")})
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


def policy_desk(task: DeskTask, runtime: Runtime[Customer]) -> dict:
    """Return and warranty questions: the order's facts, then the cited rule.

    The rule comes from the policy graph, which answers only the rule. Joining
    "delivered 7 May" with "30 days" is compose's job, from raw data.
    """
    sub = task["sub_issue"]
    facts = None
    if sub.order_id:
        facts = order_facts(sub.order_id, str(task["ticket"]["received"]),
                            who(task["ticket"], runtime))
    result = policy_graph().invoke({"question": sub.question})
    return finding("policy", task, note=result["answer"], facts=facts)


# ── 5. compose ───────────────────────────────────────────────────────────────

def render_findings(findings: list[dict]) -> str:
    """The findings as compose_prompt reads them. Formatting happens HERE, in the
    node that needs it: the state holds the raw dicts."""
    blocks = []
    for number, f in enumerate(findings, start=1):
        block = f"Issue {number}. The customer asked: {f['question']}"
        if f["facts"]:
            block += f"\n   Order facts: {json.dumps(f['facts'])}"
        block += f"\n   Answer found: {f['note']}"
        blocks.append(block)
    return "\n\n".join(blocks)


# A citation looks like [returns-policy · 1. The return window]. compose_prompt
# says those are the only square brackets allowed, and models still add
# [ORD-5003] or [1] now and then. A customer reading "[ORD-5001]" after "I
# cannot discuss that" sees a label that looks like a source and isn't one. The
# prompt asks; this guarantees.
NOT_A_CITATION = re.compile(r"\s*\[(?![a-z0-9-]+ · )[^\]]*\]")


def compose(state: InboxState, runtime: Runtime[Customer]) -> dict:
    """One reply that answers every sub-issue.

    Sorted by `position`, because the reducer appends in whatever order the
    parallel desks' writes are applied, and the customer's order is the one
    that matters to them.
    """
    findings = sorted(state["findings"], key=lambda f: f["position"])
    customer = who(state["ticket"], runtime)
    first_name = CUSTOMERS[customer.email]["name"].split()[0]
    prompt = COMPOSE_PROMPT.format(name=first_name, findings=render_findings(findings))
    reply = _model().invoke(prompt, config={"run_name": "compose",
                                            **stamp("compose_prompt", COMPOSE_PROMPT)})
    return {"reply": NOT_A_CITATION.sub("", reply.text)}
