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

recall and remember, the customer-memory nodes, are in workflows/memory.py.

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
"""
from __future__ import annotations

import json
import re
from datetime import date
from functools import lru_cache

from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime
from langgraph.types import Send, interrupt
from pydantic import ValidationError

from core.config import settings
from core.constants import RETURN_WINDOW_DAYS
from kb.query import rewrite_query
from middleware import Customer, ShopContext, customer_for, desk_middleware, owner_of
from models import ask_json, build_model, stamp
from prompts.loader import load_prompts
from schemas import ManagerDecision, TicketSplit, TriageResult
from shopwise_data import CUSTOMERS, ORDERS, REFUNDS
from tools import (check_refund_status, get_order, get_product, get_shipment,
                   issue_refund, list_orders, search_policy)
from triage import triage_system
from workflows.corrective_rag import build_corrective
from workflows.inbox.state import RESET, DeskTask, InboxState

PROMPTS = load_prompts("inbox")
SPLIT_PROMPT = PROMPTS["split_prompt"]
DESK_PROMPT = PROMPTS["desk_prompt"]
COMPOSE_PROMPT = PROMPTS["compose_prompt"]
SUMMARIZE_PROMPT = PROMPTS["summarize_prompt"]


@lru_cache
def _model():
    return build_model()


def emit(event: dict) -> None:
    """Tell whoever is streaming this run what is happening (v8).

    get_stream_writer() only works inside a running graph. A node called
    directly, from a notebook cell, has nobody to tell, so this does nothing."""
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return
    writer(event)


def progress(node: str, text: str) -> None:
    emit({"type": "progress", "node": node, "text": text})


def as_block(ticket: dict) -> str:
    """The ticket wrapped in tags, so the model reads it as data, not
    instructions. A chat message has no subject line, so none is shown."""
    subject = f"Subject: {ticket['subject']}\n" if ticket.get("subject") else ""
    return f'<ticket id="{ticket["id"]}">\n{subject}{ticket["body"]}\n</ticket>'


def reading(state: InboxState) -> dict:
    """The ticket as the graph should understand it this turn: with the standalone
    rewrite as its body when there is one (v8), otherwise as it arrived."""
    if state.get("standalone"):
        return {**state["ticket"], "subject": "", "body": state["standalone"]}
    return state["ticket"]


def who(ticket: dict, runtime: Runtime[ShopContext]) -> Customer:
    """Whose ticket this is: the runtime context, set by the caller from the login.
    LangGraph Studio starts runs without one, so the sender is the fallback: same
    email, same customer_id, same authorize rail. Never the conversation."""
    return runtime.context or customer_for(ticket["from"])


# ── 1. a turn begins (v8) ────────────────────────────────────────────────────

def intake(state: InboxState) -> dict:
    """The turn's message joins the conversation, and last turn's working keys
    are cleared: `findings` through its reducer's RESET, the others by overwrite.

    The caller sends a ticket dict on every turn: the inbox sends the email as
    it arrived, chat.py sends the customer's line wrapped as one. Turning it into
    a HumanMessage HERE, inside the graph, means both callers get the same
    conversation and neither has to know how the state is shaped."""
    ticket = state["ticket"]
    text = f"{ticket['subject']}\n\n{ticket['body']}" if ticket.get("subject") else ticket["body"]
    return {"messages": [HumanMessage(text)], "findings": RESET, "standalone": None,
            "sub_issues": [], "memories": [], "decisions": [], "refunds": [],
            "remembered": [], "reply": ""}


def summarize(state: InboxState) -> dict:
    """Short-term memory inside the graph: keep a long thread inside a budget.

    Past SUMMARIZE_AFTER messages, every message but the newest SUMMARY_KEEP is
    folded into `summary` and deleted with RemoveMessage: add_messages, the
    reducer that appends, also deletes by id. Below the threshold this node
    makes no call and returns nothing.

    The thread's messages are only the customer's and ShopWise's, never tool
    calls (those stay inside the desks' own agents), so deleting from the front
    cannot orphan a tool result from the call that asked for it. A graph whose
    state does hold tool messages must cut at a turn boundary."""
    messages = state["messages"]
    if len(messages) <= settings.SUMMARIZE_AFTER:
        return {}
    old = messages[:-settings.SUMMARY_KEEP]
    progress("summarize", f"Summarizing {len(old)} older messages…")
    lines = "\n".join(f"{'Customer' if m.type == 'human' else 'ShopWise'}: {m.text}" for m in old)
    prompt = SUMMARIZE_PROMPT.format(summary=state.get("summary") or "(none)", messages=lines)
    summary = _model().invoke(prompt, config={"run_name": "summarize",
                                              **stamp("summarize_prompt", SUMMARIZE_PROMPT)})
    return {"summary": summary.text.strip(),
            "messages": [RemoveMessage(id=m.id) for m in old]}


def rewrite(state: InboxState) -> dict:
    """A follow-up becomes a standalone question, before anything reads it.

    No history, no call: the first message of a thread has nothing to resolve.
    REWRITE_FOLLOWUPS=false turns the node into a pass-through, which is how the
    notebook shows the bug it fixes."""
    history = state["messages"][:-1]
    if not settings.REWRITE_FOLLOWUPS or (not history and not state.get("summary")):
        return {}
    message = state["messages"][-1].text
    standalone = rewrite_query(history, message, state.get("summary", ""))
    if standalone != message:
        progress("rewrite", f'Read as: "{standalone}"')
    return {"standalone": standalone}


# ── 2. triage and split ──────────────────────────────────────────────────────

def triage_node(state: InboxState) -> dict:
    """Session 3's triage, now on the turn as the graph reads it."""
    result, _ = ask_json(_model(), TriageResult,
                         [("system", triage_system()), ("human", as_block(reading(state)))])
    return {"triage": result}


def split(state: InboxState) -> dict:
    """Every distinct issue in the turn, each with its own category."""
    result, _ = ask_json(_model(), TicketSplit,
                         [("system", SPLIT_PROMPT), ("human", as_block(reading(state)))])
    return {"sub_issues": result.sub_issues}


# ── 3. routing: a dict, not a model ──────────────────────────────────────────
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
    """One Send per sub-issue, straight to its desk."""
    return [Send(DESK_FOR[sub.category],
                 {"ticket": state["ticket"], "sub_issue": sub, "position": position})
            for position, sub in enumerate(state["sub_issues"])]


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


def orders_desk(task: DeskTask, runtime: Runtime[ShopContext]) -> dict:
    return run_desk("orders", task, runtime)


def general_desk(task: DeskTask, runtime: Runtime[ShopContext]) -> dict:
    return run_desk("general", task, runtime)


REFUND_ID = re.compile(r"\bREF-\d+\b", re.IGNORECASE)


def order_of_refund(text: str) -> str | None:
    """Customers name the refund ("REF-9001") more often than its order. Find the
    order in code, from the refund record; the authorize check still decides
    whether this customer may touch it."""
    match = REFUND_ID.search(text)
    refund = REFUNDS.get(match.group(0).upper()) if match else None
    return refund["order_id"] if refund else None


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


# ── 5. the policy desk: facts in code, then the rule ─────────────────────────

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


def policy_desk(task: DeskTask, runtime: Runtime[ShopContext]) -> dict:
    """Return and warranty questions: the order's facts, then the cited rule."""
    sub = task["sub_issue"]
    facts = None
    if sub.order_id:
        facts = order_facts(sub.order_id, str(task["ticket"]["received"]),
                            who(task["ticket"], runtime))
    progress("policy", "Searching the policies…")
    result = policy_graph().invoke({"question": sub.question})
    progress("policy", f"Found {len(result.get('documents', []))} policy passages")
    return finding("policy", task, note=result["answer"], facts=facts)


# ── 6. refunds: code works it out, a manager decides (v8) ────────────────────
# RETURN_WINDOW_DAYS moved to core/constants.py in v9: the refunds specialist reads it too.


def refund_case(order_id: str, ticket: dict, customer: Customer, reason: str) -> dict:
    """Work out a refund request in code. Returns {"note", "facts", "proposal"}.

    `proposal` is None when a hard rule refuses outright, and there is nothing
    for a person to decide: another customer's order, or a refund on it already
    paid. Anything else becomes a proposal with a recommendation the manager may
    overrule. Outside the return window is a judgement, not a rule: shops make
    exceptions, and making one is a person's call.
    """
    facts = order_facts(order_id, str(ticket["received"]), customer)
    if "error" in facts:
        return {"note": f"No refund possible: {facts['error']}.", "facts": None, "proposal": None}

    on_order = {rid: r for rid, r in REFUNDS.items() if r["order_id"] == facts["order_id"]}
    for rid, r in on_order.items():
        if r["status"] == "processed":
            return {"note": f"{rid} for ${r['amount']:.2f} on this order has already been paid. "
                            "Nothing more is due.", "facts": facts, "proposal": None}

    open_request = next((rid for rid, r in on_order.items() if r["status"] == "requested"), None)
    total = ORDERS[facts["order_id"]]["total"]
    amount = on_order[open_request]["amount"] if open_request else total
    days = facts.get("days_since_delivery")
    if days is None:
        recommendation, why = "reject", "not delivered yet: cancel the order rather than refund it"
    elif days > RETURN_WINDOW_DAYS:
        recommendation, why = "reject", f"outside the {RETURN_WINDOW_DAYS}-day return window ({days} days since delivery)"
    else:
        recommendation, why = "approve", f"inside the {RETURN_WINDOW_DAYS}-day return window ({days} days since delivery)"

    # The approval form: everything a manager needs to decide, and nothing that
    # is not plain JSON (interrupt rule 4: the payload is saved in a checkpoint).
    proposal = {"action": "issue_refund", "ticket_id": ticket["id"],
                "customer_id": customer.customer_id, "order_id": facts["order_id"],
                "amount": amount, "max_amount": total, "open_request": open_request,
                "reason": reason, "days_since_delivery": days,
                "recommendation": recommendation, "why": why}
    return {"note": f"A refund of ${amount:.2f} on {facts['order_id']} was put to a ShopWise "
                    "manager. What they decided is under REFUND OUTCOME.",
            "facts": facts, "proposal": proposal}


def check_answer(answer, proposal: dict) -> str | None:
    """What is wrong with a manager's answer, or None. Validate human input like
    any other input from outside: a person can mistype an amount too.

    It reads only the answer and the proposal, never the runtime context, and
    that is deliberate. On resume every earlier answer is REPLAYED through this
    loop, but under the context of the run doing the resuming. A check like
    `runtime.context.role == "manager"` here would refuse a customer's answer on
    the day they sent it, then accept the same saved answer when a manager
    resumes the thread for an unrelated reason, and pay it. A check that runs
    again on replay must give the same verdict on replay. Who may resume is
    decided before the resume, by the caller (workflows/events.stream_ticket)."""
    try:
        decision = ManagerDecision.model_validate(answer)
    except ValidationError as err:
        return "; ".join(e["msg"] for e in err.errors())
    if decision.type == "edit" and decision.amount > proposal["max_amount"]:
        return (f"${decision.amount:.2f} is more than the ${proposal['max_amount']:.2f} "
                f"paid for {proposal['order_id']}")
    return None


def ask_manager(proposal: dict) -> dict:
    """Pause until a manager sends a valid answer about one proposal.

    `interrupt(payload)` stops the run and saves it; the payload reaches the
    caller as the interrupt's value. `Command(resume=answer)` on the same thread
    restarts the node FROM ITS FIRST LINE, and this time interrupt() returns the
    answer instead of pausing.

    An invalid answer is not trusted: the loop asks again, with the problem in
    the payload. Safe under rule 3 (same interrupts, same order), because each
    resume replays the earlier answers in the order they were given."""
    request = proposal
    while True:
        answer = interrupt(request)
        problem = check_answer(answer, proposal)
        if problem is None:
            return ManagerDecision.model_validate(answer).model_dump()
        request = {**proposal, "error": problem}


def proposals(state: InboxState) -> list[dict]:
    """Every refund proposal in this turn, in the order the customer raised them."""
    return [f["proposal"] | {"position": f["position"]}
            for f in sorted(state["findings"], key=lambda f: f["position"]) if f.get("proposal")]


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


def settle(decision: dict) -> dict:
    """Carry out one decision. Returns what happened, for compose."""
    proposal = decision["proposal"]
    if decision["type"] == "reject":
        return {"position": proposal["position"], "order_id": proposal["order_id"],
                "status": "declined", "reason": decision["reason"]}
    amount = decision["amount"] if decision["type"] == "edit" else proposal["amount"]
    emit({"type": "tool", "node": "refund", "name": "issue_refund", "status": "started"})
    paid = issue_refund.invoke({"order_id": proposal["order_id"], "amount": amount,
                                "reason": f"{proposal['reason']} [{proposal['ticket_id']}]"})
    emit({"type": "tool", "node": "refund", "name": "issue_refund", "status": "finished"})
    if decision["type"] == "edit":
        paid |= {"asked_for": proposal["amount"], "edit_reason": decision["reason"]}
    return paid | {"position": proposal["position"]}


def refund(state: InboxState) -> dict:
    """Pay what the manager approved. The only node in the graph that moves money.

    Its own node, so the pause and the payment are in different steps: the
    decision is checkpointed before any money moves. If the process dies inside
    this node, the next run on the thread starts here again with the saved
    decisions, and issue_refund's idempotency check finds the refund already
    paid instead of paying twice (durable execution, notebook section 9)."""
    return {"refunds": [settle(d) for d in state.get("decisions", [])]}


# ── 7. compose ───────────────────────────────────────────────────────────────

def describe(outcome: dict | None) -> str:
    """A refund outcome in words, for compose's prompt."""
    if outcome is None:
        return "not decided"
    if outcome.get("status") == "declined":
        return f"declined by ShopWise. Reason: {outcome['reason']}"
    if "error" in outcome:
        return f"could not be paid: {outcome['error']}"
    issued = f"issued, {outcome['refund_id']} for ${outcome['amount']:.2f}, to the original payment method"
    if "asked_for" in outcome:
        issued += (f". A manager changed the amount from ${outcome['asked_for']:.2f}. "
                   f"Their reason: {outcome['edit_reason']}")
    return issued


def render_findings(findings: list[dict], refunds: list[dict]) -> str:
    """The findings as compose_prompt reads them. Formatting happens HERE.
    v8: a finding that carried a proposal gets its outcome, matched by position."""
    outcome = {r["position"]: r for r in refunds}
    blocks = []
    for number, f in enumerate(findings, start=1):
        block = f"Issue {number}. The customer asked: {f['question']}"
        if f["facts"]:
            block += f"\n   Order facts: {json.dumps(f['facts'])}"
        block += f"\n   Answer found: {f['note']}"
        if f.get("proposal"):
            block += f"\n   REFUND OUTCOME: {describe(outcome.get(f['position']))}"
        blocks.append(block)
    return "\n\n".join(blocks)


# A citation looks like [returns-policy · 1. The return window]. Any other
# square brackets the model adds ([ORD-5003], [1]) look like sources and are not.
NOT_A_CITATION = re.compile(r"\s*\[(?![a-z0-9-]+ · )[^\]]*\]")


def compose(state: InboxState, runtime: Runtime[ShopContext]) -> dict:
    """One reply that answers every sub-issue, added to the conversation (v8).

    Its tokens are the only ones the customer sees stream (workflows/events.py
    filters on this node's name)."""
    findings = sorted(state["findings"], key=lambda f: f["position"])
    customer = who(state["ticket"], runtime)
    first_reply = not any(m.type == "ai" for m in state["messages"])
    memories = "\n".join(f"- {m}" for m in state.get("memories", [])) or "(nothing yet)"
    prompt = COMPOSE_PROMPT.format(name=CUSTOMERS[customer.email]["name"].split()[0],
                                   first="yes" if first_reply else "no",
                                   memories=memories,
                                   findings=render_findings(findings, state.get("refunds", [])))
    reply = _model().invoke(prompt, config={"run_name": "compose",
                                            **stamp("compose_prompt", COMPOSE_PROMPT)})
    text = NOT_A_CITATION.sub("", reply.text)
    # The message keeps the model call's id. The caller already watched this
    # reply stream token by token; with the same id, LangGraph knows the message
    # this node returns IS that one and does not stream it a second time.
    return {"reply": text, "messages": [AIMessage(text, id=reply.id)]}
