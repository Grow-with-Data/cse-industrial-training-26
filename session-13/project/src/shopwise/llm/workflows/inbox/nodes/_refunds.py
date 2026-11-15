"""Refunds, worked out in code: the case, the manager's form, the answer check, the payment."""
from __future__ import annotations

from langgraph.types import interrupt
from pydantic import ValidationError
from shopwise.core.constants import RETURN_WINDOW_DAYS
from shopwise.llm.middleware import Customer
from shopwise.llm.schemas import ManagerDecision
from shopwise.llm.llm_data.shopwise_data import ORDERS, REFUNDS
from shopwise.llm.tools import issue_refund
from shopwise.llm.workflows.inbox.state import InboxState
from shopwise.llm.workflows.inbox.nodes._common import emit
from shopwise.llm.workflows.inbox.nodes._desks import order_facts


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
