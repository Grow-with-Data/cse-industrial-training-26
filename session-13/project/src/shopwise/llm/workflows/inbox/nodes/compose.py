"""compose: one call that turns every desk's finding, in order, into one reply."""
from __future__ import annotations

import json
import re
from langchain.messages import AIMessage
from langgraph.runtime import Runtime
from shopwise.llm.middleware import ShopContext
from shopwise.llm.models import stamp
from shopwise.llm.llm_data.shopwise_data import CUSTOMERS
from shopwise.llm.workflows.inbox.state import InboxState
from shopwise.llm.workflows.inbox.nodes._common import COMPOSE_PROMPT, _model, who


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
