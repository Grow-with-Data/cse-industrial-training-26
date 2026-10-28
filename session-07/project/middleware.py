"""The v4 safety rails — every guard the agent runs inside, in one place. New in v4.

Middleware is code that runs AROUND the agent loop: once before the agent starts,
before or after each model call, or wrapped around each tool call. The agent
itself does not change: `create_agent(..., middleware=build_middleware(...))` is
the whole wiring. Session 6's SummarizationMiddleware was already one entry.

Each rail does one job. Read them top to bottom; build_middleware() at the end
puts them in order and says why that order.

    1. content_filter    ends the run politely on known injection phrases
    2. PII rules         card numbers never reach the model; emails are kept
                         out of tool results and replies
    3. authorize         a customer can only touch their OWN orders and refunds
    4. limits            a fuse: too many model or tool calls ends the run
    5. retry + errors    a tool that RAISES a transient error is tried again;
                         if it still fails, the model gets a safe message
    6. blocked_reply     a reply the provider emptied (its own safety filter,
                         or an output cap) becomes an escalation, never a blank
    7. refund_approval   issue_refund pauses the run until a manager decides

    uv run python test_rails.py      # every rail, tested with no key
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    PIIMiddleware,
    ToolCallLimitMiddleware,
    ToolErrorMiddleware,
    ToolRetryMiddleware,
    after_model,
    before_agent,
    wrap_tool_call,
)
from langchain.messages import AIMessage, RemoveMessage, ToolMessage

from shopwise_data import CUSTOMERS, ORDERS, REFUNDS


# ── Who is asking ────────────────────────────────────────────────────────────
@dataclass
class Customer:
    """The logged-in customer: RUNTIME CONTEXT, supplied by the app on every call.

    It comes from the login (in chat.py, `--as <email>`), never from the
    conversation: anything in the messages can be typed by anyone.
    """
    email: str
    customer_id: str


def customer_for(email: str) -> Customer:
    """The context for a known customer. Raises KeyError for an unknown email."""
    email = email.strip().lower()
    return Customer(email=email, customer_id=CUSTOMERS[email]["customer_id"])


# ── 1. content_filter ────────────────────────────────────────────────────────
# Deterministic and cheap: a phrase list, checked before the model is called at
# all. It misses any attack it has no phrase for (the notebook shows one), which
# is why it is the FIRST rail and not the only one.
BANNED_PHRASES = [
    "ignore your instructions",
    "ignore your previous instructions",
    "ignore all previous instructions",
    "you are now",
    "system message",
    "[system:",
]
FILTER_REPLY = ("I can't help with that request here. A member of the ShopWise team "
                "will review your message and follow up.")


@before_agent(can_jump_to=["end"])
def content_filter(state, runtime):
    text = state["messages"][-1].text.lower()
    if any(phrase in text for phrase in BANNED_PHRASES):
        return {"messages": [AIMessage(FILTER_REPLY)], "jump_to": "end"}
    return None


# ── 2. PII rules ─────────────────────────────────────────────────────────────
# One decision per kind of PII: who actually needs to see it?
#
#   credit card  nobody here. Redacted on the way IN, so the model, the trace
#                and the checkpoint never hold it.
#   email        the model needs it to call lookup_customer, so input is left
#                alone. It is redacted where nobody needs it: in tool results,
#                and in the replies the assistant writes.
PII_RULES = [
    PIIMiddleware("credit_card", strategy="redact"),
    PIIMiddleware("email", strategy="redact", apply_to_input=False,
                  apply_to_tool_results=True, apply_to_output=True),
]


# ── 3. authorize ─────────────────────────────────────────────────────────────
def owner_of(tool_name: str, args: dict) -> str | None:
    """Which customer_id a tool call would touch. None means public or unknown.

    Unknown ids are let through on purpose: the tool itself answers
    {"error": "no order with ID ..."}, which is the honest reply.
    """
    if tool_name == "lookup_customer":
        record = CUSTOMERS.get(args.get("email", "").strip().lower())
        return record["customer_id"] if record else None
    if tool_name == "list_orders":
        return args.get("customer_id", "").strip().upper() or None
    if tool_name in ("get_order", "get_shipment", "issue_refund"):
        order = ORDERS.get(args.get("order_id", "").strip().upper())
        return order["customer_id"] if order else None
    if tool_name == "check_refund_status":
        refund = REFUNDS.get(args.get("refund_id", "").strip().upper())
        order = ORDERS.get(refund["order_id"]) if refund else None
        return order["customer_id"] if order else None
    return None                      # get_product: the catalogue is public


@wrap_tool_call
def authorize(request, handler):
    """Run the tool only if its record belongs to the logged-in customer.

    The model may well ASK for someone else's order: it is trying to help
    whoever is typing. Whether it may is not the model's decision, so it is code.
    """
    call = request.tool_call
    me: Customer | None = request.runtime.context
    owner = owner_of(call["name"], call["args"])
    if owner is not None and (me is None or owner != me.customer_id):
        return ToolMessage(
            content=json.dumps({"error": "not authorised: that record belongs to another "
                                         "customer. Do not reveal anything about it."}),
            tool_call_id=call["id"], name=call["name"], status="error")
    return handler(request)


# ── 4. limits — the runaway-loop fuse ────────────────────────────────────────
# One customer message never needs more than a handful of model calls. If it
# takes more, something is looping, and your API credit is paying for it. These
# replace v3's MAX_STEPS, which counted graph steps, middleware hooks included,
# rather than the calls that cost money.
LIMITS = [
    ModelCallLimitMiddleware(run_limit=8),
    ToolCallLimitMiddleware(run_limit=10),
]


# ── 5. retry, then a safe error ──────────────────────────────────────────────
# Only TRANSIENT failures are retried: a timeout or a dropped connection may
# work next time. A tool that RETURNS {"error": ...} did its job and is never
# retried; a bug in our own code (KeyError, TypeError) still raises, loudly.
TRANSIENT = (TimeoutError, ConnectionError)

RETRY = ToolRetryMiddleware(max_retries=2, retry_on=TRANSIENT, initial_delay=0.5,
                            on_failure="error")    # still failing? raise, to TOOL_ERRORS


def tool_unavailable(exc: Exception, request) -> str | None:
    """What the model is told when a tool is still failing after its retries.

    It names the tool and the error TYPE, never the raw message: an exception's
    text can carry hostnames, keys or customer data. Anything not transient
    returns None, which lets it raise.
    """
    if not isinstance(exc, TRANSIENT):
        return None
    return json.dumps({"error": f"{request.tool_call['name']} is not responding right now "
                                f"({type(exc).__name__}). Tell the customer you could not "
                                "check this, and that a colleague will follow up."})


TOOL_ERRORS = ToolErrorMiddleware(tool_unavailable)


# ── 6. blocked_reply — when the provider sends back nothing ──────────────────
# The provider runs its own safety filter before ours, and when it blocks, the
# message usually just comes back EMPTY: no error is raised. How it says why
# depends on the route (the notebook, section 8):
#
#   Gemini direct     finish_reason "SAFETY" (output) or prompt_feedback with a
#                     block_reason (input)
#   OpenRouter        finish_reason "content_filter", and the host's own reason
#                     in native_finish_reason
#   either            finish_reason "length" / "MAX_TOKENS": not a block, the
#                     output cap ran out, often inside a reasoning model's thinking
#
# Sent as-is, any of these is a blank reply to a customer.
BLOCKED_REPLY = ("I'm not able to answer that one myself. I've passed your message "
                 "to a member of the ShopWise team, who will reply to you directly.")


def blocked_reason(message) -> str | None:
    """Why a model reply is empty, or None if it isn't. Works for every route above."""
    if message.text.strip() or message.tool_calls:
        return None
    meta = message.response_metadata
    finish = str(meta.get("finish_reason") or "").lower()
    if finish in ("safety", "content_filter", "prohibited_content", "blocklist", "spii"):
        return "output_blocked"
    block = str((meta.get("prompt_feedback") or {}).get("block_reason") or "")
    if block.upper() not in ("", "0", "NONE", "BLOCK_REASON_UNSPECIFIED"):
        return "input_blocked"
    if finish in ("length", "max_tokens"):
        return "out_of_tokens"
    return "empty"


@after_model
def blocked_reply(state, runtime):
    last = state["messages"][-1]
    reason = blocked_reason(last)
    if reason is None:
        return None                                  # a normal reply, or a tool request
    # Replace the empty message rather than add to it. The reason stays in the
    # metadata, so the trace says WHY the customer got an escalation.
    return {"messages": [RemoveMessage(id=last.id),
                         AIMessage(BLOCKED_REPLY, response_metadata={"escalated": reason})]}


# ── 7. refund_approval — a person signs for the money ────────────────────────
# issue_refund is in TOOLS from v4, but it never runs on the model's say-so: the
# run pauses and waits for a manager's decision. A paused run has to be saved
# somewhere to be resumed, so this rail needs a checkpointer.
REFUND_APPROVAL = HumanInTheLoopMiddleware(
    interrupt_on={"issue_refund": {"allowed_decisions": ["approve", "reject"]}},
    description_prefix="Refund needs a manager's approval",
)


def build_middleware(summarizer) -> list:
    """The stack, in order. Order matters:

    - before_* hooks run top to bottom, so the free content filter runs first and
      a filtered message costs no model call at all;
    - PII runs before anything sends text to the model;
    - wrap_tool_call rails nest in list order, outermost first: authorize wraps
      the error handler, which wraps retry, which wraps the tool. A refused call
      is never retried, and only a spent retry reaches the error handler;
    - after_* hooks run bottom to top, so refund_approval sees the model's
      request first, closest to the tool it guards.
    """
    return [
        content_filter,
        *PII_RULES,
        authorize,
        TOOL_ERRORS,
        RETRY,
        *LIMITS,
        summarizer,
        blocked_reply,
        REFUND_APPROVAL,
    ]
