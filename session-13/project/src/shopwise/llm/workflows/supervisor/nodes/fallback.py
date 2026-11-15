"""specialist_failed: what the team does when a specialist breaks. New in v10.

    builder.add_node("orders", orders, retry_policy=RETRY, error_handler=specialist_failed)

THE TAXONOMY (what to do with a failure depends on what kind it is)
    transient        a timeout, a dropped connection: RETRY it (RetryPolicy)
    model-fixable    a tool said "no such order": give it back to the model
                     (the v4 rails already do this inside each agent)
    user-fixable     a missing order number: ask the customer (interrupt)
    unexpected       anything else, or a transient one that kept failing:
                     DO NOT kill the turn. Land here.

WHAT THIS HANDLER DOES
    LangGraph calls it with the failed node's name and exception (NodeError)
    after the retries are spent. It writes a finding that says so honestly, and
    sends the turn back to the supervisor, which carries on with the other
    issues and lets compose write the reply. The customer reads "I couldn't
    check your order right now, a colleague will follow up" instead of the
    whole conversation failing because one carrier API was down.

    The exception's type goes to the log and the trace; its text never goes to
    the customer.
"""
from __future__ import annotations

from langgraph.errors import NodeError
from langgraph.types import Command

from shopwise.core.logging_config import get_logger
from shopwise.llm.workflows.inbox.nodes import progress
from shopwise.llm.workflows.specialists.common import report

log = get_logger(__name__)

FALLBACK_NOTE = ("I couldn't complete this check right now. A ShopWise colleague will look "
                 "into it and follow up with you.")


def specialist_failed(state: dict, error: NodeError) -> Command:
    log.warning("specialist %s failed: %s", error.node, type(error.error).__name__)
    progress(error.node, f"{error.node} is unavailable right now; carrying on without it.")
    task = state.get("task") or {"question": "(the customer's question)", "position": 0}
    return Command(goto="supervisor", update=report(error.node, task, note=FALLBACK_NOTE))
