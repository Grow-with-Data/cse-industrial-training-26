"""split: one structured call for the turn's sub-issues; route_sub_issues sends each to a desk."""
from __future__ import annotations

from langgraph.types import Send
from shopwise.llm.models import ask_json
from shopwise.llm.schemas import TicketSplit
from shopwise.llm.workflows.inbox.state import InboxState
from shopwise.llm.workflows.inbox.nodes._common import SPLIT_PROMPT, _model, as_block, reading


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
