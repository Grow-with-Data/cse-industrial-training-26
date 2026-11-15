"""The supervisor's routing decision for one step."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


# ── v9: the supervisor's decision (workflows/supervisor/nodes.py) ───────────
class Route(BaseModel):
    """One step of the supervisor: who works next, on what, and why.

    `next` is a closed list, so the supervisor cannot invent a desk. `task` is
    what the specialist sees INSTEAD of the conversation: a specialist with a
    clean context answers one question well. `reason` is for people: it is
    written into route_log, where a trace, a reviewer and a test can read it."""

    next: Literal["orders", "knowledge", "refunds", "market", "FINISH"] = Field(
        description="The specialist who works next, or FINISH when every part of the "
                    "customer's message has been handled.")
    task: str = Field(
        default="",
        description="One self-contained task for that specialist, with every order id, "
                    "refund id, product and amount it needs. Empty when next is FINISH.")
    order_id: str | None = Field(
        default=None, description="The order the task is about, if it is about one.")
    reason: str = Field(description="One short sentence: why this specialist, now.")
