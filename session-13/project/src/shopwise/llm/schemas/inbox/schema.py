"""The inbox's split: one ticket in, its sub-issues out."""
from __future__ import annotations

from pydantic import BaseModel, Field
from shopwise.llm.schemas.triage.schema import TicketCategory


# ── v7: the inbox graph's split node ─────────────────────────────────────────
class SubIssue(BaseModel):
    """One thing a ticket asks about. Each is answered by exactly one desk."""

    question: str = Field(
        description="This one issue, restated as a self-contained question in the "
                    "customer's own terms. Someone reading only this line must "
                    "understand what is being asked.")
    order_id: str | None = Field(
        default=None,
        description="The order this issue is about, e.g. ORD-5001, if the ticket names "
                    "one for it. Never guess one.")
    category: TicketCategory = Field(
        description="What this one issue is about. Pick the single best fit.")


class TicketSplit(BaseModel):
    """The split node's output: every distinct issue in the ticket, in order."""

    sub_issues: list[SubIssue] = Field(
        min_length=1,
        description="One entry per distinct issue, in the order the customer raised "
                    "them. A ticket about one thing has exactly one entry.")
