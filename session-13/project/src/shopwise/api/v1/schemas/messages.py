"""HTTP shapes for sending a message and getting ShopWise's reply."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from shopwise.api.v1.schemas.common import ThreadStatus


class MessageIn(BaseModel):
    text: str = Field(min_length=1, max_length=4000,
                      description="What the customer wrote. 1 to 4000 characters.",
                      examples=["Where is ORD-5002?"])


class CitationOut(BaseModel):
    label: str = Field(examples=["returns-policy · 1. The return window"])
    source: str = Field(examples=["returns-policy"])
    section: str = Field(examples=["1. The return window"])


class ApprovalRequest(BaseModel):
    """The form a manager must answer before a refund is paid (session 11's interrupt).

    extra="ignore": the graph's payload carries internal fields too (ticket id,
    customer id); the API passes on only what a manager needs to decide.
    """

    model_config = ConfigDict(extra="ignore")

    interrupt_id: str = Field(description="Send it back with the decision when a thread has two.")
    order_id: str = Field(examples=["ORD-5001"])
    amount: float = Field(description="What the customer asked for, in USD.", examples=[129.0])
    max_amount: float = Field(description="What was paid for the order: the most a refund can be.")
    recommendation: str = Field(description="What the rules suggest: approve or reject.",
                                examples=["approve"])
    why: str = Field(examples=["inside the 30-day return window (12 days since delivery)"])
    error: str | None = Field(default=None,
                              description="Why the previous decision was refused, if one was.")


class MessageOut(BaseModel):
    thread_id: str
    reply: str = Field(description="Empty when the turn paused for a manager.")
    citations: list[CitationOut]
    status: ThreadStatus
    interrupt: ApprovalRequest | None = Field(
        default=None, description="Present when status is awaiting_approval.")
