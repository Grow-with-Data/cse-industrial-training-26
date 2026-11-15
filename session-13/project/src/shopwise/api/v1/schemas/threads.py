"""HTTP shapes for threads: a conversation between one customer and ShopWise.

These are API schemas: what a client sends and receives. They are versioned with
the URL (/v1) and change only on purpose. They are deliberately NOT the graph's
state (TeamState) and NOT the LLM schemas (llm/schemas/): those are internal and
may change any week without a client noticing.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from shopwise.api.v1.schemas.common import ThreadStatus


class ThreadCreate(BaseModel):
    """Start a conversation for one customer.

    Until session 14 adds API keys, the customer is named in the body. That is the
    limitation this version admits: anyone who can reach the API can act for anyone.
    """
    customer_email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$", max_length=254,
                                examples=["ayesha@example.com"])


class ThreadOut(BaseModel):
    id: str = Field(description="Minted by the server. Never chosen by the client.")
    customer_email: str
    title: str | None = Field(description="The first message, shortened. None until one is sent.")
    status: ThreadStatus
    created_at: datetime
    updated_at: datetime


class ThreadList(BaseModel):
    items: list[ThreadOut]
    next_cursor: str | None = Field(
        description="Pass as ?cursor= to get the next page. None on the last page.")


class HistoryMessage(BaseModel):
    role: Literal["customer", "assistant"]
    text: str


class ThreadHistory(BaseModel):
    thread_id: str
    summary: str | None = Field(
        description="Older turns, folded into one summary once the thread grew long (session 11).")
    messages: list[HistoryMessage]
