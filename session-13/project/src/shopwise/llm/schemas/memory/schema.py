"""What a turn taught ShopWise about a customer."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


# ── v8: customer memory (workflows/memory.py) ────────────────────────────────
class MemoryNote(BaseModel):
    """One lasting fact about a customer, worth knowing on their next ticket."""

    kind: Literal["preference", "fact"] = Field(
        description="preference = how they want to be treated or contacted. "
                    "fact = something lasting about them (not about one order).")
    text: str = Field(
        max_length=160,
        description="One sentence about the customer, in the third person, e.g. "
                    "'Prefers a replacement to a refund when something is faulty.'")


class MemoryNotes(BaseModel):
    """What one turn taught us. Usually nothing: most messages are about an order."""

    memories: list[MemoryNote] = Field(
        default_factory=list, max_length=3,
        description="Only what the customer said about themselves that will still be "
                    "true next month. Nothing already known. Empty is the usual answer.")
