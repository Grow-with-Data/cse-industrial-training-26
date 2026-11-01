"""The shapes the assistant reads and writes — project version v0.2.

Three models, three jobs:

    Ticket         one email from data/tickets.yml, checked when the inbox loads
    TriageResult   what the model DECIDES about a ticket (picked from closed lists)
    TicketFacts    what the model COPIES OUT of a ticket (checked against the ticket)

`TriageResult` survives to the end of the course: session 10 makes it the graph's
triage output and session 13 the API response model. Don't rename its fields.

The model writes JSON fields in the order they are declared, so a field can only
be informed by the fields above it. Decisions first, summary last.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

# The nine hand-labelled categories in tickets.yml.
TicketCategory = Literal[
    "billing", "complaint", "order_status", "other",
    "product_question", "refund", "return", "shipping", "warranty",
]

# Categories where money moves. The assistant never settles one of these alone.
MONEY_CATEGORIES = {"billing", "refund", "return", "warranty"}

# Customers write in English and Bangla, and Bangla has its own digits.
BANGLA_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

# What an order number can look like when a customer types it: "ORD-5002",
# "ord5004", "#5002", "5002". Anything else (REF-9001, SW-88231) is not one.
ORDER_NUMBER = re.compile(r"^(ORD)?[\s\-#]*(\d+)$", re.IGNORECASE)


class Ticket(BaseModel):
    """One customer email, as stored in tickets.yml.

    The file is edited by hand, so it is data from outside like any other. A typo
    in a category or a date now fails when the inbox loads, naming the ticket.
    """

    id: str = Field(pattern=r"^TCK-\d{3}$")
    subject: str
    body: str = Field(min_length=1)
    sender: str = Field(alias="from")        # `from` is a Python keyword
    category: TicketCategory                 # the hand label we score against
    received: date                           # "2026-08-02" in the file, a real date here

    def as_block(self) -> str:
        """The ticket wrapped in tags, so the model reads it as data, not instructions."""
        return f'<ticket id="{self.id}">\nSubject: {self.subject}\n{self.body}\n</ticket>'


class TriageResult(BaseModel):
    """What the assistant decides about a ticket before it answers it.

    Every field changes what the code does next: `category` routes, `urgency`
    orders the queue, `needs_human` stops the assistant promising money, and
    `summary` is what a person reads in a list.
    """

    category: TicketCategory = Field(
        description="What the ticket is about. Pick the single best fit.")
    urgency: Literal["low", "medium", "high"] = Field(
        description="high = the customer is blocked, out of pocket, or threatening to leave.")
    sentiment: Literal["calm", "frustrated", "angry"] = Field(
        description="How the customer sounds, not how serious the problem is.")
    needs_human: bool = Field(
        description="True if money moves, or the customer asks for a person.")
    summary: str = Field(
        max_length=120,
        description="One sentence of at most 15 words, for a dashboard row. No greeting.")

    @field_validator("summary")
    @classmethod
    def tidy_summary(cls, value: str) -> str:
        """Cleaning: collapse the stray newlines and double spaces models put in strings."""
        return re.sub(r"\s+", " ", value).strip()

    @model_validator(mode="after")
    def money_needs_a_human(self) -> TriageResult:
        """Policy: if money moves, a person decides, whatever the model said.

        This only ever escalates. A rule that could switch needs_human off would
        one day switch off the one that mattered.
        """
        if self.category in MONEY_CATEGORIES:
            self.needs_human = True
        return self


class TicketFacts(BaseModel):
    """What the customer says, copied out of the ticket.

    These are claims, not facts about the order. "I've had them nine days" is what
    the customer wrote; session 5 checks it against the order record.
    """

    order_ids: list[str] = Field(
        description="Every order number (ORD-NNNN) exactly as the customer wrote it. "
                    "Empty list if there is none. Never invent one.")
    product_kind: Literal["audio", "wearable", "kitchen", "bag"] | None = Field(
        description="audio = headphones, earbuds, speakers. wearable = smartwatch. "
                    "kitchen = kettle. bag = backpack. null if no product is mentioned.")
    days_since_delivery: int | None = Field(
        ge=0,
        description="How many days the customer says they have had the item. "
                    "null if they don't say.")
    opened: bool | None = Field(
        description="True if the customer says they opened or used the item. "
                    "null if they don't say.")

    @field_validator("order_ids", mode="before")
    @classmethod
    def normalise_order_ids(cls, ids: list) -> list[str]:
        """Runs BEFORE type checking: 'ord5004', '5002' and 'ORD-৫০০৩' all become ORD-NNNN.

        A refund reference or a tracking number in this list is rejected, not
        converted: REF-9001 is not order 9001.
        """
        clean = []
        for raw in ids:
            text = str(raw).strip().translate(BANGLA_DIGITS)
            if not text:
                continue                          # an empty string means "none"
            match = ORDER_NUMBER.match(text)
            if match is None:
                raise ValueError(f"{raw!r} is not an order number")
            clean.append(f"ORD-{match.group(2)}")
        return clean

    @field_validator("order_ids")
    @classmethod
    def must_appear_in_ticket(cls, ids: list[str], info: ValidationInfo) -> list[str]:
        """Every order number must be in the ticket the model read.

        The ticket text arrives through validation context:
            TicketFacts.model_validate_json(text, context={"ticket": ticket_text})
        Without context there is nothing to check against, so the check is skipped.
        """
        ticket = (info.context or {}).get("ticket")
        if ticket is None:
            return ids
        text = ticket.translate(BANGLA_DIGITS)
        for order_id in ids:
            if order_id.removeprefix("ORD-") not in text:
                raise ValueError(f"{order_id} is not in the ticket")
        return ids
