"""The shapes the assistant reads and writes — project version v8.

v8 adds three, at the bottom:

    Standalone        kb/query.py: a follow-up rewritten as a question that
                      stands on its own
    MemoryNote(s)     workflows/memory.py: what a turn taught us about the customer
    ManagerDecision   workflows/inbox/nodes.py: a manager's answer to a paused
                      refund, validated like any other input from outside

v7 adds four, at the bottom, one per structured call the graphs make:

    SubIssue, TicketSplit   the inbox's split node: one ticket -> its issues
    Rephrasings             RAG-Fusion: one question -> several search phrasings
    Grade                   corrective RAG: do these passages hold the answer?

--- v0.2 notes follow ---

Three models, three jobs:

    Ticket         one email from labs/data/tickets.yml, checked when the inbox loads
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


# ── v7: RAG-Fusion's query expansion ─────────────────────────────────────────
class Rephrasings(BaseModel):
    """Other ways to search for the same rule (kb/fusion.py).

    The original question is always searched as well, by code: a list the model
    writes could leave it out, and the customer's own words are often the best query."""

    queries: list[str] = Field(
        min_length=1, max_length=6,
        description="Different phrasings of the question, each using the words a "
                    "store's policy document would use for it.")


# ── v7: the corrective-RAG grader's verdict ──────────────────────────────────
class Grade(BaseModel):
    """The grader's verdict on one set of passages (workflows/corrective_rag).

    Reason first: fields are written in order, so the verdict comes after the
    model has said which passage answers, or what is missing."""

    reason: str = Field(description="One short sentence: which passage answers it, or what is missing.")
    relevant: bool = Field(description="True if the passages contain the rule needed to answer.")


# ── v8: conversational query rewriting (kb/query.py) ─────────────────────────
class Standalone(BaseModel):
    """A follow-up, rewritten so it means the same thing with no conversation."""

    question: str = Field(
        description="The latest message rewritten so it can be understood without "
                    "the conversation: name the product, order or topic it refers to. "
                    "If it already stands alone, return it unchanged.")


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


# ── v8: the manager's answer to a paused refund (workflows/inbox/nodes.py) ───
class ManagerDecision(BaseModel):
    """What a manager sends to resume a paused refund: ShopWise's own design.

    This is input from outside the program, like a ticket or a .env file, so it
    is validated like one. The order-total check is not here because it needs
    the order: the billing desk applies it."""

    type: Literal["approve", "edit", "reject"]
    amount: float | None = Field(default=None, gt=0,
                                 description="edit only: the amount to refund instead.")
    reason: str = Field(default="", description="Why. Required to reject or edit; "
                                                "the customer is told it.")

    @model_validator(mode="after")
    def edits_and_rejections_say_why(self) -> ManagerDecision:
        if self.type == "edit" and self.amount is None:
            raise ValueError("an edit needs an amount")
        if self.type in ("edit", "reject") and not self.reason.strip():
            raise ValueError(f"a {self.type} needs a reason: the customer will be told it")
        return self
