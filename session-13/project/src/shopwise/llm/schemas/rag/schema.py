"""The knowledge base's structured calls: rephrasings, grades, standalone questions, filters."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


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


# ── v9: self-query (kb/query.py) ─────────────────────────────────────────────
class Filters(BaseModel):
    """The filters a question IMPLIES, read out of it before searching.

    "What's the warranty on wearables?" implies topic=warranty and
    product_type=wearables; "what does the regulator say" implies
    authority=regulator. Every field is a closed list, so a value the model
    invents fails validation instead of becoming a filter that matches nothing:
    kb/query.py then searches with no filter at all. Null means "not implied",
    which is the usual answer for most fields."""

    topic: Literal["returns", "shipping", "warranty"] | None = Field(
        default=None, description="The policy topic, only if the question names one.")
    authority: Literal["shop", "regulator"] | None = Field(
        default=None,
        description='"regulator" only if they ask what the law or a regulator says; '
                    '"shop" only if they ask for ShopWise\'s own rule as opposed to the law.')
    product_type: Literal["audio", "wearables", "home", "outdoor"] | None = Field(
        default=None,
        description="audio (headphones, earbuds, speakers), wearables (smartwatches, "
                    "fitness bands), home (kettles, kitchen appliances), outdoor "
                    "(backpacks, bottles, camping gear). Only if the question names one.")
