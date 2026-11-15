"""Triage's structured output: what the model decides (TriageResult) and copies out (TicketFacts)."""
from __future__ import annotations

from shopwise.llm.schemas.triage.schema import (
    TicketCategory,
    MONEY_CATEGORIES,
    BANGLA_DIGITS,
    ORDER_NUMBER,
    Ticket,
    TriageResult,
    TicketFacts,
)

__all__ = ["TicketCategory", "MONEY_CATEGORIES", "BANGLA_DIGITS", "ORDER_NUMBER", "Ticket", "TriageResult", "TicketFacts"]
