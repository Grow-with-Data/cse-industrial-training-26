"""The shapes the assistant reads and writes — project version v9.

v9 adds two, at the bottom:

    Route             workflows/supervisor/nodes.py: the supervisor's decision for
                      one step: which specialist next, with what task, and why
    Filters           kb/query.py: the metadata filters a question implies
                      (self-query), each from a closed list

v8 adds three:

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

v10: schemas.py became this package, one folder per feature (triage, inbox, rag,
memory, refund, supervisor). These are the LLM schemas: what a model is asked to
return. The HTTP schemas a client sees live in api/v1/schemas/ and are never the
same class, even when two look alike today. Everything is re-exported here, so
`from shopwise.llm.schemas import TriageResult` keeps working.
"""
from __future__ import annotations

from shopwise.llm.schemas.triage import TicketCategory, MONEY_CATEGORIES, BANGLA_DIGITS, ORDER_NUMBER, Ticket, TriageResult, TicketFacts
from shopwise.llm.schemas.inbox import SubIssue, TicketSplit
from shopwise.llm.schemas.rag import Rephrasings, Grade, Standalone, Filters
from shopwise.llm.schemas.memory import MemoryNote, MemoryNotes
from shopwise.llm.schemas.refund import ManagerDecision
from shopwise.llm.schemas.supervisor import Route

__all__ = [
    "TicketCategory",
    "MONEY_CATEGORIES",
    "BANGLA_DIGITS",
    "ORDER_NUMBER",
    "Ticket",
    "TriageResult",
    "TicketFacts",
    "SubIssue",
    "TicketSplit",
    "Rephrasings",
    "Grade",
    "Standalone",
    "Filters",
    "MemoryNote",
    "MemoryNotes",
    "ManagerDecision",
    "Route",
]
