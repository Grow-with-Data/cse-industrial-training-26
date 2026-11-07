"""What the inbox graph remembers while it works on one ticket.

STORE RAW DATA, NOT FORMATTED TEXT
    Every key holds data: the ticket as a dict, the TriageResult object, the
    SubIssue objects, the desks' findings as dicts. None holds a prompt. Each
    node formats what it needs, when it needs it, from these.

v8: THE STATE IS SAVED AFTER EVERY STEP
    The graph now has a checkpointer, so everything here is written to
    CHECKPOINT_DB after every super-step and read back, perhaps on Monday, by a
    different process. Every value must survive that: plain dicts, lists,
    strings, LangChain messages and the project's own Pydantic models do (graph.py
    lists the Pydantic classes a checkpoint may rebuild). An open file, a client
    object or a lambda would not.

v8: A TICKET IS NOW A CONVERSATION
    One thread is one ticket, and a ticket can have several turns: the customer
    writes, ShopWise replies, the customer writes again. `messages` keeps the
    turns, with add_messages as its reducer: a node returns new messages and they
    are appended, and a RemoveMessage deletes one (the summarize node). Every
    other key describes the CURRENT turn, and the intake node resets them when a
    turn starts.

THE TWO REDUCERS
    messages   add_messages: append, or delete by id with RemoveMessage.
    findings   append, like v7, because several desks write it in one step. And
               one addition: the intake node writes RESET to empty it, or turn
               two's compose would answer turn one's findings as well. A reducer
               decides what "writing to a key" means, and that includes clearing it.
"""
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from schemas import SubIssue, TriageResult

RESET = "__reset__"


def add_or_reset(current: list[dict] | None, update: list[dict] | str) -> list[dict]:
    """findings' reducer: append the desks' findings; RESET empties the list."""
    if update == RESET:
        return []
    return (current or []) + update


class InboxState(TypedDict, total=False):
    """The graph's state. `total=False` because keys arrive as the nodes run: the
    caller supplies `ticket` on every turn, and every other key is written by a node."""

    # the conversation (v8)
    messages: Annotated[list[AnyMessage], add_messages]   # every turn, both sides
    summary: str                                          # older turns, folded (summarize)

    # the current turn
    ticket: dict                                    # what arrived: id, from, subject, body, received
    standalone: str | None                          # v8 rewrite: the message as a standalone question
    triage: TriageResult                            # whole turn: queue and urgency
    memories: list[str]                             # v8 recall: what we already know about them
    sub_issues: list[SubIssue]                      # split's output
    findings: Annotated[list[dict], add_or_reset]   # one per sub-issue, from the desks
    decisions: list[dict]                           # v8 approval: one per refund proposal
    refunds: list[dict]                             # v8 refund: what was paid, or why not
    reply: str                                      # compose's output
    remembered: list[str]                           # v8 remember: what this turn added


class DeskTask(TypedDict):
    """What `Send` hands a desk. Not the graph state: one sub-issue and the
    ticket it came from. A Send target sees its payload and nothing else."""

    ticket: dict
    sub_issue: SubIssue
    position: int        # where the customer raised it: compose sorts by this
