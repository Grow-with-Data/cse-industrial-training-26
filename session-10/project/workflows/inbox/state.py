"""What the inbox graph remembers while it works on one ticket.

STORE RAW DATA, NOT FORMATTED TEXT
    Every key holds data: the ticket as a dict, the TriageResult object, the
    SubIssue objects, the desks' findings as dicts. None holds a prompt. Each
    node formats what it needs, when it needs it, from these. Store a formatted
    prompt and every later node has to parse it back, and the first person to
    change the wording breaks a node they have never opened.

ONE KEY HAS A REDUCER, AND WHY
    By default a node's return OVERWRITES the key. That is right for every key
    here but one. `findings` is written by every desk, several at once in the
    same step when `Send` fans a ticket out, and two writes to an ordinary key
    in one step is an error (InvalidUpdateError: "Can receive only one value per
    step"). `operator.add` tells LangGraph how to combine them: append.
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from schemas import SubIssue, TriageResult


class InboxState(TypedDict, total=False):
    """The graph's state. `total=False` because keys arrive as the nodes run:
    the caller supplies `ticket`, and every other key is written by a node."""

    ticket: dict                                    # the inbound email, as loaded from YAML
    triage: TriageResult                            # whole ticket: queue and urgency
    sub_issues: list[SubIssue]                      # split's output
    findings: Annotated[list[dict], operator.add]   # one per sub-issue, from the desks
    reply: str                                      # compose's output


class DeskTask(TypedDict):
    """What `Send` hands a desk. Not the graph state: one sub-issue and the
    ticket it came from. A Send target sees its payload and nothing else."""

    ticket: dict
    sub_issue: SubIssue
    position: int        # where the customer raised it: compose sorts by this
