"""What the supervisor graph remembers while the team works on one turn.

v8's InboxState, plus five keys for routing. The conversation keys and the
turn keys keep v8's names and meanings on purpose: v8's intake, summarize,
rewrite, triage, compose and the memory nodes run here UNCHANGED, and they read
and write exactly these keys.

THE FIVE NEW KEYS
    task        what the supervisor gave the specialist working now:
                {"question", "order_id", "position", "category"}. A specialist
                reads this, not the transcript.
    route_log   one entry per routing decision, appended: who chose (code, the
                supervisor, a handoff, the hop cap), where to, and why. Routing
                is STATE now, so it is in every checkpoint and every trace,
                and a manager can ask about it next week (read_trace).
    hops        how many specialists the supervisor has called this turn. The
                ceiling is settings.MAX_HOPS.
    handoff     a specialist's summary for the next one (supervisor/handoffs.py)
    refunds     now APPENDS: the refunds specialist may run more than once in
                a turn, and each run adds what it paid or declined.

RESET
    route_log, findings and refunds append, so a new turn must clear them, or
    turn two's supervisor would read turn one's route. intake writes RESET to
    each: v8's add_or_reset reducer, reused.
"""
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from schemas import SubIssue, TriageResult
from workflows.inbox.state import RESET, add_or_reset

__all__ = ["RESET", "TeamState"]


class TeamState(TypedDict, total=False):
    # the conversation (v8)
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str

    # the current turn (v8)
    ticket: dict
    standalone: str | None
    triage: TriageResult
    memories: list[str]
    sub_issues: list[SubIssue]                       # split's output: the code route reads it
    findings: Annotated[list[dict], add_or_reset]    # one per specialist run
    decisions: list[dict]                            # v8 intake still clears it; unused in v9
    refunds: Annotated[list[dict], add_or_reset]     # v9: appended per refunds run
    reply: str
    remembered: list[str]

    # routing (v9)
    task: dict | None
    route_log: Annotated[list[dict], add_or_reset]
    hops: int
    handoff: dict | None
