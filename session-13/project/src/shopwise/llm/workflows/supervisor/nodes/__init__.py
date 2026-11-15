"""The supervisor graph's own nodes: intake, split and the supervisor itself.

Everything else in the graph is reused: v8's summarize, rewrite, triage and
compose (workflows/inbox/nodes.py), v8's recall and remember (workflows/
memory.py), and the four specialists (workflows/specialists/).

THE SUPERVISOR, IN FOUR CASES, CHEAPEST FIRST
    1. The turn was code-routed and its specialists have reported: finish.
       No model call.
    2. The code route. Split found ONE issue with a clear category (ROUTING=
       hybrid), or routing is all code (ROUTING=code): send each issue to its
       specialist by category, all at once, with Send. No model call. This is
       v7's routing, kept because the comparison said it wins on single-issue
       tickets.
    3. The hop cap: MAX_HOPS specialists already called this turn. Finish with
       what we have. A loop with no ceiling is a bill with no ceiling.
    4. The model decides: one structured call returns a Route (schemas.py), and
       the node returns Command(goto=specialist, update=...). The route and its
       reason are appended to route_log, so they are visible state.

Command(goto=...) does the routing AND the update in one return value. There
are no conditional edges out of the supervisor: the node itself says where
next, which is why graph.py passes `destinations` (for the drawing only).

v10: one node per file; this file re-exports them.
"""
from __future__ import annotations

from shopwise.llm.workflows.supervisor.nodes._common import REPLY, finish, task_for, team
from shopwise.llm.workflows.supervisor.nodes.intake import intake
from shopwise.llm.workflows.supervisor.nodes.split import split
from shopwise.llm.workflows.supervisor.nodes.supervisor import supervisor

__all__ = [
    "REPLY",
    "finish",
    "task_for",
    "team",
    "intake",
    "split",
    "supervisor",
]
