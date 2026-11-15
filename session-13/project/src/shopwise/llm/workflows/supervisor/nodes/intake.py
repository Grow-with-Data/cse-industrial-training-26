"""intake: v8's intake, plus the routing keys a new turn must clear."""
from __future__ import annotations

from shopwise.llm.workflows.inbox.nodes import intake as v8_intake
from shopwise.llm.workflows.supervisor.state import RESET, TeamState


def intake(state: TeamState) -> dict:
    """v8's intake, plus a clean slate for routing."""
    return v8_intake(state) | {"refunds": RESET, "route_log": RESET, "hops": 0,
                               "task": None, "handoff": None}
