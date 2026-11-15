"""split: v8's split, skipped when the turn has one clear issue (hybrid routing)."""
from __future__ import annotations

from shopwise.core.config import settings
from shopwise.llm.workflows.inbox.nodes import split as v8_split
from shopwise.llm.workflows.supervisor.state import TeamState


def split(state: TeamState, routing: str | None = None) -> dict:
    """v8's split, which the code route reads. Pure supervisor routing does not
    need it, so it costs nothing there."""
    if (routing or settings.ROUTING) == "supervisor":
        return {"sub_issues": []}
    return v8_split(state)
