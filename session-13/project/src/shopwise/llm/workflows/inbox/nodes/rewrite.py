"""rewrite: a follow-up becomes a standalone question (llm/kb/query.py) (v8)."""
from __future__ import annotations

from shopwise.core.config import settings
from shopwise.llm.kb.query import rewrite_query
from shopwise.llm.workflows.inbox.state import InboxState
from shopwise.llm.workflows.inbox.nodes._common import progress


def rewrite(state: InboxState) -> dict:
    """A follow-up becomes a standalone question, before anything reads it.

    No history, no call: the first message of a thread has nothing to resolve.
    REWRITE_FOLLOWUPS=false turns the node into a pass-through, which is how the
    notebook shows the bug it fixes."""
    history = state["messages"][:-1]
    if not settings.REWRITE_FOLLOWUPS or (not history and not state.get("summary")):
        return {}
    message = state["messages"][-1].text
    standalone = rewrite_query(history, message, state.get("summary", ""))
    if standalone != message:
        progress("rewrite", f'Read as: "{standalone}"')
    return {"standalone": standalone}
