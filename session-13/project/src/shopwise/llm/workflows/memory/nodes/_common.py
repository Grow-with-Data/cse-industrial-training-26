"""What the two memory nodes share: the remember prompt, the model, the turn as text."""
from __future__ import annotations

from functools import lru_cache
from shopwise.llm.models import build_model
from shopwise.llm.prompts.loader import load_prompts


REMEMBER_PROMPT = load_prompts("inbox")["remember_prompt"]




@lru_cache
def _model():
    return build_model()


def turn_text(state: dict) -> str:
    """What the customer means this turn: the standalone rewrite if there is one.

    `state` is the inbox graph's InboxState, typed as a dict here because
    importing workflows.inbox from this module would import the graph, which
    imports this module: a circle."""
    return state.get("standalone") or state["messages"][-1].text
