"""Helpers every inbox node shares: the model, the stream writer, and how a ticket reads."""
from __future__ import annotations

from functools import lru_cache
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime
from shopwise.llm.middleware import Customer, ShopContext, customer_for
from shopwise.llm.models import build_model
from shopwise.llm.prompts.loader import load_prompts
from shopwise.llm.workflows.inbox.state import InboxState


PROMPTS = load_prompts("inbox")


SPLIT_PROMPT = PROMPTS["split_prompt"]


DESK_PROMPT = PROMPTS["desk_prompt"]


COMPOSE_PROMPT = PROMPTS["compose_prompt"]


SUMMARIZE_PROMPT = PROMPTS["summarize_prompt"]


@lru_cache
def _model():
    return build_model()


def emit(event: dict) -> None:
    """Tell whoever is streaming this run what is happening (v8).

    get_stream_writer() only works inside a running graph. A node called
    directly, from a notebook cell, has nobody to tell, so this does nothing."""
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return
    writer(event)


def progress(node: str, text: str) -> None:
    emit({"type": "progress", "node": node, "text": text})


def as_block(ticket: dict) -> str:
    """The ticket wrapped in tags, so the model reads it as data, not
    instructions. A chat message has no subject line, so none is shown."""
    subject = f"Subject: {ticket['subject']}\n" if ticket.get("subject") else ""
    return f'<ticket id="{ticket["id"]}">\n{subject}{ticket["body"]}\n</ticket>'


def reading(state: InboxState) -> dict:
    """The ticket as the graph should understand it this turn: with the standalone
    rewrite as its body when there is one (v8), otherwise as it arrived."""
    if state.get("standalone"):
        return {**state["ticket"], "subject": "", "body": state["standalone"]}
    return state["ticket"]


def who(ticket: dict, runtime: Runtime[ShopContext]) -> Customer:
    """Whose ticket this is: the runtime context, set by the caller from the login.
    LangGraph Studio starts runs without one, so the sender is the fallback: same
    email, same customer_id, same authorize rail. Never the conversation."""
    return runtime.context or customer_for(ticket["from"])
