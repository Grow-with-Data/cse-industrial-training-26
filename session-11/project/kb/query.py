"""Conversational query rewriting: a follow-up becomes a standalone question. New in v8.

    from kb.query import rewrite_query
    rewrite_query(history, "And for the watch?")
    # -> "What is the return window for a smartwatch?"

    uv run python evals/run_eval.py --followups      # does it fix them? does it break the rest?

THE BUG IT FIXES
    Once a conversation persists (v8's checkpointer), customers ask follow-ups.
    "What's the return window for headphones?" is retrieved and cited correctly.
    "And for the watch?" is embedded as written, and the nearest passages to
    "and for the watch?" are about watches in general: the warranty table, not
    the return window. It is the most common RAG bug in production, and session
    9's eval never saw it, because every gold question was asked cold.

THE FIX, AND WHERE IT RUNS
    Before anything reads the message, one cheap structured call condenses the
    last few turns plus the follow-up into a question that stands alone. In the
    inbox graph that is the `rewrite` node, at the FRONT of the turn rather than
    in front of `retrieve`: triage and split read the message too, and split
    cannot file "and for the watch?" under the right desk without knowing what
    it follows.

    It is LangChain's create_history_aware_retriever idea (in langchain_classic
    since v1), written as one function so it can be a node.

WHAT IT COSTS
    One call on settings.MODEL, on turns that have history, and none on the first
    message of a thread: there is nothing to resolve. evals/run_eval.py
    --followups measures both halves: the follow-ups it should fix, and the
    standalone questions it must not damage.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.messages import BaseMessage

from models import ask_json, build_model
from prompts.loader import load_prompts
from schemas import Standalone

STANDALONE_PROMPT = load_prompts("rag")["standalone_prompt"]
HISTORY_TURNS = 6          # the last six messages: three exchanges is plenty to resolve "it"


@lru_cache
def _model():
    return build_model()


def render_history(history: list[BaseMessage], summary: str = "") -> str:
    """The conversation as the prompt reads it: the running summary (if a long
    thread has one), then the last few messages, one line each."""
    lines = [f"(summary of earlier turns) {summary}"] if summary else []
    for message in history[-HISTORY_TURNS:]:
        who = "Customer" if message.type == "human" else "ShopWise"
        lines.append(f"{who}: {' '.join(message.text.split())}")
    return "\n".join(lines) or "(none)"


def rewrite_query(history: list[BaseMessage], message: str, summary: str = "") -> str:
    """The latest message as a standalone question. No history: returned as is,
    at no cost."""
    if not history and not summary:
        return message
    prompt = STANDALONE_PROMPT.format(history=render_history(history, summary), message=message)
    result, _ = ask_json(_model(), Standalone, [("human", prompt)])
    return result.question.strip() or message
