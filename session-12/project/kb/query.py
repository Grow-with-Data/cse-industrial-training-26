"""What the question means, before anything searches for it. Two functions:

    rewrite_query()     v8: a follow-up becomes a standalone question
    extract_filters()   v9: the metadata filters a question implies (self-query)

--- v9: self-query, at the bottom ---

    from kb.query import extract_filters, policy_filter, tickets_filter
    f = extract_filters("What's the warranty on wearables?")
    # Filters(topic='warranty', authority=None, product_type='wearables')
    policy_filter(f)      # metadata.source in [warranty-policy, cfr-16-part-700]
    tickets_filter(f)     # metadata.product_type == wearables

THE BUG IT FIXES
    Every policy chunk has carried `source` and `authority` since session 8, and
    every past ticket carries `product_type`. Nothing used them: "what does the
    regulator say about late delivery?" searched every document by meaning
    alone, and ShopWise's own shipping clause outranked the law it was asked
    for. A filter is what metadata is FOR.

    langchain_classic's SelfQueryRetriever does this with a query-constructor
    chain. Here it is one structured call and two dictionaries, so every step
    is visible.

THE GUARDRAIL
    A wrong filter is worse than no filter: it hides the right passage
    completely, and nothing downstream can bring it back. So:
      - every field is a closed list (schemas.Filters); a value outside it
        fails validation, and the question is searched with NO filter;
      - a filtered search that returns nothing is searched again unfiltered
        (workflows/specialists/knowledge.py);
      - a filter only applies where the metadata exists: product_type filters
        past tickets, never policies, which carry no product_type.

--- v8 notes follow ---

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
from qdrant_client import models

from core.constants import TOPIC_SOURCES
from schemas import Filters, Standalone

STANDALONE_PROMPT = load_prompts("rag")["standalone_prompt"]
SELF_QUERY_PROMPT = load_prompts("rag")["self_query_prompt"]
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


# ── v9: self-query ───────────────────────────────────────────────────────────

NO_FILTERS = Filters()


def extract_filters(question: str) -> Filters:
    """The filters `question` implies. One structured call. Any failure, a
    value outside the closed lists included, means no filters, never an error:
    the search still runs, just unfiltered."""
    try:
        result, _ = ask_json(_model(), Filters,
                             [("human", SELF_QUERY_PROMPT.format(question=question))])
    except Exception:            # noqa: BLE001 — the fallback IS the design
        return NO_FILTERS
    return result


def _condition(key: str, values: list[str]) -> models.FieldCondition:
    match = (models.MatchValue(value=values[0]) if len(values) == 1
             else models.MatchAny(any=values))
    return models.FieldCondition(key=f"metadata.{key}", match=match)


def policy_filter(f: Filters) -> models.Filter | None:
    """The filter for the POLICY collections: topic picks the files (the shop's
    policy and the regulator document on the same subject), authority picks shop
    or law. product_type is ignored: policies carry no product_type."""
    must = []
    if f.topic:
        must.append(_condition("source", TOPIC_SOURCES[f.topic]))
    if f.authority:
        must.append(_condition("authority", [f.authority]))
    return models.Filter(must=must) if must else None


def tickets_filter(f: Filters) -> models.Filter | None:
    """The filter for the PAST TICKETS: product_type only. A ticket's category is
    triage's label, not a policy topic, so topic is not applied here."""
    if f.product_type:
        return models.Filter(must=[_condition("product_type", [f.product_type])])
    return None
