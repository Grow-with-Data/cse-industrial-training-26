"""The knowledge specialist: rules and precedent, from two sources, as a subgraph.

    START -> self_query -> research -> note -> END

    self_query   one structured call: the filters the question implies (kb/query.py)
    research     an agent with TWO retriever tools whose docstrings route:
                 search_policy for rules, search_past_tickets for precedent
    note         the finding compose reads

    from workflows.specialists.knowledge import knowledge_graph, research_question
    research_question("Has a PulseFit strap snapped before?")["answer"]

A SUBGRAPH ADDED AS A NODE
    The supervisor graph adds this compiled graph with add_node("knowledge",
    knowledge_graph()). That works because the two graphs SHARE keys: `task`
    goes in, `findings` comes out, with the same names and meanings on both
    sides. input_schema and output_schema make the boundary exact: the parent
    hands over only `task`, and takes back only `findings`. Everything else
    (filters, documents, the answer) is PRIVATE: it lives in this graph's state
    and never appears in the supervisor's. The refunds specialist is the other
    case, a different schema, so it is called and translated instead.

    Why output_schema matters more than it looks: `findings` appends. Without
    an output schema the subgraph would hand back its whole state, and if the
    parent's findings had come in, they would be appended to the parent a second
    time. With input_schema = {task}, this graph's findings start empty, and
    only its own finding goes out.

FILTERS REACH THE TOOLS AS CONTEXT, NOT AS ARGUMENTS
    The model chooses WHAT to search (the query) and WHERE (which tool). It does
    not choose the filters: self_query already read them out of the question, and
    they reach the tools through ToolRuntime context, the way the customer_id
    reaches the authorize rail. A model that could pass its own filter could
    also pass a wrong one.

THE CORRECTIVE LOOP, INSIDE THE AGENT
    Session 10's corrective graph graded passages and searched again. An agent
    with retriever tools does the same thing by itself: if the passages miss, it
    searches again with other words, up to the call limit in desk_middleware().
    That is the agentic-RAG shape (docs: langgraph/agentic-rag), extended to two
    sources. workflows/corrective_rag/ stays as the policy-only path the
    scorecard still runs (`evals/run_eval.py --arms`).

[OPT] CACHE WHAT REPEATS
    With KB_CACHE_TTL > 0, `research` is cached by question and filters
    (CachePolicy): the same policy question asked fifty times a day is answered
    once. `note` is NOT cached, because it carries the turn's position, and a
    cached position would be last week's. Never cache anything customer-specific
    or anything with a side effect; this node has neither.
"""
from __future__ import annotations

import json
import operator
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Annotated, TypedDict

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.documents import Document
from langgraph.cache.memory import InMemoryCache
from langgraph.graph import END, START, StateGraph
from langgraph.types import CachePolicy

from core.config import settings
from kb.query import NO_FILTERS, extract_filters, policy_filter, tickets_filter
from kb.rag import as_passages, search
from middleware import desk_middleware
from models import build_model
from prompts.loader import load_prompts
from schemas import Filters
from workflows.inbox.nodes import progress
from workflows.specialists.common import report, run_agent

KNOWLEDGE_PROMPT = load_prompts("team")["knowledge_prompt"]


@dataclass
class Lens:
    """The knowledge agent's runtime context: what its tools know that the model
    does not choose. Only the self-query filters, for now."""
    filters: Filters = field(default_factory=Filters)


def filtered(query: str, flt, collection: str | None = None) -> list[Document]:
    """Search with the filter; if that finds nothing, search without it. The
    guardrail: a filter may narrow the results, never empty them."""
    docs = search(query, flt=flt, collection=collection)
    return docs if docs or flt is None else search(query, collection=collection)


@tool(parse_docstring=True)
def search_policy(query: str, runtime: ToolRuntime[Lens]) -> list[dict]:
    """Search ShopWise's policy documents and the regulator's documents for RULES.

    Use it for any question about what is allowed, how long, or what it costs:
    return windows and conditions, refunds, shipping options and costs, warranty
    cover and claims, and what the law requires of sellers. Returns up to four
    passages, best first, each with `content`, `label` (cite it exactly, in
    square brackets) and `authority` ("shop" is ShopWise's own policy,
    "regulator" is the law). Passages that do not answer mean the documents do
    not cover it.

    Args:
        query: What to look for, in plain words, e.g. "warranty period smartwatch".
    """
    lens = runtime.context or Lens()
    return as_passages(filtered(query, policy_filter(lens.filters)))


@tool(parse_docstring=True)
def search_past_tickets(query: str, runtime: ToolRuntime[Lens]) -> list[dict]:
    """Search how ShopWise's support team resolved similar tickets before: PRECEDENT, not rules.

    Use it when the question asks whether something has happened before, what
    was done for other customers, or how a case like this usually goes. Returns
    up to four past tickets, best first, each with `content` (what the customer
    wrote and how it was resolved) and `label`, like "similar case TCK-H01":
    mention that id in plain text, never in square brackets. A past ticket shows
    what was done once. It is never a rule and never a promise.

    Args:
        query: The situation to look for, in plain words, e.g. "smartwatch strap broke".
    """
    lens = runtime.context or Lens()
    return as_passages(filtered(query, tickets_filter(lens.filters),
                                collection=settings.tickets_collection))


@lru_cache
def knowledge_agent():
    return create_agent(model=build_model(), tools=[search_policy, search_past_tickets],
                        system_prompt=KNOWLEDGE_PROMPT, middleware=desk_middleware(),
                        context_schema=Lens)


def passages_of(tool_messages) -> list[Document]:
    """The passages the agent was shown, as Documents, in the order it saw them."""
    docs = []
    for m in tool_messages:
        try:
            rows = json.loads(m.content)
        except (TypeError, ValueError):
            continue
        for row in rows if isinstance(rows, list) else []:
            if isinstance(row, dict) and "content" in row:
                docs.append(Document(page_content=row["content"], metadata=row))
    return docs


def research_question(question: str, filters: Filters | None = None) -> dict:
    """The knowledge specialist's work on one question, outside any graph:
    {"answer", "documents", "filters"}. The node below and the scorecard both call it."""
    filters = filters or NO_FILTERS
    seen: list = []
    answer = run_agent("knowledge", knowledge_agent(), question, Lens(filters),
                       "knowledge_prompt", KNOWLEDGE_PROMPT, seen=seen)
    return {"answer": answer, "documents": passages_of(seen), "filters": filters}


# ── the subgraph ─────────────────────────────────────────────────────────────

class KnowledgeInput(TypedDict):
    task: dict                                        # shared with the parent: in


class KnowledgeOutput(TypedDict):
    findings: Annotated[list[dict], operator.add]     # shared with the parent: out


class KnowledgeState(TypedDict, total=False):
    task: dict
    findings: Annotated[list[dict], operator.add]
    filters: dict                     # private: what self_query read
    answer: str                       # private
    documents: list[Document]         # private: what the agent was shown


def self_query(state: KnowledgeState) -> dict:
    filters = extract_filters(state["task"]["question"]) if settings.SELF_QUERY else NO_FILTERS
    implied = {k: v for k, v in filters.model_dump().items() if v}
    if implied:
        progress("knowledge", f"Filtering by {implied}")
    return {"filters": filters.model_dump()}


def research(state: KnowledgeState) -> dict:
    out = research_question(state["task"]["question"], Filters(**state["filters"]))
    return {"answer": out["answer"], "documents": out["documents"]}


def note(state: KnowledgeState) -> dict:
    return report("knowledge", state["task"], note=state["answer"])


def cache_key(state: KnowledgeState) -> str:
    """The question and its filters, nothing else: two customers asking the same
    policy question share an answer; nothing about either of them is in the key."""
    return json.dumps([" ".join(state["task"]["question"].lower().split()), state["filters"]])


@lru_cache
def knowledge_graph():
    ttl = settings.KB_CACHE_TTL
    builder = StateGraph(KnowledgeState, input_schema=KnowledgeInput, output_schema=KnowledgeOutput)
    builder.add_node("self_query", self_query)
    builder.add_node("research", research,
                     cache_policy=CachePolicy(key_func=cache_key, ttl=ttl) if ttl else None)
    builder.add_node("note", note)
    builder.add_edge(START, "self_query")
    builder.add_edge("self_query", "research")
    builder.add_edge("research", "note")
    builder.add_edge("note", END)
    return builder.compile(cache=InMemoryCache() if ttl else None)
