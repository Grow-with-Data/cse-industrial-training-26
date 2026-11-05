"""The policy graph: search, optionally grade, then answer with a citation.

    from workflows.corrective_rag import build_corrective
    graph = build_corrective()                       # shape from settings
    graph = build_corrective(grade=True, fusion=True)   # or chosen per call
    graph.invoke({"question": "Can I return opened headphones after 20 days?"})["answer"]

    uv run python -m workflows.corrective_rag.graph "money back for a gift?"

THE SHAPES, ONE PER SWITCH
    search is one of three, chosen by settings.RAG_FUSION / RAG_HYDE:
        plain    retrieve                                      (session 9's, unchanged)
        fusion   expand --Send x N--> search_one -> fuse       (kb/fusion.py)
        hyde     hyde                                          (kb/fusion.py)
    then, by settings.RAG_GRADE:
        off      search -> generate -> END                     (v6's straight line)
        on       search -> grade --relevant--> generate -> END
                            \\--missed--> rewrite -> search    (at most RAG_MAX_ATTEMPTS)

`retrieve` and `generate` are session 9's functions from kb/rag.py, added as
nodes WITHOUT A CHANGE. That was the promise of their dict-in, dict-out shape:
a function that takes a dict and returns only the keys it adds is already a
LangGraph node.

WHY IT IS ITS OWN GRAPH, AND NOT NODES IN THE INBOX GRAPH
    Two reasons, both about `Send` (session 10, section 13).
    A `Send` branch hands its payload to ONE node, and only that node; these
    nodes pass question, documents and attempts between themselves. And two
    policy sub-issues can run in parallel (TCK-M02 has two): if both wrote
    `documents` in the parent graph, they would write one key in one step. In
    its own graph, each call has its own state. So the policy desk CALLS this
    graph, the way the other desks call their agents. Session 12 names the
    pattern: a subgraph.

THE REWRITE OVERWRITES `question`
    `retrieve` reads state["question"], unchanged by design, so the rewrite has
    to put its better query there, and `generate` then answers the rewritten
    question too. The policy desk keeps the customer's own words on its side,
    and compose works from those.
"""
from __future__ import annotations

import sys

from langgraph.graph import END, START, StateGraph

from config import settings
from kb import fusion as fusion_nodes
from kb.rag import generate, retrieve
from workflows.corrective_rag import nodes
from workflows.corrective_rag.state import RAGState


def _pick(value: bool | None, default: bool) -> bool:
    return default if value is None else value


def build_corrective(grade: bool | None = None, fusion: bool | None = None,
                     hyde: bool | None = None):
    """The compiled policy graph. Each argument overrides its setting
    (RAG_GRADE, RAG_FUSION, RAG_HYDE), which is how the eval scores every arm in
    one process. Fusion and HyDE are two answers to the same question, "what do
    we search with?", so they cannot both be on."""
    grade_on = _pick(grade, settings.RAG_GRADE)
    fusion_on = _pick(fusion, settings.RAG_FUSION)
    hyde_on = _pick(hyde, settings.RAG_HYDE)
    if fusion_on and hyde_on:
        raise ValueError("RAG_FUSION and RAG_HYDE are alternatives: switch one off")

    builder = StateGraph(RAGState)
    builder.add_node("generate", generate)          # kb/rag.py, unchanged

    # 1. The search: which nodes, and where it starts and finishes.
    if fusion_on:
        builder.add_node("expand", fusion_nodes.expand)
        builder.add_node("search_one", fusion_nodes.search_one)
        builder.add_node("fuse", fusion_nodes.fuse)
        builder.add_conditional_edges("expand", fusion_nodes.fan_out, ["search_one"])
        builder.add_edge("search_one", "fuse")      # the fan-in: waits for every branch
        first, last = "expand", "fuse"
    elif hyde_on:
        builder.add_node("hyde", fusion_nodes.hyde)
        first = last = "hyde"
    else:
        builder.add_node("retrieve", retrieve)      # kb/rag.py, unchanged
        first = last = "retrieve"
    builder.add_edge(START, first)

    # 2. Straight to the answer, or through the grader.
    if grade_on:
        builder.add_node("grade", nodes.grade)
        builder.add_node("rewrite", nodes.rewrite)
        builder.add_edge(last, "grade")
        builder.add_conditional_edges("grade", nodes.after_grade, ["generate", "rewrite"])
        builder.add_edge("rewrite", first)          # the loop, capped by after_grade
    else:
        builder.add_edge(last, "generate")

    builder.add_edge("generate", END)
    return builder.compile()


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Can I get my money back for a gift?"
    for arm in ({}, {"fusion": True}, {"grade": True}, {"fusion": True, "grade": True}):
        print(f"\n── {arm or 'plain'} ──")
        for update in build_corrective(**{"grade": False, "fusion": False, "hyde": False, **arm}).stream(
                {"question": question}, stream_mode="updates"):
            for node, delta in update.items():
                shown = {k: (f"{len(v)} passages" if k in ("documents", "ranked") else v)
                         for k, v in (delta or {}).items()}
                print(f"  {node:10} {shown}")
