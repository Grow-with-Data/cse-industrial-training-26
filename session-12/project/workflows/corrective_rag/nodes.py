"""Corrective RAG's two new nodes and its router.

`retrieve` and `generate` are not here: they are session 9's functions in
kb/rag.py, used as nodes without a change. RAG-Fusion's and HyDE's nodes are in
kb/fusion.py, beside the search they improve.
"""
from __future__ import annotations

from functools import lru_cache

from core.config import settings
from kb.rag import format_context
from schemas import Grade
from models import ask_json, build_model, stamp
from prompts.loader import load_prompts
from workflows.corrective_rag.state import RAGState

PROMPTS = load_prompts("rag")
GRADE_PROMPT = PROMPTS["grade_prompt"]
REWRITE_PROMPT = PROMPTS["rewrite_prompt"]


@lru_cache
def _model():
    return build_model()


def grade(state: RAGState) -> dict:
    """One structured call over all the passages: do they hold the answer?

    One call per SET, not per passage. LangChain's agentic-RAG tutorial grades
    the set too; grading each passage separately would cost RAG_K calls per
    search for a verdict the router only needs once."""
    prompt = GRADE_PROMPT.format(question=state["question"],
                                 context=format_context(state["documents"]))
    verdict, _ = ask_json(_model(), Grade, [("human", prompt)])
    return {"relevant": verdict.relevant}


def rewrite(state: RAGState) -> dict:
    """A better search query, written into `question` (see graph.py's note).

    It also counts: every rewrite means one more search. The count lives in
    state because the router has to read it, and a router reads only state."""
    prompt = REWRITE_PROMPT.format(question=state["question"])
    reply = _model().invoke(prompt, config={"run_name": "rewrite",
                                            **stamp("rewrite_prompt", REWRITE_PROMPT)})
    return {"question": reply.text.strip(), "attempts": state.get("attempts", 1) + 1}


def after_grade(state: RAGState) -> str:
    """The router. It does no work and makes no model call: it reads two keys.

    Out of attempts and still not relevant? Generate anyway. The grounding
    contract in rag_prompt then refuses with its fixed sentence, which is the
    right answer to a question the documents do not cover, and the handoff to a
    person that sentence promises is the answer, never a guess.
    """
    if state["relevant"] or state.get("attempts", 1) >= settings.RAG_MAX_ATTEMPTS:
        return "generate"
    return "rewrite"
