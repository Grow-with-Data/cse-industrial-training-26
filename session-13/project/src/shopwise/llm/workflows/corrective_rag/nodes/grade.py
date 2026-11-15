"""grade: is what was retrieved good enough to answer from? after_grade routes on it."""
from __future__ import annotations

from shopwise.core.config import settings
from shopwise.llm.kb.rag import format_context
from shopwise.llm.schemas import Grade
from shopwise.llm.models import ask_json
from shopwise.llm.workflows.corrective_rag.state import RAGState
from shopwise.llm.workflows.corrective_rag.nodes._common import GRADE_PROMPT, _model


def grade(state: RAGState) -> dict:
    """One structured call over all the passages: do they hold the answer?

    One call per SET, not per passage. LangChain's agentic-RAG tutorial grades
    the set too; grading each passage separately would cost RAG_K calls per
    search for a verdict the router only needs once."""
    prompt = GRADE_PROMPT.format(question=state["question"],
                                 context=format_context(state["documents"]))
    verdict, _ = ask_json(_model(), Grade, [("human", prompt)])
    return {"relevant": verdict.relevant}


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
