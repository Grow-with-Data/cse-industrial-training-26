"""rewrite: the question, reworded for another search."""
from __future__ import annotations

from shopwise.llm.models import stamp
from shopwise.llm.workflows.corrective_rag.state import RAGState
from shopwise.llm.workflows.corrective_rag.nodes._common import REWRITE_PROMPT, _model


def rewrite(state: RAGState) -> dict:
    """A better search query, written into `question` (see graph.py's note).

    It also counts: every rewrite means one more search. The count lives in
    state because the router has to read it, and a router reads only state."""
    prompt = REWRITE_PROMPT.format(question=state["question"])
    reply = _model().invoke(prompt, config={"run_name": "rewrite",
                                            **stamp("rewrite_prompt", REWRITE_PROMPT)})
    return {"question": reply.text.strip(), "attempts": state.get("attempts", 1) + 1}
