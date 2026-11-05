"""2-step RAG: retrieve, then generate — two functions with a seam between them.

    from kb.rag import answer_with_rag
    answer_with_rag("Can I return opened headphones after 20 days?")["answer"]

    uv run python kb/rag.py "What's your price-match policy?"

THE SHAPE IS THE POINT
    retrieve(state) -> {"documents": [...]}
    generate(state) -> {"answer": "..."}

Each takes a dict and returns only the keys it adds. That is exactly the shape
of a LangGraph node, and session 10 wires these two functions into a graph
WITHOUT rewriting them, then puts a grading loop between them (ADR-0009). Rename
them, merge them, or make them return something else, and that stops being true.

TWO-STEP MEANS IT ALWAYS RETRIEVES
    Every question pays for one search and gets one answer: predictable and
    cheap, right for an FAQ widget. The agent (agent.py) does the other thing:
    it has search_policy as a tool and decides for itself when to search.

THE CONTRACT LIVES IN prompts/rag.yml (rag_prompt); THIS FILE KEEPS ITS PROMISES
    format_context() puts a [source · section] label on every passage, which is
    the only reason a citation can be checked. REFUSED and LABEL read the two
    things the contract fixes, the refusal sentence and the label shape, so
    evals/run_eval.py can score answers without a judge.
"""
from __future__ import annotations

import re
import sys
from functools import lru_cache
from pathlib import Path

# `python kb/rag.py` puts kb/ on sys.path, not the project folder above it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document  # noqa: E402
from langchain_qdrant import QdrantVectorStore  # noqa: E402

from config import settings  # noqa: E402
from kb import store  # noqa: E402
from models import build_model, stamp  # noqa: E402
from prompts.loader import load_prompts  # noqa: E402

PROMPTS = load_prompts("rag")
RAG_PROMPT = PROMPTS["rag_prompt"]

# The contract's refusal, and how to recognise one. Detection is a PATTERN, not
# the exact sentence: a model that writes "do not" for "don't" has still refused,
# and an eval that scored it as answering would be the eval's bug, not the model's.
REFUSAL = "Our policy documents don't cover that. A member of the ShopWise team will follow up."
REFUSED = re.compile(r"policy documents (?:don't|do not|doesn't|does not) cover", re.IGNORECASE)
LABEL = re.compile(r"\[([a-z0-9-]+) · ([^\]]+)\]")


def label(doc: Document) -> str:
    """The citation label for a chunk: [source · section]."""
    return f"[{doc.metadata['source']} · {doc.metadata['section']}]"


# ── retrieve ────────────────────────────────────────────────────────────────
@lru_cache
def _reranker():
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    return TextCrossEncoder(settings.RERANK_MODEL)


def rerank(question: str, docs: list[Document], keep: int) -> list[Document]:
    """Re-order `docs` by a cross-encoder's score for (question, chunk); keep `keep`.

    An embedding compares two vectors made separately; a cross-encoder reads the
    question and the chunk TOGETHER, so it notices whether the chunk answers
    rather than merely resembles. Too slow to run over a whole corpus, fast
    enough over twenty candidates. Recall is unchanged: only the order, and so
    which chunks survive the cut."""
    scores = list(_reranker().rerank(question, [d.page_content for d in docs]))
    ranked = sorted(zip(scores, docs), key=lambda pair: pair[0], reverse=True)
    return [d for _, d in ranked[:keep]]


def search(question: str, k: int | None = None, mode: str | None = None,
           rerank_on: bool | None = None, vs: QdrantVectorStore | None = None) -> list[Document]:
    """The top `k` chunks for `question`. Every argument defaults to settings.

    `vs` lets an experiment search a different collection (the notebook's
    200-character one) through the same code. Everything else passes None."""
    k = k or settings.RAG_K
    mode = mode or settings.RETRIEVAL_MODE
    rerank_on = settings.RERANK if rerank_on is None else rerank_on
    if vs is None:
        emb = None if mode == "keyword" else store.embeddings()
        vs = store.vector_store(store.shared_client(), emb, mode=mode)
    if rerank_on:
        return rerank(question, vs.similarity_search(question, k=settings.RERANK_FETCH), k)
    return vs.similarity_search(question, k=k)


def retrieve(state: dict) -> dict:
    """The node: state["question"] in, {"documents": [...]} out.

    Optional state keys, for experiments: "k", "mode", "rerank"."""
    return {"documents": search(state["question"], state.get("k"), state.get("mode"),
                                state.get("rerank"))}


# ── generate ────────────────────────────────────────────────────────────────
def format_context(docs: list[Document]) -> str:
    """Each passage under its label and authority: what the contract cites.

    Whitespace is collapsed because markdown line-wraps cost tokens and carry
    no meaning."""
    return "\n\n".join(
        f"{label(d)} (authority: {d.metadata['authority']})\n{' '.join(d.page_content.split())}"
        for d in docs)


@lru_cache
def _model():
    return build_model()


def generate(state: dict) -> dict:
    """The node: an answer to state["question"] from state["documents"] only.

    The run is stamped with rag_prompt's version and a hash of the template,
    so a score in LangSmith can always be traced to the wording behind it."""
    prompt = RAG_PROMPT.format(context=format_context(state["documents"]),
                               question=state["question"])
    config = {"run_name": "generate", **stamp("rag_prompt", RAG_PROMPT)}
    return {"answer": _model().invoke(prompt, config=config).text}


def answer_with_rag(question: str, **options) -> dict:
    """retrieve, then generate. Returns the whole state: question, documents, answer."""
    state = {"question": question, **options}
    state |= retrieve(state)
    state |= generate(state)
    return state


# ── reading an answer, for the eval ─────────────────────────────────────────
def refused(answer: str) -> bool:
    """Did the answer use the contract's refusal? Curly apostrophes count too."""
    return bool(REFUSED.search(answer.replace("’", "'")))


def cited_sources(answer: str) -> set[str]:
    """The policy files an answer cites, read from its [source · section] labels."""
    return {source for source, _ in LABEL.findall(answer)}


def citations(answer: str) -> set[tuple[str, str]]:
    """Every (source, section) pair an answer cites."""
    return {(source, section.strip()) for source, section in LABEL.findall(answer)}


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Can I return opened headphones after 20 days?"
    result = answer_with_rag(question)
    print(f"Q: {question}\n")
    for d in result["documents"]:
        print(f"  retrieved {label(d)}")
    print(f"\n{result['answer']}")
