"""Two ways to search better than one question, one search — new in v7.

    from kb.fusion import rrf
    rrf([list_a, list_b, list_c])            # fused, best first, top RAG_K

    uv run python kb/fusion.py "money back for a gift?"    # the queries, each list, the fusion

RAG-FUSION (Rackauckas 2024, arXiv 2402.03367) — parallel fan-out, fan-in
    expand      one structured call: FUSION_QUERIES rephrasings of the question.
                The customer's own question is always searched too, added by code.
    search_one  one search per phrasing. `fan_out` returns one Send per query, so
                every search runs in the SAME super-step, in parallel.
    fuse        reciprocal-rank fusion over the ranked lists:
                    score(d) = sum over lists of 1 / (RRF_K + rank of d in that list)
                then keep the top RAG_K. A chunk ranked well by several phrasings
                beats one ranked first by only one.

    Why ranks and not scores: each query has its own cosine scores, and a 0.71
    for one query is not comparable with a 0.71 for another. A rank means the
    same thing in every list. That is why RRF needs no normalisation.

HyDE (Gao et al. 2022, arXiv 2212.10496) — the contrast arm
    hyde        one call writes a passage that WOULD answer the question; that
                passage is embedded and searched with, instead of the question.
                A fake answer is often nearer the real answer in embedding space
                than a question is. It may be factually wrong; nobody reads it.

These are NODES, not a pipeline: each takes the state dict and returns only the
keys it adds, like kb/rag.py's retrieve. workflows/corrective_rag/graph.py wires
them in when settings.RAG_FUSION or RAG_HYDE is on. Session 10's scorecard says
whether either one earns its extra calls on ShopWise.

We do not import langchain-classic's MultiQueryRetriever: the technique is two
nodes and a formula, and building it is how you see what it costs.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

# `python kb/fusion.py` puts kb/ on sys.path, not the project folder above it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document  # noqa: E402
from langgraph.types import Send  # noqa: E402

from config import settings  # noqa: E402
from kb.rag import label, search  # noqa: E402
from schemas import Rephrasings  # noqa: E402
from models import ask_json, build_model, stamp  # noqa: E402
from prompts.loader import load_prompts  # noqa: E402

PROMPTS = load_prompts("rag")
EXPAND_PROMPT = PROMPTS["expand_prompt"]
HYDE_PROMPT = PROMPTS["hyde_prompt"]


@lru_cache
def _model():
    return build_model()


# ── the formula ──────────────────────────────────────────────────────────────
def doc_key(doc: Document) -> str:
    """The identity RRF adds scores under. The same chunk found by two queries
    is two Document objects, so object identity would never match. Qdrant's
    point id is the real identity; the label and opening text are a fallback."""
    return doc.metadata.get("_id") or f"{label(doc)} {doc.page_content[:80]}"


def rrf(ranked_lists: list[list[Document]], k: int | None = None,
        keep: int | None = None) -> list[Document]:
    """Reciprocal-rank fusion: merge ranked lists by rank alone; best first.

    Ranks start at 1. k (default settings.RRF_K, 60 as in the paper) damps the
    head of each list: with k=60, first place is worth 1/61 and tenth 1/70, so
    agreement across lists matters more than winning any single one."""
    k = k or settings.RRF_K
    keep = keep or settings.RAG_K
    scores: dict[str, float] = defaultdict(float)
    first_seen: dict[str, Document] = {}
    for docs in ranked_lists:
        for rank, doc in enumerate(docs, start=1):
            key = doc_key(doc)
            scores[key] += 1 / (k + rank)
            first_seen.setdefault(key, doc)
    best = sorted(scores, key=scores.get, reverse=True)[:keep]
    return [first_seen[key] for key in best]


# ── RAG-Fusion's nodes ───────────────────────────────────────────────────────
def expand(state: dict) -> dict:
    """The question plus FUSION_QUERIES rephrasings, in `queries`.

    The original goes first and is added by code: a model asked for "other
    phrasings" might drop it, and the customer's words are sometimes exactly
    the policy's words."""
    prompt = EXPAND_PROMPT.format(n=settings.FUSION_QUERIES, question=state["question"])
    result, _ = ask_json(_model(), Rephrasings, [("human", prompt)])
    extra = [q for q in result.queries if q.strip() and q != state["question"]]
    return {"queries": [state["question"], *extra[: settings.FUSION_QUERIES]]}


def fan_out(state: dict) -> list[Send]:
    """One Send per query. How many searches run is decided here, at run time.

    Each payload carries the attempt number, because the corrective loop can
    send the graph round again, and `ranked` (an append reducer) still holds
    the last attempt's lists. fuse() reads only the current attempt's."""
    attempt = state.get("attempts", 1)
    return [Send("search_one", {"query": q, "k": state.get("k"), "attempt": attempt})
            for q in state["queries"]]


def search_one(payload: dict) -> dict:
    """One phrasing, one search. Receives only its Send payload, not the state."""
    docs = search(payload["query"], payload.get("k"))
    return {"ranked": [{"attempt": payload["attempt"], "query": payload["query"], "documents": docs}]}


def fuse(state: dict) -> dict:
    """RRF over this attempt's lists, into `documents`: what generate reads."""
    attempt = state.get("attempts", 1)
    lists = [r["documents"] for r in state["ranked"] if r["attempt"] == attempt]
    return {"documents": rrf(lists, keep=state.get("k"))}


# ── HyDE's node ──────────────────────────────────────────────────────────────
def hyde(state: dict) -> dict:
    """Search with a hypothetical answer. The passage is kept in state so you
    can read what was searched for; nothing downstream ever shows it."""
    reply = _model().invoke(HYDE_PROMPT.format(question=state["question"]),
                            config={"run_name": "hyde", **stamp("hyde_prompt", HYDE_PROMPT)})
    passage = reply.text.strip()
    return {"hypothesis": passage, "documents": search(passage, state.get("k"))}


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Can I get my money back for a gift?"
    queries = expand({"question": question})["queries"]
    lists = []
    for q in queries:
        docs = search(q)
        lists.append(docs)
        print(f"\n{q}")
        for rank, d in enumerate(docs, start=1):
            print(f"  {rank}. {label(d)}")
    print("\nfused:")
    for d in rrf(lists):
        print(f"  {label(d)}")
