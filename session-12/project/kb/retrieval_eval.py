"""Score retrieval against the gold questions, before any answer is generated.

    uv run python kb/retrieval_eval.py            # k=4
    uv run python kb/retrieval_eval.py --k 1

Two numbers, both plain Python, no judge:
    hit-rate@k   the share of answerable questions whose gold chunk appears
                 anywhere in the top k. A gold chunk comes from the expected
                 source and holds EVERY evidence phrase, so a chunk that kept
                 "15 days" but lost "Opened audio" does not count.
    MRR          mean reciprocal rank: 1 if the gold chunk came first, 1/2 if
                 second, ... 0 if it never appeared. Rewards ranking it HIGH.

The unanswerable questions in data/kb_questions.yml are skipped here: there is
no right chunk to find. They are session 9's, where the measure is whether the
assistant refuses.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

from core.config import settings  # noqa: E402

Search = Callable[[str, int], list[Document]]


def gold_questions() -> list[dict]:
    rows = yaml.safe_load((settings.DATA_DIR / "kb_questions.yml").read_text(encoding="utf-8"))
    return [q for q in rows if q["answerable"]]


def is_gold(q: dict, doc: Document) -> bool:
    """From the expected source, and holding every evidence phrase.

    Whitespace is collapsed first: the markdown wraps lines, so "free return
    shipping" can arrive as "free return\\nshipping"."""
    text = " ".join(doc.page_content.split()).lower()
    return (doc.metadata.get("source") == q["source"]
            and all(e.lower() in text for e in q["evidence"]))


def rank_of(q: dict, hits: list[Document]) -> int | None:
    """1-based position of the first gold hit, or None."""
    return next((i for i, d in enumerate(hits, 1) if is_gold(q, d)), None)


def score(search: Search, k: int = 4) -> dict:
    """Run every answerable gold question through `search(question, k)`."""
    ranks = {q["id"]: rank_of(q, search(q["question"], k)) for q in gold_questions()}
    n = len(ranks)
    return {"k": k,
            "hit_rate": sum(r is not None for r in ranks.values()) / n,
            "mrr": sum(1 / r for r in ranks.values() if r) / n,
            "ranks": ranks}


if __name__ == "__main__":
    from kb import store
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()

    client = store.open_client()
    try:
        vs = store.vector_store(client)
        result = score(lambda q, k: vs.similarity_search(q, k=k), args.k)
    finally:
        client.close()
    print(f"{settings.collection} · k={args.k}")
    print(f"hit-rate@{args.k} {result['hit_rate']:.2f}   MRR {result['mrr']:.2f}")
    for qid, r in result["ranks"].items():
        print(f"  {qid}  {'miss' if r is None else f'rank {r}'}")
