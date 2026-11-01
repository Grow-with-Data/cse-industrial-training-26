"""Tests for v5's knowledge base, with no key and no network: a fake embedding model.

DeterministicFakeEmbedding turns each text into a fixed random vector (the same
text, the same vector). Its "meaning" is nonsense, so nothing here tests search
QUALITY; that is the notebook's job with the real model. What it can test is
everything around the vectors: loading, metadata, chunking, stable ids, the
collection name, and the scorer.

    uv run python test_kb.py
"""
from __future__ import annotations

import os

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from config import Settings, settings
from kb import ingest, retrieval_eval, store

FAKE = DeterministicFakeEmbedding(size=64)


def test_markdown_keeps_sections():
    docs = ingest.load_markdown(settings.POLICIES_DIR / "returns-policy.md")
    sections = [d.metadata["section"] for d in docs]
    assert sections[0] == "preamble"
    assert "1. The return window" in sections
    assert all(d.metadata["authority"] == "shop" for d in docs)


def test_pdf_is_one_document_per_page_and_regulator():
    docs = ingest.load_pdf(settings.POLICIES_DIR / "ftc-prompt-delivery.pdf")
    assert docs[0].metadata["section"] == "p.1" and docs[0].metadata["page"] == 1
    assert all(d.metadata["authority"] == "regulator" for d in docs)


def test_pdf_hyphenation_is_rejoined():
    assert ingest.clean_pdf_text("WAR -\nRANTY act") == "WARRANTY act"


def test_corpus_never_indexes_both_twins():
    sources = {d.metadata["source"]: d.metadata["format"] for d in ingest.load_corpus()}
    assert sources["returns-policy"] == "md"          # the .pdf twin is not in
    assert sources["cfr-16-part-700"] == "pdf"


def test_small_chunks_cut_the_exception_from_its_number():
    """Section 6's staircase, as a test: at 200 the clause is cut, at 800 it is whole."""
    policy = (settings.POLICIES_DIR / "returns-policy.md").read_text()
    whole = [Document(page_content=policy, metadata={"source": "returns-policy"})]

    def clause_whole(size):
        chunks = ingest.chunk(whole, size, size // 5)
        holder = next(c for c in chunks if "Unopened audio" in c.page_content)
        return "30-day window" in holder.page_content

    assert not clause_whole(200)
    assert clause_whole(800)


def test_chunk_ids_are_stable_and_unique():
    chunks = ingest.chunk(ingest.load_corpus())
    first = [ingest.chunk_id(c) for c in chunks]
    second = [ingest.chunk_id(c) for c in ingest.chunk(ingest.load_corpus())]
    assert first == second
    assert len(set(first)) == len(first)


def test_writing_twice_stores_once():
    client = store.open_client(":memory:")
    store.ensure_collection(client, FAKE, "t")
    chunks = ingest.chunk(ingest.load_corpus())
    ids = [ingest.chunk_id(c) for c in chunks]
    writer = store.vector_store(client, FAKE, name="t")
    writer.add_documents(chunks, ids=ids)
    writer.add_documents(chunks, ids=ids)
    assert client.count("t").count == len(chunks)


def test_vector_size_is_probed_not_typed():
    client = store.open_client(":memory:")
    assert store.ensure_collection(client, FAKE, "t") == 64


def test_filter_keeps_out_the_other_authority():
    client = store.open_client(":memory:")
    store.ensure_collection(client, FAKE, "t")
    chunks = ingest.chunk(ingest.load_corpus())
    vs = store.vector_store(client, FAKE, name="t")
    vs.add_documents(chunks, ids=[ingest.chunk_id(c) for c in chunks])
    hits = vs.similarity_search("how long until my order ships?", k=5,
                                filter=store.where(authority="regulator"))
    assert hits and all(h.metadata["authority"] == "regulator" for h in hits)


def test_collection_is_named_after_its_space():
    assert settings.collection == f"policies__{settings.embed_name}-{settings.EMBED_DIM}" \
        or settings.embed_provider == "fastembed"
    os.environ["SHOPWISE_EMBED_MODEL"] = "fastembed:BAAI/bge-small-en-v1.5"
    try:
        assert Settings().collection == "policies__bge-small-en-v1.5"
    finally:
        del os.environ["SHOPWISE_EMBED_MODEL"]


def test_scorer_counts_ranks():
    """A fake search that returns the gold chunk second for every question."""
    questions = retrieval_eval.gold_questions()
    golds = {q["question"]: Document(page_content=" ".join(q["evidence"]),
                                     metadata={"source": q["source"]}) for q in questions}
    decoy = Document(page_content="nothing", metadata={"source": "nowhere"})
    result = retrieval_eval.score(lambda q, k: [decoy, golds[q]][:k], k=2)
    assert result["hit_rate"] == 1.0 and result["mrr"] == 0.5
    assert retrieval_eval.score(lambda q, k: [decoy, golds[q]][:k], k=1)["hit_rate"] == 0.0


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in list(globals().items()) if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"\n{len(tests)} passed")
