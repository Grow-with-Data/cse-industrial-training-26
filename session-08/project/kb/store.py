"""Opening the knowledge base: one door, used by everything that touches it.

The notebook, kb/ingest.py, and from session 9 the `search_policy` tool all open
the knowledge base through this module, so they cannot disagree about which
embedding model, which collection, or which folder. Three things live here:

    embeddings()      the embedding model named by settings.EMBED_MODEL
    open_client()     the embedded Qdrant client on settings.QDRANT_PATH
    vector_store()    LangChain's view of the collection

THE COLLECTION HAS TWO KINDS OF VECTOR
    "dense" — the embedding model's meaning-coordinates. Session 8 searches these.
    "bm25"  — a sparse keyword vector, computed on your CPU, no key. Session 8
              writes it and does not use it yet; session 9's hybrid search does.
    Both are declared when the collection is created (ensure_collection),
    because a Qdrant collection's vector layout is fixed at birth: adding the
    keyword vector later would mean rebuilding the collection.
"""
from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models

from config import settings

DENSE, SPARSE = "dense", "bm25"
BM25_MODEL = "Qdrant/bm25"

# pypdf logs a warning per page for fonts it can only partly decode (the FTC PDF
# does this). The text still comes out, so from here on it is noise.
logging.getLogger("pypdf").setLevel(logging.ERROR)


class FastEmbedEmbeddings(Embeddings):
    """The free fallback: a small model on your CPU, through fastembed.

    Five lines of adapter rather than langchain-community's version, which would
    pull in a much larger dependency tree for the same two methods."""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding
        self.model = TextEmbedding(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self.model.query_embed(text))).tolist()


def embeddings() -> Embeddings:
    """The embedding model for the knowledge base: the same one, always."""
    if settings.embed_provider == "fastembed":
        return FastEmbedEmbeddings(settings.embed_name)
    from langchain.embeddings import init_embeddings
    return init_embeddings(f"google_genai:{settings.embed_name}",
                           output_dimensionality=settings.EMBED_DIM)


def open_client(path: Path | str | None = None) -> QdrantClient:
    """Embedded Qdrant on disk. Pass ":memory:" for a throwaway experiment.

    One client per folder: close() it before another process (ingest.py) opens
    the same folder, or that process fails with "already accessed"."""
    path = str(path or settings.QDRANT_PATH)
    if path == ":memory:":
        return QdrantClient(location=":memory:")
    Path(path).mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=path)


def ensure_collection(client: QdrantClient, emb: Embeddings, name: str | None = None) -> int:
    """Create the collection if it is missing; return its dense vector size.

    The size is PROBED from a real embedding rather than written down: a
    hardcoded 768 is a number that is right until someone changes EMBED_DIM."""
    name = name or settings.collection
    size = len(emb.embed_query("probe"))
    if not client.collection_exists(name):
        client.create_collection(
            name,
            vectors_config={DENSE: models.VectorParams(size=size, distance=models.Distance.COSINE)},
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
    return size


def vector_store(client: QdrantClient, emb: Embeddings | None = None, *,
                 mode: str = "dense", name: str | None = None) -> QdrantVectorStore:
    """LangChain's view of the collection. mode: "dense", "keyword" or "hybrid".

    Searching in session 8 is "dense". Writing goes through "hybrid", so every
    chunk gets both vectors and session 9 can search either way."""
    retrieval_mode = {"dense": RetrievalMode.DENSE, "keyword": RetrievalMode.SPARSE,
                      "hybrid": RetrievalMode.HYBRID}[mode]
    return QdrantVectorStore(
        client=client, collection_name=name or settings.collection,
        embedding=None if mode == "keyword" else (emb or embeddings()),
        sparse_embedding=None if mode == "dense" else FastEmbedSparse(model_name=BM25_MODEL),
        retrieval_mode=retrieval_mode, vector_name=DENSE, sparse_vector_name=SPARSE,
    )


def where(**fields: str) -> models.Filter:
    """A payload filter on metadata: where(authority="shop"), where(source="returns-policy")."""
    return models.Filter(must=[
        models.FieldCondition(key=f"metadata.{key}", match=models.MatchValue(value=value))
        for key, value in fields.items()])
