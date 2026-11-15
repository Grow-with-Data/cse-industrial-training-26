"""Customer memory's Store and its three operations (save, load, search) plus forget."""
from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from langgraph.store.base import BaseStore
from langgraph.store.sqlite import SqliteStore
from shopwise.core.config import settings
from shopwise.llm.kb.store import embeddings

RECALL_LIMIT = 3


def namespace(customer_id: str) -> tuple[str, str]:
    """Where one customer's memories live. A tuple is a folder path."""
    return ("customers", customer_id)


def embed_dims() -> int:
    """How long the memory index's vectors are: EMBED_DIM for Gemini; for a
    fastembed model, whatever it produces (measured once, locally, free)."""
    if settings.embed_provider == "fastembed":
        return len(embeddings().embed_query("dimension probe"))
    return settings.EMBED_DIM


def index_config() -> dict:
    """Embed each memory's `text` with the knowledge base's model, so the Store
    can be searched by meaning. Every put, and every search with a query, is one
    embedding call: on Gemini's free tier that is quota."""
    return {"embed": embeddings(), "dims": embed_dims(), "fields": ["text"]}


@contextmanager
def open_memory(path=None, index: bool = True):
    """The Store on disk (settings.MEMORY_DB), searchable by meaning.

    The index needs SQLite to load an extension (sqlite-vec). The labs
    environment's Python can; the python.org installer for macOS ships a SQLite
    that cannot. index=False opens it without one: memories still persist and
    load by key, and search returns the newest instead of the nearest.

    A context manager because the Store holds an open SQLite connection, and a
    connection left open is a file another process cannot safely write.
    """
    path = path or settings.MEMORY_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
    store = SqliteStore(conn, index=index_config() if index else None)
    store.setup()
    try:
        yield store
    finally:
        conn.close()


def save_memory(store: BaseStore, customer_id: str, text: str, kind: str = "preference",
                source: str = "") -> str:
    """Write one memory. Returns its key. The value is a dict, as every Store value is."""
    key = str(uuid.uuid4())
    store.put(namespace(customer_id), key, {"text": text, "kind": kind, "source": source,
                                            "saved": settings.today})
    return key


def load_memories(store: BaseStore, customer_id: str, limit: int = 50) -> list[str]:
    """Every memory for one customer, no ranking: a search with no query."""
    return [item.value["text"] for item in store.search(namespace(customer_id), limit=limit)]


def search_memories(store: BaseStore, customer_id: str, query: str,
                    limit: int = RECALL_LIMIT) -> list[str]:
    """The customer's memories nearest in meaning to `query`, best first."""
    hits = store.search(namespace(customer_id), query=query, limit=limit)
    return [hit.value["text"] for hit in hits]


def forget_customer(store: BaseStore, customer_id: str) -> int:
    """Delete every memory kept about one customer. Returns how many went.

    The namespace makes this a loop over one folder. It does not touch the
    checkpoints: their tickets stay in CHECKPOINT_DB, thread by thread, until
    those threads are deleted too (checkpointer.delete_thread)."""
    gone = 0
    while items := store.search(namespace(customer_id), limit=100):
        for item in items:
            store.delete(namespace(customer_id), item.key)
            gone += 1
    return gone
