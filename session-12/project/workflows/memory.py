"""Customer memory: what ShopWise knows about a customer across tickets. New in v8.

    open_memory()                       the Store, on disk, with a search index
    namespace(customer_id)              where one customer's memories live
    save_memory / load_memories / search_memories
                                        the three operations, as plain functions
    forget_customer(store, cid)         delete everything kept about one person
    recall                              node: this customer's memories nearest the turn
    remember                            node: what this turn taught us, written down

TWO MEMORIES, TWO PLACES
    The checkpointer remembers ONE TICKET: every step of its run, saved under its
    thread id, automatically. The Store remembers ONE CUSTOMER across all their
    tickets, and nothing goes into it unless code puts it there. Delete a thread
    and the ticket is gone; the customer's memories outlive every ticket.

THE NAMESPACE IS THE AUTHORIZATION BOUNDARY
    Every memory lives under ("customers", customer_id), and customer_id comes
    from the runtime context (middleware.ShopContext), which the caller sets
    from the login. Nothing in this file builds a namespace from anything a
    customer wrote, so no message can reach another customer's memories.

MEMORY SHAPES WORDING, NEVER DECISIONS
    `recall` hands its memories to compose and to nothing else. No desk reads
    them, so "they said they were promised a refund" can never move money.
"""
from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from functools import lru_cache

from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.store.sqlite import SqliteStore

from core.config import settings
from kb.store import embeddings
from middleware import ShopContext
from models import ask_json, build_model
from prompts.loader import load_prompts
from schemas import MemoryNotes

REMEMBER_PROMPT = load_prompts("inbox")["remember_prompt"]
RECALL_LIMIT = 3


@lru_cache
def _model():
    return build_model()


def namespace(customer_id: str) -> tuple[str, str]:
    """Where one customer's memories live. A tuple is a folder path."""
    return ("customers", customer_id)


# ── the Store ────────────────────────────────────────────────────────────────

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


# ── the three operations ─────────────────────────────────────────────────────

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


# ── the two nodes ────────────────────────────────────────────────────────────

def turn_text(state: dict) -> str:
    """What the customer means this turn: the standalone rewrite if there is one.

    `state` is the inbox graph's InboxState, typed as a dict here because
    importing workflows.inbox from this module would import the graph, which
    imports this module: a circle."""
    return state.get("standalone") or state["messages"][-1].text


def recall(state: dict, runtime: Runtime[ShopContext]) -> dict:
    """This customer's memories nearest in meaning to this turn. One embedding call."""
    if runtime.store is None or runtime.context is None:     # no Store, or Studio's context-free run
        return {"memories": []}
    return {"memories": search_memories(runtime.store, runtime.context.customer_id,
                                        turn_text(state))}


def remember(state: dict, runtime: Runtime[ShopContext]) -> dict:
    """Write down what this turn taught us about the customer, if anything.

    One structured call, after compose, so the customer never waits for it: the
    reply has already streamed. It is shown what is already known, so a
    preference is not saved twice, and it may return nothing, which is the usual
    answer: most messages are about an order, not about the person.
    """
    if runtime.store is None or runtime.context is None:
        return {"remembered": []}
    cid = runtime.context.customer_id
    known = load_memories(runtime.store, cid)
    prompt = REMEMBER_PROMPT.format(
        known="\n".join(f"- {k}" for k in known) or "(nothing yet)",
        message=next(m.text for m in reversed(state["messages"]) if m.type == "human"))
    notes, _ = ask_json(_model(), MemoryNotes, [("human", prompt)])
    saved = []
    for note in notes.memories:
        save_memory(runtime.store, cid, note.text, note.kind, source=state["ticket"]["id"])
        saved.append(note.text)
    return {"remembered": saved}
