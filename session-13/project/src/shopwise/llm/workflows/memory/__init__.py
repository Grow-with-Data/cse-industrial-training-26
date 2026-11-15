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

v10: workflows/memory.py became this package. store.py holds the Store and its
operations; nodes/ holds recall and remember, one per file. Both are re-exported
here, so `from shopwise.llm.workflows.memory import open_memory, recall` works.
"""
from __future__ import annotations

from shopwise.llm.workflows.memory.nodes import recall, remember
from shopwise.llm.workflows.memory.store import (
    forget_customer,
    index_config,
    load_memories,
    namespace,
    open_memory,
    save_memory,
    search_memories,
)

__all__ = [
    "forget_customer", "index_config", "load_memories", "namespace", "open_memory",
    "recall", "remember", "save_memory", "search_memories",
]
