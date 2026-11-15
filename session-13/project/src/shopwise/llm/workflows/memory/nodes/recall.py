"""recall: load what ShopWise already knows about this customer, before triage."""
from __future__ import annotations

from langgraph.runtime import Runtime
from shopwise.llm.middleware import ShopContext
from shopwise.llm.workflows.memory.store import search_memories
from shopwise.llm.workflows.memory.nodes._common import turn_text


def recall(state: dict, runtime: Runtime[ShopContext]) -> dict:
    """This customer's memories nearest in meaning to this turn. One embedding call."""
    if runtime.store is None or runtime.context is None:     # no Store, or Studio's context-free run
        return {"memories": []}
    return {"memories": search_memories(runtime.store, runtime.context.customer_id,
                                        turn_text(state))}
