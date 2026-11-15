"""remember: after the reply, save anything worth knowing next time."""
from __future__ import annotations

from langgraph.runtime import Runtime
from shopwise.llm.middleware import ShopContext
from shopwise.llm.models import ask_json
from shopwise.llm.schemas import MemoryNotes
from shopwise.llm.workflows.memory.store import load_memories, save_memory
from shopwise.llm.workflows.memory.nodes._common import REMEMBER_PROMPT, _model


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
