"""Wire the inbox graph. One function compiles it; one opens it on disk.

    START -> intake -> summarize -> rewrite -> triage -> recall -> split
          --Send x N--> orders | billing | policy | general
          -> approval -> refund -> compose -> remember -> END

    from workflows.inbox import open_inbox
    with open_inbox() as graph:                         # checkpoints + memory on disk
        config = {"configurable": {"thread_id": "TCK-M05"}}
        graph.invoke({"ticket": ticket}, config, context=context_for(ticket["from"]))

    build_inbox() alone compiles it with whatever you pass: a notebook cell gives
    it an InMemorySaver, LangGraph Studio gives it nothing and brings its own.

`uv run python -m workflows.inbox.graph` prints the topology and rewrites diagram.md.

v8: IT STOPS, IT RESUMES, IT REMEMBERS
    The billing desk proposes a refund; the approval node pauses with
    interrupt() until a manager decides; the refund node pays. A paused run has
    to be SAVED to be resumed: that is the checkpointer, CHECKPOINT_DB. Every
    super-step is written there under the thread id, so the same file gives you
    a ticket's conversation across restarts, its full history, and the points
    you can replay or fork from. Customer memory lives in a second file, the
    Store (MEMORY_DB), shared across every thread.

    compile(checkpointer=...) is the same parameter create_agent took in
    session 6. create_agent builds a graph like this one and compiles it the
    same way.

THE CUSTOMER IS CONTEXT
    `context_schema=ShopContext`: whose ticket it is and who is acting, supplied
    by the caller on every invoke and resume, read by nodes through `runtime`,
    and never visible to the model.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from core.config import settings
from middleware import ShopContext
from workflows.inbox.nodes import (approval, billing_desk, compose, general_desk, intake,
                                   orders_desk, policy_desk, refund, rewrite, route_sub_issues,
                                   split, summarize, triage_node)
from workflows.inbox.state import InboxState
from workflows.memory import open_memory, recall, remember

DESK_NODES = {
    "orders": orders_desk,
    "billing": billing_desk,
    "policy": policy_desk,
    "general": general_desk,
}


def build_inbox(checkpointer=None, store=None):
    """The compiled v8 graph. Without a checkpointer it still answers a turn with
    no refund in it; a refund's interrupt() needs one to pause, and a second turn
    needs one to remember the first."""
    builder = StateGraph(InboxState, context_schema=ShopContext)
    builder.add_node("intake", intake)
    builder.add_node("summarize", summarize)
    builder.add_node("rewrite", rewrite)
    builder.add_node("triage", triage_node)
    builder.add_node("recall", recall)
    builder.add_node("split", split)
    for name, desk in DESK_NODES.items():
        builder.add_node(name, desk)
    builder.add_node("approval", approval)
    builder.add_node("refund", refund)
    builder.add_node("compose", compose)
    builder.add_node("remember", remember)

    builder.add_edge(START, "intake")
    builder.add_edge("intake", "summarize")
    builder.add_edge("summarize", "rewrite")
    builder.add_edge("rewrite", "triage")
    builder.add_edge("triage", "recall")
    builder.add_edge("recall", "split")
    builder.add_conditional_edges("split", route_sub_issues, list(DESK_NODES))
    for name in DESK_NODES:
        builder.add_edge(name, "approval")      # the fan-in
    # approval always runs: with no proposals it asks nothing and refund pays
    # nothing, so a ticket without a refund costs two empty steps, not a branch.
    builder.add_edge("approval", "refund")
    builder.add_edge("refund", "compose")
    builder.add_edge("compose", "remember")
    builder.add_edge("remember", END)
    return builder.compile(checkpointer=checkpointer, store=store)


# A checkpoint stores the project's own Pydantic models (TriageResult, SubIssue)
# as "module + class + fields". Reading one back means importing and building
# that class, so the serializer only does it for classes it has been told about:
# an allowlist. Without it, LangGraph warns on every load ("Deserializing
# unregistered type ...") and a future version refuses. Messages, dicts, lists
# and strings need no permission.
CHECKPOINT_TYPES = [("schemas", "TriageResult"), ("schemas", "SubIssue")]


def checkpoint_serde() -> JsonPlusSerializer:
    """The serializer every inbox checkpointer needs, on disk or in memory:
    `InMemorySaver(serde=checkpoint_serde())` in a notebook."""
    return JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)


@contextmanager
def open_saver(path: Path | None = None):
    """The checkpointer, on disk. Every ticket is a thread in this one file.

    check_same_thread=False because Send runs desks in worker threads, and each
    writes its result through this connection; SqliteSaver locks around it."""
    path = path or settings.CHECKPOINT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    try:
        yield SqliteSaver(conn, serde=checkpoint_serde())
    finally:
        conn.close()


@contextmanager
def open_inbox():
    """The graph as the programs use it: checkpoints and memory on disk."""
    with open_saver() as saver, open_memory() as store:
        yield build_inbox(checkpointer=saver, store=store)


def mermaid_text(graph) -> str:
    """The graph as plain mermaid that any previewer renders (v7's four repairs)."""
    text = graph.get_graph().draw_mermaid()
    if text.startswith("---"):
        text = text.split("---", 2)[2].lstrip()
    lines = ["graph TD"]
    for raw in text.splitlines():
        line = raw.strip().rstrip(";")
        if not line or line.startswith(("graph ", "classDef")):
            continue
        line = line.split(":::")[0]
        line = line.replace("<p>", "").replace("</p>", "")
        line = line.replace("__start__([__start__])", "graph_start([START])")
        line = line.replace("__end__([__end__])", "graph_end([END])")
        line = line.replace("__start__", "graph_start").replace("__end__", "graph_end")
        lines.append(f"    {line}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    mermaid = mermaid_text(build_inbox())
    print(mermaid)
    out = Path(__file__).with_name("diagram.md")
    out.write_text("# The v8 inbox graph\n\nGenerated by `uv run python -m workflows.inbox.graph`. "
                   "Do not edit by hand.\n\n```mermaid\n" + mermaid + "```\n", encoding="utf-8")
    print(f"wrote {out}")
