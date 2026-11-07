"""Wire the supervisor graph: v9, a team of specialists with a manager.

    START -> intake -> summarize -> rewrite -> triage -> recall -> split -> supervisor
    supervisor --Command--> orders | knowledge | refunds | market   (one, or Send to several)
    each specialist -> supervisor        (it reads what came back and decides again)
    supervisor --FINISH--> compose -> remember -> END

    from workflows.supervisor import open_team
    with open_team() as graph:                          # checkpoints + memory on disk
        stream_ticket(graph, "TCK-M05", ticket, ctx)    # workflows/events.py, unchanged

    build_team(routing="supervisor")   # the comparison builds one graph per routing mode

`uv run python -m workflows.supervisor.graph` prints the topology and writes diagram.md.

WHAT IS REUSED, AND HOW
    v8's nodes (intake, summarize, rewrite, triage, compose) and v8's memory
    nodes run unchanged; specialists are the desks' successors. Two kinds of
    composition sit side by side here, on purpose:

      knowledge   a compiled subgraph ADDED AS A NODE: it shares keys with this
                  graph (task in, findings out) and keeps the rest private
      refunds     a plain node that CALLS the refund graph: different schema,
                  so the node translates in and out

    The checkpointer passed to compile() covers both: subgraphs compiled with no
    checkpointer of their own save into this graph's checkpoints, which is why a
    refund paused three levels down still pauses this graph's thread, and
    get_state(config, subgraphs=True) can show the child's state.

v8's inbox graph (workflows/inbox/) is still here and still runs: the
comparison scores it as "v8 static router".
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import partial
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from core.config import settings
from middleware import ShopContext
from workflows.inbox.graph import mermaid_text, open_saver
from workflows.inbox.nodes import compose, rewrite, summarize, triage_node
from workflows.memory import open_memory, recall, remember
from workflows.specialists.knowledge import knowledge_graph
from workflows.specialists.orders import orders
from workflows.specialists.refunds import refunds
from workflows.supervisor import nodes
from workflows.supervisor.state import TeamState


# A specialist that can hand off returns Command(goto=...) itself, so it must
# have NO static edge: a node with an edge AND a Command goes to both (orders.py).
ROUTES_ITSELF = {"orders": ("supervisor", "refunds")}


def specialists() -> dict:
    """The specialist nodes on duty. The market analyst only with MARKET_ANALYST."""
    team = {"orders": orders, "knowledge": knowledge_graph(), "refunds": refunds}
    if settings.MARKET_ANALYST:
        from workflows.specialists.market import market
        team["market"] = market
    return team


def build_team(checkpointer=None, store=None, routing: str | None = None):
    """The compiled v9 graph. `routing` overrides settings.ROUTING for this graph:
    "hybrid", "supervisor" or "code"."""
    team = specialists()
    builder = StateGraph(TeamState, context_schema=ShopContext)
    builder.add_node("intake", nodes.intake)
    builder.add_node("summarize", summarize)
    builder.add_node("rewrite", rewrite)
    builder.add_node("triage", triage_node)
    builder.add_node("recall", recall)
    builder.add_node("split", partial(nodes.split, routing=routing))
    builder.add_node("supervisor", partial(nodes.supervisor, routing=routing),
                     destinations=(*team, nodes.REPLY))   # for the drawing: Command routes
    for name, specialist in team.items():
        if name in ROUTES_ITSELF:
            builder.add_node(name, specialist, destinations=ROUTES_ITSELF[name])
        else:
            builder.add_node(name, specialist)
            builder.add_edge(name, "supervisor")          # every specialist reports back
    builder.add_node("compose", compose)
    builder.add_node("remember", remember)

    builder.add_edge(START, "intake")
    builder.add_edge("intake", "summarize")
    builder.add_edge("summarize", "rewrite")
    builder.add_edge("rewrite", "triage")
    builder.add_edge("triage", "recall")
    builder.add_edge("recall", "split")
    builder.add_edge("split", "supervisor")
    builder.add_edge("compose", "remember")
    builder.add_edge("remember", END)
    return builder.compile(checkpointer=checkpointer, store=store)


@contextmanager
def open_team(routing: str | None = None):
    """The graph as the programs use it: checkpoints and memory on disk (v8's files)."""
    with open_saver() as saver, open_memory() as store:
        yield build_team(checkpointer=saver, store=store, routing=routing)


if __name__ == "__main__":
    mermaid = mermaid_text(build_team())
    print(mermaid)
    out = Path(__file__).with_name("diagram.md")
    out.write_text("# The v9 supervisor graph\n\nGenerated by `uv run python -m "
                   "workflows.supervisor.graph`. Do not edit by hand.\n\n```mermaid\n"
                   + mermaid + "```\n", encoding="utf-8")
    print(f"wrote {out}")
