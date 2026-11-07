r"""v8's refund flow as a graph of its own: assess → approval → pay.

    START -> assess --proposal--> approval -> pay -> END
                    \--refused--> END

    from workflows.refund import build_refund
    graph = build_refund()           # compiled WITHOUT a checkpointer: see below
    out = graph.invoke({"order_id": "ORD-5001", ...})   # inside a parent node

WHY IT BECAME A GRAPH
    In v8 the approval and refund steps were nodes of the inbox graph, reading
    the inbox's findings. In v9 the refunds specialist owns them, and a team
    member should own its whole process: this graph is that process, with its
    own state (state.py), testable on its own and drawable on its own.

compile() WITH NO CHECKPOINTER INHERITS THE PARENT'S
    A paused refund must be saved, so this graph needs a checkpointer, and it
    gets the caller's: a graph compiled with checkpointer=None and invoked
    inside a node of a checkpointed graph saves into the parent's checkpoints,
    under the calling node's namespace. That is what makes the interrupt()
    inside approval pause the WHOLE run, and Command(resume=...) on the parent
    thread resume it: the parent node runs again from its first line, calls
    this graph again, and this graph resumes at approval instead of starting
    over (session 12, section 5). Give it its own checkpointer and the pause
    would be saved somewhere the parent's thread id cannot find.

`uv run python -m workflows.refund.graph` prints the topology.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from workflows.refund import nodes
from workflows.refund.state import RefundState


def build_refund():
    builder = StateGraph(RefundState)
    builder.add_node("assess", nodes.assess)
    builder.add_node("approval", nodes.approval)
    builder.add_node("pay", nodes.pay)
    builder.add_edge(START, "assess")
    builder.add_conditional_edges("assess", nodes.needs_manager, ["approval", END])
    builder.add_edge("approval", "pay")
    builder.add_edge("pay", END)
    return builder.compile()


if __name__ == "__main__":
    from workflows.inbox.graph import mermaid_text
    print(mermaid_text(build_refund()))
