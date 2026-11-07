"""The graphs LangGraph Studio opens. langgraph.json points here.

    uv run --with "langgraph-cli[inmem]" langgraph dev

Then open the Studio URL it prints
(https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024), pick a
graph, paste a ticket as input and watch it move node to node.

`--with` adds the CLI for that one command and leaves pyproject.toml alone: Studio
is a debugger you use while building, not something ShopWise runs on.

v9: `team` is the supervisor graph, and the one to open first. Studio draws the
knowledge subgraph inside it, and shows the refund graph's state when a refund
pauses. `inbox` is v8's graph, kept for comparison.

Module-level compiled graphs, because that is what langgraph.json can name.
The server brings its own checkpointer, so none is passed here (v8: and its own
store, so no Store either). Studio sends no runtime context, so the inbox's
nodes fall back to the ticket's sender (workflows/inbox/nodes.who): the
authorize rail still holds, and customer memory is skipped.

v8: a refund pauses in Studio too, and Studio's "resume" box sends the
manager's answer straight to the graph, past workflows/events.stream_ticket's
role check. Studio is a developer's tool on a developer's machine; it is not a
way in for anyone else.

Example input for `team` (or `inbox`):
    {"ticket": {"id": "TCK-M05", "from": "ayesha@example.com", "received": "2026-08-09",
                "subject": "headphones refund", "body": "Please put REF-9001 through."}}
Example resume, when it pauses: {"type": "edit", "amount": 99, "reason": "one earcup"}
"""
from workflows.corrective_rag import build_corrective
from workflows.inbox import build_inbox
from workflows.supervisor import build_team

team = build_team()
inbox = build_inbox()
policy = build_corrective()
