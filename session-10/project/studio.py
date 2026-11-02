"""The graphs LangGraph Studio opens. langgraph.json points here.

    uv run --with "langgraph-cli[inmem]" langgraph dev

Then open the Studio URL it prints
(https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024), pick a
graph, paste a ticket as input and watch it move node to node.

`--with` adds the CLI for that one command and leaves pyproject.toml alone: Studio
is a debugger you use while building, not something ShopWise runs on.

Two module-level compiled graphs, because that is what langgraph.json can name.
The server brings its own checkpointer, so none is passed here. Studio sends no
runtime context, so the inbox's nodes fall back to the ticket's sender
(workflows/inbox/nodes.who): the authorize rail still holds.

Example input for `inbox`:
    {"ticket": {"id": "TCK-M01", "from": "ayesha@example.com", "received": "2026-08-09",
                "subject": "three things, sorry", "body": "..."}}
"""
from workflows.corrective_rag import build_corrective
from workflows.inbox import build_inbox

inbox = build_inbox()
policy = build_corrective()
