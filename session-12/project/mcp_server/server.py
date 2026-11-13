"""ShopWise's tools as an MCP server. New in v9.

    uv run python mcp_server/server.py                                  # stdio (what hosts start)
    uv run python mcp_server/server.py --transport streamable-http      # http://127.0.0.1:8001/mcp
    uv run mcp dev mcp_server/server.py                                 # the MCP Inspector (needs Node)

    claude mcp add shopwise -s project -- uv run python mcp_server/server.py

A sibling script this session; session 13 moves it into the package.

WHAT IT SERVES — THE THREE MCP PRIMITIVES
    tools       get_order, get_shipment, get_product, check_refund_status,
                search_policy: the functions in tools.py, registered UNCHANGED.
                One implementation, two clients: ShopWise's own orders
                specialist and any MCP host (Claude Code, Claude Desktop).
                recent_runs and read_trace: a ticket's LangSmith traces, so a
                host can answer "why did TCK-009 go to that desk?" from evidence.
    resources   policy://returns, policy://shipping, policy://warranty: the
                policy files, readable by URI. Data the HOST chooses to read,
                where a tool is something the MODEL chooses to call.
    prompts     draft_reply: a reusable instruction a person picks from the
                host's menu ("/mcp__shopwise__draft_reply TCK-007" in Claude Code).

WHAT IT DOES NOT SERVE
    issue_refund. Least privilege is decided HERE, at the server: a tool that
    is not registered cannot be called by any host, however it is configured.
    Every tool below is read-only and says so in its annotations. Money still
    moves only through the graph's refund flow, after a manager's interrupt().
    And the server has no idea who the customer is: it is a STAFF tool. Inside
    ShopWise, the orders specialist's authorize rail (middleware.py) still
    checks every call against the logged-in customer, wherever the tool came from.

THE ONE RULE OF A STDIO SERVER: NOTHING ELSE ON STDOUT
    Over stdio, stdout IS the protocol: every line must be a JSON-RPC message.
    One print() in a tool puts a line of plain text in the middle of the stream
    and the client cannot parse it. Log to stderr (log() below). The notebook
    shows the bug once, on purpose (section 15).

THIS SERVER IS A SEPARATE PROCESS
    It imports tools.py and shopwise_data.py and reads the same files, in its
    own memory. Two consequences, both handled below:
      - the refund ledger is re-read before every refund lookup, because a
        refund paid by ShopWise's graph after this process started is in the
        file but not in this process's copy of REFUNDS;
      - embedded Qdrant allows one process per folder, so search_policy
        borrows the knowledge base for one search and gives it back.

MCP 1.x, NOT 2.x
    FastMCP is mcp 1.x's server class; mcp 2.x renamed it MCPServer and changed
    little else for a server this size. langchain-mcp-adapters, our client,
    requires mcp<2, and one environment cannot hold both majors.
"""
from __future__ import annotations

import argparse
import inspect
import re
import sys
from pathlib import Path

# `python mcp_server/server.py` puts mcp_server/ on sys.path, not the project.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402
from mcp.types import ToolAnnotations  # noqa: E402

import shopwise_data  # noqa: E402
from core.config import settings  # noqa: E402
from tools import check_refund_status, get_order, get_product, get_shipment, search_policy  # noqa: E402

mcp = FastMCP(
    "shopwise",
    instructions="ShopWise's support records and policies, read-only. Look up orders, "
                 "shipments, products and refunds; search the policies; read the "
                 "assistant's own traces. Nothing here changes any record.",
    port=8001,
    log_level="WARNING",      # INFO logs every request to stderr: noise in a host's log
)

READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False,
                            idempotentHint=True, openWorldHint=False)


def log(*parts) -> None:
    """Print for people: to STDERR, which hosts show in their logs. Never stdout."""
    print(*parts, file=sys.stderr, flush=True)


def serve(lc_tool, fn=None) -> None:
    """Register one tool from tools.py, unchanged, as an MCP tool.

    A LangChain @tool keeps the original function in `.func`. FastMCP builds
    the input schema from that function's type hints, and the description is
    the whole docstring, Args included: FastMCP does not parse Args lines into
    the schema the way parse_docstring=True does, so they stay in the text the
    model reads. `fn` replaces the function when the server must wrap it."""
    mcp.add_tool(fn or lc_tool.func, name=lc_tool.name,
                 description=inspect.getdoc(lc_tool.func), annotations=READ_ONLY)


# ── tools: records ───────────────────────────────────────────────────────────
serve(get_order)
serve(get_shipment)
serve(get_product)


def check_refund_status_fresh(refund_id: str) -> dict:
    # Another process (the graph) may have paid a refund since this server
    # started: read the ledger again first. load_ledger() is a replay, so
    # running it twice changes nothing.
    shopwise_data.load_ledger()
    return check_refund_status.func(refund_id)


serve(check_refund_status, check_refund_status_fresh)


# ── tools: rules ─────────────────────────────────────────────────────────────
def search_policy_borrowed(query: str) -> list[dict]:
    from kb import store
    try:
        return search_policy.func(query)
    except RuntimeError as exc:
        if "already accessed" not in str(exc):
            raise
        return [{"error": "the knowledge base is open in another ShopWise process right "
                          "now. Try again in a moment."}]
    finally:
        store.close_shared()        # give the folder back for the next process


serve(search_policy, search_policy_borrowed)


# ── tools: the assistant's own traces (LangSmith) ────────────────────────────
TICKET_ID = re.compile(r"^[A-Z0-9-]{3,40}$")


def brief(value, limit: int = 600) -> str:
    """A trace field, shortened: a model reading a trace needs the gist, not
    every token of every prompt."""
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[:limit] + " …"


@mcp.tool(annotations=READ_ONLY)
def recent_runs(ticket_id: str, limit: int = 5) -> list[dict]:
    """List the latest runs of the ShopWise assistant on one ticket, newest first.

    Use it to find the run_id of a ticket's turn, then read_trace(run_id) to see
    what happened inside it. Each run has run_id, start_time, status (success or
    error) and its total tokens. Needs LangSmith tracing to have been on when the
    ticket ran; an empty list means no traced run of that ticket was found.

    Args:
        ticket_id: A ticket or chat thread id like "TCK-009" or "CHAT-1A2B3C".
    """
    ticket_id = ticket_id.strip().upper()
    if not TICKET_ID.match(ticket_id):
        # The id is pasted into a LangSmith filter string below. Anything but
        # letters, digits and dashes could change the filter's meaning: the
        # same reason SQL takes parameters.
        return [{"error": "ticket ids are letters, digits and dashes, like TCK-009"}]
    from langsmith import Client
    runs = Client().list_runs(
        project_name=settings.LANGSMITH_PROJECT, is_root=True, limit=max(1, min(limit, 20)),
        filter=f'and(eq(metadata_key, "thread_id"), eq(metadata_value, "{ticket_id}"))')
    return [{"run_id": str(r.id), "name": r.name, "start_time": str(r.start_time),
             "status": r.status, "total_tokens": r.total_tokens} for r in runs]


@mcp.tool(annotations=READ_ONLY)
def read_trace(run_id: str) -> dict:
    """Read what happened inside one run of the ShopWise assistant.

    Returns the run's input and output, and every step directly inside it, in
    order: each step's name (intake, triage, split, supervisor, orders, knowledge,
    compose...), a shortened input and output, and any error. The supervisor's
    output shows its route_log: which specialist it chose and why. Get run_id from
    recent_runs.

    Args:
        run_id: A LangSmith run id, as recent_runs returns it.
    """
    from langsmith import Client
    run = Client().read_run(run_id, load_child_runs=True)
    steps = sorted(run.child_runs or [], key=lambda r: r.start_time)
    return {"name": run.name, "status": run.status, "error": run.error,
            "input": brief(run.inputs), "output": brief(run.outputs),
            "steps": [{"name": s.name, "input": brief(s.inputs, 300),
                       "output": brief(s.outputs, 500), "error": s.error} for s in steps]}


# ── resources: the policy files, by URI ──────────────────────────────────────
POLICIES = {"returns": "returns-policy", "shipping": "shipping-policy", "warranty": "warranty-policy"}


@mcp.resource("policy://{name}", mime_type="text/markdown")
def policy(name: str) -> str:
    """One of ShopWise's policies as written: returns, shipping or warranty."""
    if name not in POLICIES:
        raise ValueError(f"no policy {name!r}: try {', '.join(POLICIES)}")
    return (settings.POLICIES_DIR / f"{POLICIES[name]}.md").read_text(encoding="utf-8")


# ── prompts: reusable instructions, picked by a person ───────────────────────
@mcp.prompt()
def draft_reply(ticket_id: str) -> str:
    """Draft a reply to one ShopWise ticket, using the tools to check every fact."""
    tickets = yaml.safe_load((settings.DATA_DIR / "tickets.yml").read_text(encoding="utf-8"))
    ticket = next((t for t in tickets if t["id"] == ticket_id.strip().upper()), None)
    if ticket is None:
        return f"There is no ticket {ticket_id}. Ask which ticket they mean."
    return (f"Draft a reply to this ShopWise support ticket.\n\n"
            f"Subject: {ticket['subject']}\n{ticket['body']}\n\n"
            "Check every order, shipment and refund the ticket names with the shopwise "
            "tools before you state anything about it, and search_policy for every rule. "
            "Cite each rule's label in square brackets. Do not promise refunds: a manager "
            "decides those. Keep it to a short, warm paragraph.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()
    mcp.settings.port = args.port
    if args.transport != "stdio":
        log(f"ShopWise MCP server on http://127.0.0.1:{args.port}/mcp")
    mcp.run(transport=args.transport)
