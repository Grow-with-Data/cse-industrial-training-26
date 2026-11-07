"""ShopWise's tools, fetched over MCP, callable from a synchronous graph. New in v9.

    from workflows.specialists.mcp_client import records_tools
    tools = records_tools()        # get_order, get_shipment, ... from mcp_server/server.py

WHAT THIS FILE IS FOR
    The orders specialist used to import its tools from tools.py. From v9 it
    asks ShopWise's MCP server for them, like any other MCP host would (Claude
    Code is the other one). The agent code does not change: a tool is a name, a
    description, a schema and a function, wherever it came from.

    Which server, and how, is settings.MCP_TRANSPORT:
        stdio            start mcp_server/server.py as a child process and talk
                         over its stdin/stdout (the default: nothing to run first)
        streamable_http  connect to a server already running at settings.MCP_URL
        off              import tools.py directly, as v8 did

THE CLIENT IS langchain-mcp-adapters
    MultiServerMCPClient, the client for LangChain 1.3.x. It needs mcp<2 (see
    pyproject.toml); on langchain>=1.4 the same job is `langchain.mcp.MCPAdapter`.

TWO PROBLEMS THIS FILE SOLVES, BOTH NOT OBVIOUS
    1. MCP tools are ASYNC ONLY. The adapters give each tool a coroutine and no
       function, so `tool.invoke(...)` raises. ShopWise's graph runs synchronously
       (chat.py, inbox.py and the notebook all call it that way). So one
       background thread runs an asyncio event loop, and every MCP call is
       handed to it and waited for. Each tool is re-wrapped as a normal tool
       whose function does exactly that.
    2. By default the adapters open a NEW SESSION FOR EVERY CALL. Over stdio a
       session is a process: every get_order would start Python, import
       ShopWise and exit, a second or more each. So one session is opened once,
       held open by a task on that loop for the life of this process, and every
       tool uses it. close() ends it (and the server process) cleanly.

ENVIRONMENT
    The MCP SDK starts a stdio server with a MINIMAL environment (PATH, HOME and
    a few others), not yours. Our server reads SHOPWISE_* variables (the
    notebook points the refund ledger at its own file, for instance), so the
    whole environment is passed explicitly. Drop `env` and the server quietly
    reads different files from the client: the worst kind of bug, the silent kind.
"""
from __future__ import annotations

import asyncio
import atexit
import os
import sys
import threading
from functools import lru_cache
from pathlib import Path

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, StructuredTool

from core.config import settings
from core.exceptions import MCPUnavailable

PROJECT = Path(__file__).resolve().parents[2]
SERVER = PROJECT / "mcp_server" / "server.py"
SERVER_NAME = "shopwise"

# Server tools that are NOT records, and so not the orders specialist's: rules
# belong to the knowledge specialist, traces to whoever debugs. Anything else the
# server lists, the orders specialist gets: add a record tool to the server and
# it arrives here with no change to any agent (exercise 2).
NOT_RECORDS = {"search_policy", "recent_runs", "read_trace"}


def connection(transport: str | None = None) -> dict:
    """The adapters' connection config for one transport."""
    transport = transport or settings.MCP_TRANSPORT
    if transport == "stdio":
        return {"transport": "stdio", "command": sys.executable, "args": [str(SERVER)],
                "cwd": str(PROJECT), "env": dict(os.environ)}
    if transport == "streamable_http":
        return {"transport": "streamable_http", "url": settings.MCP_URL}
    raise ValueError(f"no MCP connection for transport {transport!r}")


def as_text(content) -> str:
    """An MCP tool result as plain text: what a local tool's result looks like
    to the model, and what middleware.PII_RULES knows how to scan."""
    if isinstance(content, ToolMessage):
        content = content.content
    if isinstance(content, tuple):                    # (content, artifact)
        content = content[0]
    if isinstance(content, list):
        return "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


class MCPSession:
    """One open session to the ShopWise server, owned by a background event loop."""

    def __init__(self, conn: dict, timeout: float = 60):
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain_mcp_adapters.tools import load_mcp_tools

        self.timeout = timeout
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, name="mcp-client", daemon=True).start()
        self.tools: list[BaseTool] = []
        self._ready, self._error = threading.Event(), None

        async def hold() -> None:
            # Enter and leave the session in ONE task: the stdio transport's
            # cancel scopes must be exited by the task that entered them.
            self._stop = asyncio.Event()
            try:
                client = MultiServerMCPClient({SERVER_NAME: conn})
                async with client.session(SERVER_NAME) as session:
                    self.tools = await load_mcp_tools(session, server_name=SERVER_NAME)
                    self._ready.set()
                    await self._stop.wait()
            except Exception as exc:                  # noqa: BLE001 — reported below
                self._error = exc
                self._ready.set()

        self._held = asyncio.run_coroutine_threadsafe(hold(), self.loop)
        if not self._ready.wait(timeout) or self._error is not None:
            self.close()
            raise MCPUnavailable(f"the ShopWise MCP server did not start "
                                 f"({type(self._error).__name__ if self._error else 'timeout'})")

    def run(self, coroutine):
        """Run one coroutine on the session's loop and wait for its result."""
        future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        try:
            return future.result(self.timeout)
        except MCPUnavailable:
            raise
        except Exception as exc:                      # noqa: BLE001 — transport failures
            future.cancel()
            raise MCPUnavailable(f"MCP call failed ({type(exc).__name__})") from exc

    def sync_tool(self, tool: BaseTool) -> BaseTool:
        """The same tool, callable synchronously, returning plain text."""
        def call(**kwargs) -> str:
            return as_text(self.run(tool.coroutine(**kwargs)))
        return StructuredTool(name=tool.name, description=tool.description,
                              args_schema=tool.args_schema, func=call)

    def close(self) -> None:
        """End the session (over stdio: the server process exits) and the loop."""
        if getattr(self, "_stop", None) is not None:
            self.loop.call_soon_threadsafe(self._stop.set)
            try:
                self._held.result(10)
            except Exception:                         # noqa: BLE001 — closing anyway
                pass
        self.loop.call_soon_threadsafe(self.loop.stop)


@lru_cache
def session() -> MCPSession:
    """The process's one session, opened on first use and closed at exit."""
    opened = MCPSession(connection())
    atexit.register(opened.close)
    return opened


def server_tools() -> list[BaseTool]:
    """Every tool the server lists, made callable from a synchronous graph."""
    opened = session()
    return [opened.sync_tool(t) for t in opened.tools]


def records_tools() -> list[BaseTool]:
    """The orders specialist's tools: over MCP, or from tools.py when MCP_TRANSPORT=off."""
    if settings.MCP_TRANSPORT == "off":
        from tools import check_refund_status, get_order, get_product, get_shipment
        return [get_order, get_shipment, get_product, check_refund_status]
    return [t for t in server_tools() if t.name not in NOT_RECORDS]


def close() -> None:
    """Close the session now (a notebook does this before restarting the server)."""
    if session.cache_info().currsize:
        session().close()
        session.cache_clear()
