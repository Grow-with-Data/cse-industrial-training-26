"""The orders specialist: this customer's records, through tools that arrive over MCP.

v8's orders and billing desks, merged, with two changes: its tools come from
ShopWise's MCP server (mcp_client.records_tools), and it can hand a customer
who wants their money back to refunds (supervisor/handoffs.py). The agent code
does not know which transport its tools came by; MCP_TRANSPORT=off gives it the
same tools by import.
"""
from __future__ import annotations

from functools import lru_cache

from langchain.agents import create_agent
from langgraph.runtime import Runtime
from langgraph.types import Command

from middleware import Customer, ShopContext, desk_middleware
from models import build_model
from prompts.loader import load_prompts
from workflows.inbox.nodes import who
from workflows.specialists.common import brief, report, run_agent
from workflows.specialists.mcp_client import records_tools
from workflows.supervisor.handoffs import transfer_to_refunds

PROMPTS = load_prompts("team")
ORDERS_PROMPT = PROMPTS["orders_prompt"].format(money=PROMPTS["orders_money_handoff"])


@lru_cache
def orders_agent(handoff: bool = True):
    """Built once per process: the MCP session opens here, on first use.
    handoff=False is pattern 1's orders expert: a tool of another agent has no
    graph around it to hand over in, so it reports instead."""
    money = PROMPTS["orders_money_handoff" if handoff else "orders_money_report"]
    tools = [*records_tools(), transfer_to_refunds] if handoff else records_tools()
    return create_agent(model=build_model(), tools=tools,
                        system_prompt=PROMPTS["orders_prompt"].format(money=money),
                        middleware=desk_middleware(), context_schema=Customer)


def orders(state: dict, runtime: Runtime[ShopContext]) -> Command:
    """Back to the supervisor with a finding, or (inside run_agent) to refunds
    with a handoff. Command, not a static edge: a node with an edge to the
    supervisor AND a handoff Command would go to BOTH, and the supervisor would
    finish the turn while refunds was still asking a manager."""
    customer = who(state["ticket"], runtime)
    note = run_agent("orders", orders_agent(), brief(state["task"], customer), customer,
                     "orders_prompt", ORDERS_PROMPT)
    return Command(goto="supervisor", update=report("orders", state["task"], note=note))
