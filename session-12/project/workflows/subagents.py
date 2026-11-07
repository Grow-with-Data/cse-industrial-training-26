"""Pattern 1, subagents as tools: the same team, with the supervisor as ONE agent. New in v9.

    from workflows.subagents import answer_with_team
    answer_with_team("Where is ORD-5003, and how long is its warranty?", context_for(email))

ZERO NEW PRIMITIVES
    Count them: create_agent (session 5), @tool (session 5), ToolRuntime context
    (session 7). A specialist is an agent; wrapped in a @tool, it is something
    another agent can call. The supervisor here is a create_agent whose ONLY
    tools are those wrappers, plus load_skill.

THE WRAPPER'S DOCSTRING IS THE ROUTING LOGIC
    The supervisor model never sees the specialists' code or prompts. It sees
    four tool names and four docstrings, and from those alone it decides who to
    consult. Session 5's docstring lesson, promoted to org design: blur one
    docstring and the routing breaks (exercise 1).

WHY SHOPWISE SHIPS THE GRAPH INSTEAD
    This is the docs' recommended default and the fastest to build. The graph
    (workflows/supervisor/) is what you reach for when routing must be VISIBLE
    (route_log is state; here it is buried in tool-call messages), CAPPED (a hop
    limit in state; here only the call-limit middleware) and TESTABLE with a
    fake model. And when a specialist must pause: here a refund's interrupt()
    would fire inside a tool inside an agent, so the refunds expert below only
    works out a proposal and pays nothing. docs/multiagent-comparison.md
    measures both; the numbers, not taste, picked the graph.

CONTEXT REACHES THE SUBAGENTS THROUGH ToolRuntime
    The orders expert runs inside the authorize rail, which needs the logged-in
    customer. The supervisor agent was invoked with context=ShopContext, and
    each wrapper reads it from runtime.context and passes it on. Never from the
    conversation.
"""
from __future__ import annotations

from functools import lru_cache

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import HumanMessage

from core.config import settings
from kb.query import NO_FILTERS, extract_filters
from middleware import LIMITS, ShopContext, blocked_reply, content_filter
from models import build_model, stamp
from prompts.loader import load_prompts
from shopwise_data import CUSTOMERS
from workflows.inbox.nodes import order_of_refund, refund_case
from workflows.skills import load_skill
from workflows.specialists.common import brief, run_agent
from workflows.specialists.knowledge import research_question
from workflows.specialists.orders import PROMPTS as TEAM_PROMPTS
from workflows.specialists.orders import orders_agent

TEAM_AGENT_PROMPT = load_prompts("team")["team_agent_prompt"]


@tool
def ask_orders_expert(question: str, runtime: ToolRuntime[ShopContext]) -> str:
    """Consult the orders specialist about THIS customer's records: where an order
    is, its tracking and delivery dates, what was charged, the status of a refund
    already requested, and a product's price, stock and warranty length. Ask one
    question, with every order id and refund id it needs. Never use it for rules."""
    customer = runtime.context
    return run_agent("orders", orders_agent(handoff=False), brief({"question": question}, customer),
                     customer, "orders_prompt", TEAM_PROMPTS["orders_prompt"])


@tool
def ask_policy_expert(question: str) -> str:
    """Consult the policy specialist for ANY question about rules: returns,
    refunds, shipping, warranty, what the law requires, and how similar cases
    were handled before. Never answer a policy question yourself. Its answer
    carries [label] citations: keep them exactly."""
    filters = extract_filters(question) if settings.SELF_QUERY else NO_FILTERS
    return research_question(question, filters)["answer"]


@tool
def ask_refunds_expert(order_id: str, reason: str, runtime: ToolRuntime[ShopContext]) -> str:
    """Consult the refunds specialist when the customer asks for money back on a
    named order (or refund id). It works out what is due and puts it to a
    ShopWise manager. It never pays anything itself."""
    customer = runtime.context
    order_id = order_id.strip().upper() or order_of_refund(reason) or ""
    ticket = {"id": "TEAM-AGENT", "received": settings.today}
    return refund_case(order_id, ticket, customer, reason)["note"]


TEAM_TOOLS = [ask_orders_expert, ask_policy_expert, ask_refunds_expert, load_skill]


@lru_cache
def team_agent():
    """The supervisor as one agent. Content filter and limits, as every agent has."""
    return create_agent(model=build_model(), tools=TEAM_TOOLS, system_prompt=TEAM_AGENT_PROMPT,
                        middleware=[content_filter, *LIMITS, blocked_reply],
                        context_schema=ShopContext)


def answer_with_team(text: str, customer: ShopContext, tools: list | None = None) -> dict:
    """One message through the subagents-as-tools team. Returns
    {"reply", "consulted", "messages"}: `consulted` is which experts it called,
    the routing this pattern keeps only in its messages.

    `tools` swaps the toolset for one call (exercise 1 blurs a docstring)."""
    agent = team_agent() if tools is None else create_agent(
        model=build_model(), tools=tools, system_prompt=TEAM_AGENT_PROMPT,
        middleware=[content_filter, *LIMITS, blocked_reply], context_schema=ShopContext)
    name = CUSTOMERS[customer.email]["name"].split()[0]
    out = agent.invoke({"messages": [HumanMessage(f"(Customer: {name})\n{text}")]},
                       config={"run_name": "team_agent", **stamp("team_agent_prompt", TEAM_AGENT_PROMPT)},
                       context=customer)
    consulted = [c["name"] for m in out["messages"] if m.type == "ai" for c in m.tool_calls]
    return {"reply": out["messages"][-1].text, "consulted": consulted, "messages": out["messages"]}
