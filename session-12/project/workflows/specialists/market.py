"""[ADV] The market analyst: a ShopWise product against the open web, with links.

On only when settings.MARKET_ANALYST is true. web_search uses Tavily when
TAVILY_API_KEY is set (free tier, made for agents), and DDGS (DuckDuckGo and
friends, no key) otherwise. Either way the prompt's three rules do the real
work: cite a URL for every price, never invent one, flag weak sources.

Opening A's question, "is your kettle overpriced?", is answered here, and the
Daily Brief's Market Watch section calls the same agent.
"""
from __future__ import annotations

import os
from functools import lru_cache

from langchain.agents import create_agent
from langgraph.runtime import Runtime
from langchain.tools import tool

from middleware import Customer, ShopContext, desk_middleware
from models import build_model
from prompts.loader import load_prompts
from tools import get_product
from workflows.inbox.nodes import who
from workflows.specialists.common import brief, report, run_agent

MARKET_PROMPT = load_prompts("team")["market_prompt"]


@tool(parse_docstring=True)
def web_search(query: str) -> list[dict]:
    """Search the web. Returns up to five results, each with title, url and a snippet.

    Args:
        query: What to search for, e.g. "electric kettle 1.7L price".
    """
    if os.getenv("TAVILY_API_KEY"):
        from langchain_tavily import TavilySearch
        hits = TavilySearch(max_results=5).invoke({"query": query}).get("results", [])
        return [{"title": h["title"], "url": h["url"], "snippet": h["content"][:300]} for h in hits]
    from ddgs import DDGS
    return [{"title": h["title"], "url": h["href"], "snippet": h["body"][:300]}
            for h in DDGS().text(query, max_results=5)]


@lru_cache
def market_agent():
    return create_agent(model=build_model(), tools=[get_product, web_search],
                        system_prompt=MARKET_PROMPT, middleware=desk_middleware(),
                        context_schema=Customer)


def market(state: dict, runtime: Runtime[ShopContext]) -> dict:
    customer = who(state["ticket"], runtime)
    note = run_agent("market", market_agent(), brief(state["task"], customer), customer,
                     "market_prompt", MARKET_PROMPT)
    return report("market", state["task"], note=note)
