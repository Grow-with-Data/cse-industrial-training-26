"""The team's specialists, one module each. New in v9.

    orders.py      this customer's records, over MCP; hands refund requests over
    knowledge.py   rules and precedent: a subgraph with two retriever tools
    refunds.py     calls the refund graph (workflows/refund/) and translates
    market.py      [ADV] prices from the open web, with links
    common.py      what every specialist shares: the brief, the report, the agent loop
    mcp_client.py  the orders specialist's tools, fetched from mcp_server/server.py

EACH SPECIALIST IS SMALL ON PURPOSE
    Every one is a few lines of new code around something an earlier session
    built: session 5's tools, session 9's search, session 11's refund flow. A
    specialist that grows long is re-teaching agents; keep the line count as
    the proof of reuse.

THE CONTRACT A SPECIALIST KEEPS
    In:  state["task"]    {"question", "order_id", "position", "category"}
         state["ticket"]  the ticket (who wrote, when it arrived)
         runtime.context  who the customer is (never from the conversation)
    Out: {"findings": [one finding]}, the same shape v8's desks wrote, so v8's
         compose node writes the reply from it unchanged.
"""
