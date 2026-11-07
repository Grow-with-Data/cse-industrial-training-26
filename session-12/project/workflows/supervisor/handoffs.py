"""Handoffs: a specialist passes the CONVERSATION on, not just an answer. New in v9.

    transfer_to_refunds    the orders specialist's way to hand a customer who
                           wants their money back to the refunds specialist

THE MECHANISM: A TOOL THAT RETURNS A Command FOR THE PARENT GRAPH
    A tool normally returns data, which becomes a ToolMessage the agent reads.
    This one returns Command(goto="refunds", graph=Command.PARENT, update=...).
    `graph=Command.PARENT` means "not this agent's graph: the one around it".
    LangGraph stops the orders agent where it is and applies the Command to the
    supervisor graph: the update is written to the supervisor's state and the
    refunds node runs next. The orders node never returns normally.

A HANDOFF CARRIES A SUMMARY, NOT THE TRANSCRIPT
    The update holds three lines the orders agent wrote, in `handoff`, and the
    refunds specialist reads only those. Cheaper, and the receiver's context is
    clean: it never sees the tool calls, the shipment JSON or the small talk.
    Exercise 4 measures the difference.

WHY NO ToolMessage
    The LangChain handoff docs add a ToolMessage to the update, answering the
    tool call, because there the agents SHARE one `messages` list, and a list
    holding an AI tool call with no answer is invalid for the next model call.
    ShopWise's supervisor state keeps only the customer's turns and ShopWise's
    replies in `messages`, never tool calls (v8's summarize node relies on it),
    and the orders agent's own messages die with the agent. So there is no tool
    call in any list that will be read again, and nothing to answer.
"""
from __future__ import annotations

from langchain.tools import tool
from langgraph.types import Command

from shopwise_data import ORDERS


@tool(parse_docstring=True)
def transfer_to_refunds(order_id: str, summary: str) -> Command | str:
    """Hand this customer to the refunds specialist: they want their money back for an order.

    Use it instead of answering when the customer asks for a refund, or to cancel
    and be refunded, on an order you have looked up. After you call it you are
    finished: do not write anything else.

    Args:
        order_id: The order the money is for, like "ORD-5003".
        summary: Three short lines for the refunds specialist: the order and its status, what the customer wants, and why they said they want it. Not the conversation.
    """
    order_id = order_id.strip().upper()
    if order_id not in ORDERS:
        # Returned, not raised: the model can fix a mistyped id and try again.
        return f"No order {order_id}. Look the order up first, then hand over."
    return Command(
        goto="refunds", graph=Command.PARENT,
        update={"handoff": {"from": "orders", "order_id": order_id, "summary": summary},
                "route_log": [{"by": "handoff", "to": "refunds", "from": "orders",
                               "reason": summary.splitlines()[0][:160] if summary else ""}]})
