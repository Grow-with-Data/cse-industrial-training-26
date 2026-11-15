"""intake: the turn's message joins the conversation; per-turn keys reset (v8)."""
from __future__ import annotations

from langchain.messages import HumanMessage
from shopwise.llm.workflows.inbox.state import RESET, InboxState


def intake(state: InboxState) -> dict:
    """The turn's message joins the conversation, and last turn's working keys
    are cleared: `findings` through its reducer's RESET, the others by overwrite.

    The caller sends a ticket dict on every turn: the inbox sends the email as
    it arrived, chat.py sends the customer's line wrapped as one. Turning it into
    a HumanMessage HERE, inside the graph, means both callers get the same
    conversation and neither has to know how the state is shaped."""
    ticket = state["ticket"]
    text = f"{ticket['subject']}\n\n{ticket['body']}" if ticket.get("subject") else ticket["body"]
    return {"messages": [HumanMessage(text)], "findings": RESET, "standalone": None,
            "sub_issues": [], "memories": [], "decisions": [], "refunds": [],
            "remembered": [], "reply": ""}
