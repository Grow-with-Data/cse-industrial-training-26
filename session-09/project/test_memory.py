"""Tests for the memory (from v3, unchanged in v4), with no key and no model: a scripted model replays fixed replies.

    uv run python test_memory.py

These test OUR side of memory — which ticket a message is filed under, what the
model is shown, what survives a summary — not the model's judgement. The
notebook's measurements do that. A real model would make these tests slow,
billed and random; a scripted one makes them instant and exact.

Session 5's appendix E built ScriptedModel; this file reuses it. Session 13
moves tests like these into pytest, which will collect the test_* functions
below as they are.
"""

from __future__ import annotations

import os
import re
import sys

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.messages import AIMessage
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langgraph.checkpoint.memory import InMemorySaver

from agent import new_ticket_id, run_config
from tools import TOOLS

# config.py loaded .env, which may switch LangSmith tracing on. Scripted runs
# are not worth a trace, so switch it off for this process only.
os.environ["LANGSMITH_TRACING"] = "false"


class ScriptedModel(GenericFakeChatModel):
    """Replays the messages it's given, and remembers what it was shown each call."""

    seen: list = []

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, *args, **kwargs):
        self.seen.append([m.text for m in messages])
        return super()._generate(messages, *args, **kwargs)


def lookup(order_id: str, call_id: str) -> AIMessage:
    """A scripted model turn that asks for get_order."""
    return AIMessage("", tool_calls=[{"name": "get_order", "args": {"order_id": order_id},
                                      "id": call_id, "type": "tool_call"}])


def ask(agent, text: str, ticket: str) -> dict:
    return agent.invoke({"messages": [{"role": "user", "content": text}]}, config=run_config(ticket))


def test_same_ticket_remembers():
    model = ScriptedModel(messages=iter([AIMessage("Noted."), AIMessage("Yes, ORD-5003.")]), seen=[])
    agent = create_agent(model, TOOLS, checkpointer=InMemorySaver())
    ticket = new_ticket_id()
    ask(agent, "My order is ORD-5003.", ticket)
    ask(agent, "Which order did I mention?", ticket)
    # The second call was shown the first turn without anyone resending it.
    assert "My order is ORD-5003." in model.seen[1]


def test_other_ticket_is_a_stranger():
    model = ScriptedModel(messages=iter([AIMessage("Noted."), AIMessage("Which order?")]), seen=[])
    agent = create_agent(model, TOOLS, checkpointer=InMemorySaver())
    ask(agent, "My card ends 4242 and my order is ORD-5003.", new_ticket_id())
    ask(agent, "Which order did I mention?", new_ticket_id())
    # Ticket B's model call contains ticket B and nothing else: the privacy boundary.
    assert model.seen[1] == ["Which order did I mention?"]


def test_ticket_ids_are_random():
    ids = {new_ticket_id() for _ in range(1000)}
    assert len(ids) == 1000
    assert all(re.fullmatch(r"TICKET-[0-9A-F]{8}", i) for i in ids)


def test_no_thread_id_is_an_error():
    model = ScriptedModel(messages=iter([AIMessage("hi")]), seen=[])
    agent = create_agent(model, TOOLS, checkpointer=InMemorySaver())
    try:
        agent.invoke({"messages": [{"role": "user", "content": "hello"}]})
    except ValueError as err:
        assert "thread_id" in str(err)
    else:
        raise AssertionError("a checkpointed agent ran without a thread id")


def test_summary_never_splits_a_tool_call_from_its_result():
    # Four turns, each a lookup and an answer: 16 messages. A tight budget makes
    # the summarizer fire, and its cut must not leave a result without its call.
    replies = []
    for n in range(4):
        replies += [lookup("ORD-5003", f"call_{n}"), AIMessage(f"Answer {n}.")]
    model = ScriptedModel(messages=iter(replies), seen=[])
    summarizer = ScriptedModel(messages=iter([AIMessage("SUMMARY of the ticket.")] * 10), seen=[])
    agent = create_agent(model, TOOLS, checkpointer=InMemorySaver(), middleware=[
        SummarizationMiddleware(model=summarizer, trigger=("messages", 8), keep=("messages", 3))])

    ticket = new_ticket_id()
    for n in range(4):
        state = ask(agent, f"Question {n} about ORD-5003?", ticket)

    assert summarizer.seen, "the summarizer never fired"
    asked = set()
    for message in state["messages"]:
        if message.type == "ai":
            asked |= {call["id"] for call in message.tool_calls}
        if message.type == "tool":
            assert message.tool_call_id in asked, f"orphaned result {message.tool_call_id}"


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"passed  {test.__name__}")
    print(f"\n{len(tests)} passed")
    sys.exit(0)
