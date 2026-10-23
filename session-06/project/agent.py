"""The ShopWise agent — project version v3. It remembers, one ticket at a time.

WHAT CHANGED FROM v2
    v2 forgot: every answer() started from an empty message list, so "and is
    that the one with the smartwatch?" arrived from a stranger. v3 adds three
    things, each one argument or one small function:

      checkpointer   create_agent(..., checkpointer=...). After every step the
                     agent's state is saved under the thread id in the call's
                     config, and the next call on that thread starts from it.
                     build_agent() takes the checkpointer as a parameter so the
                     caller decides where tickets live. In v3 that is always
                     InMemorySaver: memory that dies with the process.

      thread id      One thread is one support ticket. Not one customer, not
                     one device. new_ticket_id() makes one; run_config() puts
                     it where the checkpointer looks. A different id is a
                     different ticket, and the agent has never met you.

      summarization  The whole ticket is resent on every model call, so a long
                     ticket costs more per turn and eventually overflows the
                     context window. SummarizationMiddleware replaces the oldest
                     messages with one summary once the ticket passes
                     settings.SUMMARY_TRIGGER_TOKENS, using summary_prompt from
                     prompts.yml. It never separates a tool call from its result,
                     which a naive slice does (the notebook, section 10.1).

    Streaming is not here. It's a property of how a caller reads a run
    (agent.stream_events instead of agent.invoke), not of the agent, so it
    lives in chat.py.

WHAT IT STILL CANNOT DO
    Restart the program and every ticket is gone (session 11). It will read out
    another customer's order if asked (session 5, section 11.4), it copies
    customer details into every trace, and issue_refund is still withheld.
    Session 7 starts on all three.

    uv run python agent.py        # two turns on one ticket, then a new ticket
"""

from __future__ import annotations

import sys
import uuid

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.tracers.langchain import wait_for_all_tracers
from langgraph.checkpoint.memory import InMemorySaver

from config import KEY_NAMES, settings
from tools import TOOLS
from triage import PROMPTS
from utils import build_model, make_model

# create_agent allows 9,999 steps unless you say otherwise, which is no budget at all.
# In v2 one lookup cost two steps (the model asks, the tools run) and 12 allowed five.
# v3's summarization check is a step of its own before every model call, so a lookup
# now costs three and the answer two: 18 still allows five lookups and the answer.
# Every middleware hook you add spends this budget. Session 7 replaces it with
# call-limit middleware.
MAX_STEPS = 18


def build_summarizer() -> SummarizationMiddleware:
    """Bound a ticket's history: past the budget, the oldest messages become one summary.

    The summary is written by the plain cheap model. Compression is not reasoning,
    and SummarizationMiddleware needs a chat model, not build_model()'s fallback
    wrapper.
    """
    return SummarizationMiddleware(
        model=make_model(settings.model_id),
        trigger=("tokens", settings.SUMMARY_TRIGGER_TOKENS),
        keep=("messages", settings.SUMMARY_KEEP_MESSAGES),
        summary_prompt=PROMPTS["summary_prompt"],
    )


def build_agent(checkpointer=None):
    """The v3 agent: v2's model, tools and prompt, plus memory and a bounded history.

    Pass InMemorySaver() for memory. None gives you v2's forgetful agent back,
    which is useful as a control and nothing else.
    """
    return create_agent(
        model=build_model(),
        tools=TOOLS,
        system_prompt=PROMPTS["agent_prompt"],
        middleware=[build_summarizer()],
        checkpointer=checkpointer,
    )


def new_ticket_id() -> str:
    """A fresh thread id, shaped like a helpdesk ticket number.

    Random rather than sequential or built from the customer's email: an id
    someone can guess is a way to open someone else's ticket.
    """
    return f"TICKET-{uuid.uuid4().hex[:8].upper()}"


def run_config(thread_id: str) -> dict:
    """The config= for one turn. configurable.thread_id is the only key the checkpointer reads."""
    return {"configurable": {"thread_id": thread_id},
            "recursion_limit": MAX_STEPS, "run_name": "agent"}


def answer(agent, question: str, thread_id: str) -> dict:
    """One customer turn on one ticket. Returns the ticket's full state after the turn.

    Only the new message is sent. The checkpointer supplies the rest of the
    ticket, and that is the whole difference from v2.
    """
    return agent.invoke({"messages": [{"role": "user", "content": question}]},
                        config=run_config(thread_id))


def transcript(state: dict) -> str:
    """A ticket as a readable conversation: who asked for what, and what came back.

    v3 adds one line type, SUMMARY, for the message SummarizationMiddleware
    writes. It arrives as a HumanMessage, but the customer didn't write it.
    """
    lines = []
    for message in state["messages"]:
        if message.additional_kwargs.get("lc_source") == "summarization":
            lines.append(f"SUMMARY    {message.text}")
        elif message.type == "human":
            lines.append(f"CUSTOMER   {message.text}")
        elif message.type == "tool":
            lines.append(f"  RESULT   {message.name} -> {message.content}")
        elif message.tool_calls:
            for call in message.tool_calls:
                lines.append(f"  ASKS     {call['name']}({call['args']})")
        else:
            lines.append(f"ASSISTANT  {message.text}")
    return "\n".join(lines)


# Session 5's ending: the second question only makes sense if the first is remembered.
DEMO = [
    "I'm tanvir@example.com. Where is my order ORD-5003?",
    "And is that the one with the smartwatch in it?",
]


def main() -> int:
    if not settings.has_key:
        print(f"No {KEY_NAMES[settings.PROVIDER]} found. Put it in .env and try again.")
        return 1
    agent = build_agent(InMemorySaver())
    ticket = new_ticket_id()
    print(f"{settings.APP_NAME} v{settings.VERSION} · {settings.model_id} · "
          f"{len(TOOLS)} tools · ticket {ticket}\n")
    for question in DEMO:
        state = answer(agent, question, ticket)
    print(transcript(state))

    stranger = new_ticket_id()
    print(f"\n--- the same follow-up on a new ticket, {stranger} ---")
    print(transcript(answer(agent, DEMO[1], stranger)))
    print("\nFor the interactive, streaming version:  uv run python chat.py")
    wait_for_all_tracers()      # send the last traces before the script exits
    return 0


if __name__ == "__main__":
    sys.exit(main())
