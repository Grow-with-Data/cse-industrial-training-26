"""The ShopWise agent — project version v6. It looks up facts AND rules, and cites the rules.

v8: chat.py no longer runs this agent: it runs the inbox graph, through
workflows/events.stream_ticket(). This file stays, unchanged apart from its
imports (models.py, prompts/chat.yml), because evals/run_eval.py --target agent
still scores it, and because it is the black box the graph opened: section 4 of
session 11's notebook compares the two.

WHAT CHANGED FROM v4 (v5 changed nothing here)
    One tool and one prompt section. tools.search_policy searches the knowledge
    base and returns each passage with a label; agent_prompt's "Policy
    questions" section says: search before answering any rule, cite the label,
    and refuse in a fixed sentence when the passages do not cover it. Nothing
    routes a question to the right tool family: the model reads eight docstrings
    and decides, which is what makes this AGENTIC RAG rather than kb/rag.py's
    2-step, where retrieval always happens.

    Every turn is now stamped with agent_prompt's version and a hash of its
    text (triage.stamp; models.stamp from v8), so a trace says which wording produced it.

WHAT IT STILL CANNOT DO
    "Look the order up before applying the return rule" is a sentence in a
    prompt. Nothing enforces it, and a ticket with four issues still gets one
    context and one tool budget. Session 10 moves the process into code.

--- v4 notes follow ---

WHAT CHANGED FROM v3
    Two arguments to create_agent, and one to every call:

      middleware=build_middleware(...)   the rails in middleware.py: content
                                         filter, PII rules, authorization,
                                         retry, call limits, a guard for blank
                                         replies, and a manager's approval for
                                         every refund. v3's summarizer is one
                                         entry in the same list.
      context_schema=Customer            who is logged in. Passed on EVERY call
                                         as context=, never read from the
                                         conversation.

    issue_refund is in TOOLS now. When the model asks for it, the run PAUSES and
    the result carries "__interrupt__". pending_refunds() reads what is waiting,
    and decide() resumes the ticket with the manager's approve or reject.

    MAX_STEPS is gone. It counted graph steps, and every middleware hook is a
    step, so seven rails would have spent it on bookkeeping. The call limits in
    middleware.py count what actually costs money: model calls and tool calls.

    uv run python agent.py        # facts, rules, a refusal, and a refund that waits
"""

from __future__ import annotations

import sys
import uuid

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.tracers.langchain import wait_for_all_tracers
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from config import KEY_NAMES, settings
from middleware import Customer, build_middleware, customer_for
from tools import TOOLS
from models import build_model, make_model, stamp
from prompts.loader import load_prompts

PROMPTS = load_prompts("chat")      # v8: prompts/chat.yml


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


def build_agent(checkpointer=None, stream_tool_steps: bool = True):
    """The v4 agent: v3 plus the rails, the refund tool and a logged-in customer.

    stream_tool_steps=False is for a program that streams the run, like chat.py:
    see utils.make_model.

    From v4 a checkpointer is required in practice: a refund pauses the run, and
    a paused run must be saved somewhere to be resumed. Without one, the pause
    happens and the resume raises.
    """
    return create_agent(
        model=build_model(stream_tool_steps),
        tools=TOOLS,
        system_prompt=PROMPTS["agent_prompt"],
        middleware=build_middleware(build_summarizer()),
        context_schema=Customer,
        checkpointer=checkpointer,
    )


def new_ticket_id() -> str:
    """A fresh thread id, shaped like a helpdesk ticket number.

    Random rather than sequential or built from the customer's email: an id
    someone can guess is a way to open someone else's ticket.
    """
    return f"TICKET-{uuid.uuid4().hex[:8].upper()}"


def run_config(thread_id: str) -> dict:
    """The config= for one turn. configurable.thread_id is the only key the checkpointer reads;
    the stamp (v6) labels the run with agent_prompt's version and hash."""
    return {"configurable": {"thread_id": thread_id}, "run_name": "agent",
            **stamp("agent_prompt", PROMPTS["agent_prompt"])}


def answer(agent, question: str, thread_id: str, customer: Customer) -> dict:
    """One customer turn on one ticket, as a logged-in customer.

    If the turn ends in a refund request, the result carries "__interrupt__" and
    the ticket is paused until decide() is called.
    """
    return agent.invoke({"messages": [{"role": "user", "content": question}]},
                        config=run_config(thread_id), context=customer)


def pending_refunds(agent, thread_id: str) -> list[dict]:
    """The refund requests a paused ticket is waiting on. [] if it isn't paused."""
    state = agent.get_state(run_config(thread_id))
    return [request for interrupt in state.interrupts
            for request in interrupt.value["action_requests"]]


def decision(approve: bool, reason: str = "") -> dict:
    """A manager's decision, in the shape HumanInTheLoopMiddleware resumes with.

    The rejection message is read by the MODEL, which then writes to the
    customer, so it says plainly that this is a rejection, and why. A bare "no"
    tends to reach the customer as "your refund is under review".
    """
    if approve:
        return {"type": "approve"}
    return {"type": "reject", "message": f"REJECTED by a ShopWise manager. Reason: {reason}"}


def decide(agent, thread_id: str, customer: Customer, approve: bool, reason: str = "") -> dict:
    """Resume a paused ticket with a manager's decision. Returns the state after the turn.

    context= is passed again: runtime context is not saved with the checkpoint,
    so every call, a resume included, supplies it.
    """
    return agent.invoke(Command(resume={"decisions": [decision(approve, reason)]}),
                        config=run_config(thread_id), context=customer)


def transcript(state: dict) -> str:
    """A ticket as a readable conversation: who asked for what, and what came back.

    v4 adds one line type, PAUSED, for a refund waiting on a manager.
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
    for interrupt in state.get("__interrupt__", []):
        for request in interrupt.value["action_requests"]:
            lines.append(f"  PAUSED   {request['name']}({request['args']}) waits for a manager")
    return "\n".join(lines)


DEMO = [
    ("tanvir@example.com", "Where is my order ORD-5003?"),                       # records only
    ("priya@example.com", "Can I still return the backpack from ORD-5004?"),     # a record AND a rule
    ("ayesha@example.com", "What's your price-match policy?"),                  # not in the corpus: refuse
    ("ayesha@example.com", "My ORD-5001 headphones broke. Please refund the full 129."),
]


def main() -> int:
    if not settings.has_key:
        print(f"No {KEY_NAMES[settings.PROVIDER]} found. Put it in labs/.env and try again.")
        return 1
    agent = build_agent(InMemorySaver())
    print(f"{settings.APP_NAME} v{settings.VERSION} · {settings.model_id} · {len(TOOLS)} tools\n")
    for email, question in DEMO:
        print(f"--- logged in as {email} ---")
        print(transcript(answer(agent, question, new_ticket_id(), customer_for(email))), "\n")
    print("A refund waits for a manager. To be the manager:  "
          "uv run python chat.py --as ayesha@example.com")
    wait_for_all_tracers()      # send the last traces before the script exits
    return 0


if __name__ == "__main__":
    sys.exit(main())
