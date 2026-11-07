"""The inbox graph's event vocabulary: what the outside world is told, and when. New in v8.

    from workflows.events import stream_ticket
    for event in stream_ticket(graph, "CHAT-1A2B", "Where is ORD-5003?", ctx):
        render(event)           # chat.py prints it; session 13 sends it as SSE

SEVEN EVENTS, AND NOTHING ELSE
    token      a piece of the reply, as it is written. From compose ONLY: triage
               and split stream too, and their JSON must never reach a reply bubble.
    progress   a line a node chose to say: "Checking refund eligibility for ORD-5001…"
    tool       a tool started or finished: {"name": "get_shipment", "status": "started"}
    citation   a policy passage the reply cites: [returns-policy · 1. The return window]
    interrupt  the run paused for a person; the payload is the approval form
    done       the turn finished; carries the whole reply
    error      the turn failed; a message safe to show, never the raw exception

    Every event is a Pydantic model with a `type` field. Session 13's API turns
    each one into one `data:` frame with .model_dump_json(); session 15's React
    UI switches on `type`. Neither will need to know a single thing about
    LangGraph. That is the point of this file: it is the contract between the
    graph and everything that shows it.

WHERE THE EVENTS COME FROM
    stream_events(version="v3") gives one run several projections. Two are read:
      .messages   one ChatModelStream per model call, each tagged with the node
                  that made it. Only compose's become `token` events.
      .custom     whatever nodes wrote with get_stream_writer(). Opt-in:
                  CustomTransformer has to be registered for this projection to
                  exist. Nodes write plain dicts; here they become typed events.
    run.interleave("messages", "custom") reads both in the order things happened.
    After the run, .interrupted and .interrupts say whether it stopped for a
    person, and .output is the final state.
"""
from __future__ import annotations

import re
import warnings
from collections.abc import Iterator
from typing import Annotated, Literal

from langchain_core._api import LangChainBetaWarning
from langgraph.stream.transformers import CustomTransformer
from langgraph.types import Command
from pydantic import BaseModel, Field, TypeAdapter

from core.config import settings
from middleware import ShopContext

# stream_events(version="v3") is marked beta and warns on every call. We use it on
# purpose, so that one warning is silenced and nothing else.
warnings.filterwarnings("ignore", category=LangChainBetaWarning)

REPLY_NODE = "compose"


class Token(BaseModel):
    type: Literal["token"] = "token"
    text: str


class Progress(BaseModel):
    type: Literal["progress"] = "progress"
    node: str
    text: str


class Tool(BaseModel):
    type: Literal["tool"] = "tool"
    node: str
    name: str
    status: Literal["started", "finished"]


class Citation(BaseModel):
    type: Literal["citation"] = "citation"
    label: str            # exactly as it appears in the reply, without brackets
    source: str           # the policy file it came from, e.g. returns-policy
    section: str


class Interrupt(BaseModel):
    type: Literal["interrupt"] = "interrupt"
    id: str               # the interrupt's id: two paused at once are resumed by id
    payload: dict         # the approval form the node passed to interrupt()


class Done(BaseModel):
    type: Literal["done"] = "done"
    thread_id: str
    reply: str


class Error(BaseModel):
    type: Literal["error"] = "error"
    message: str


Event = Annotated[Token | Progress | Tool | Citation | Interrupt | Done | Error,
                  Field(discriminator="type")]
EVENT = TypeAdapter(Event)      # EVENT.validate_python(a_dict) -> the right class

CITATION = re.compile(r"\[([a-z0-9-]+) · ([^\]]+)\]")


def run_config(thread_id: str) -> dict:
    """The config that names a thread. The thread id is the ticket's save slot."""
    return {"configurable": {"thread_id": thread_id}, "run_name": "inbox",
            "metadata": {"thread_id": thread_id}}


def chat_ticket(thread_id: str, text: str, ctx: ShopContext) -> dict:
    """A chat line, wrapped as a ticket: the shape the graph's input takes."""
    return {"id": thread_id, "from": ctx.email, "subject": "", "body": text,
            "received": settings.today}


def paused_on(graph, thread_id: str) -> list:
    """The interrupts a thread is waiting on. Empty: not paused."""
    snapshot = graph.get_state(run_config(thread_id))
    return [i for task in snapshot.tasks for i in task.interrupts]


def citations(reply: str) -> list[Citation]:
    """Every distinct [source · section] label in a reply, in order."""
    seen, found = set(), []
    for source, section in CITATION.findall(reply):
        label = f"{source} · {section}"
        if label not in seen:
            seen.add(label)
            found.append(Citation(label=label, source=source, section=section))
    return found


def from_writer(data) -> Progress | Tool | None:
    """A node's get_stream_writer() dict, as a typed event. Anything that is not
    one of ours is dropped: a library's custom event is not part of the contract."""
    if isinstance(data, dict) and data.get("type") in ("progress", "tool"):
        return EVENT.validate_python(data)
    return None


def stream_ticket(graph, thread_id: str, turn: str | dict | Command,
                  ctx: ShopContext) -> Iterator[Event]:
    """Run one turn of a ticket and yield what the outside world should see.

    turn is a chat line (str), a whole ticket (dict, what inbox.py sends), or a
    Command(resume=...) carrying a manager's decision. The same generator serves
    all three, so chat.py, inbox.py and session 13's API never touch the graph's
    streaming API themselves.
    """
    if isinstance(turn, Command):
        # Who may answer a paused refund is decided HERE, before the graph runs,
        # from the login. Not inside the node: see nodes.check_answer for why a
        # check there would be replayed under the wrong person.
        if ctx.role != "manager":
            yield Error(message="Only a ShopWise manager can answer a paused refund.")
            return
        payload = turn
    else:
        if paused_on(graph, thread_id):
            # A new message on a paused thread would start a new run and leave
            # the manager's question behind. Refuse it here, once, for every caller.
            yield Error(message="This ticket is waiting for a manager's decision.")
            return
        ticket = chat_ticket(thread_id, turn, ctx) if isinstance(turn, str) else turn
        payload = {"ticket": ticket}

    try:
        run = graph.stream_events(payload, run_config(thread_id), context=ctx,
                                  version="v3", transformers=[CustomTransformer])
        for name, item in run.interleave("messages", "custom"):
            if name == "custom":
                if event := from_writer(item):
                    yield event
            elif item.node == REPLY_NODE:
                for delta in item.text:
                    yield Token(text=delta)
        if run.interrupted:
            for pending in run.interrupts:
                yield Interrupt(id=pending.id, payload=pending.value)
            return
        reply = (run.output or {}).get("reply", "")
    except Exception as exc:          # noqa: BLE001 — the turn failed; say so safely
        # The exception's TYPE, never its text: a message can carry keys, hosts or
        # customer data. The full traceback is in LangSmith, under this thread id.
        yield Error(message=f"The assistant could not finish this turn ({type(exc).__name__}).")
        return
    yield from citations(reply)
    yield Done(thread_id=thread_id, reply=reply)
