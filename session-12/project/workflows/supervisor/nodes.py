"""The supervisor graph's own nodes: intake, split and the supervisor itself.

Everything else in the graph is reused: v8's summarize, rewrite, triage and
compose (workflows/inbox/nodes.py), v8's recall and remember (workflows/
memory.py), and the four specialists (workflows/specialists/).

THE SUPERVISOR, IN FOUR CASES, CHEAPEST FIRST
    1. The turn was code-routed and its specialists have reported: finish.
       No model call.
    2. The code route. Split found ONE issue with a clear category (ROUTING=
       hybrid), or routing is all code (ROUTING=code): send each issue to its
       specialist by category, all at once, with Send. No model call. This is
       v7's routing, kept because the comparison said it wins on single-issue
       tickets.
    3. The hop cap: MAX_HOPS specialists already called this turn. Finish with
       what we have. A loop with no ceiling is a bill with no ceiling.
    4. The model decides: one structured call returns a Route (schemas.py), and
       the node returns Command(goto=specialist, update=...). The route and its
       reason are appended to route_log, so they are visible state.

Command(goto=...) does the routing AND the update in one return value. There
are no conditional edges out of the supervisor: the node itself says where
next, which is why graph.py passes `destinations` (for the drawing only).
"""
from __future__ import annotations

from functools import lru_cache

from langgraph.types import Command, Send

from core.config import settings
from core.constants import CLEAR_CATEGORIES, FINISH, SPECIALIST_FOR, SPECIALISTS
from models import ask_json, build_model
from prompts.loader import load_prompts
from schemas import Route
from workflows.inbox.nodes import as_block, progress, reading
from workflows.inbox.nodes import intake as v8_intake
from workflows.inbox.nodes import split as v8_split
from workflows.supervisor.state import RESET, TeamState

SUPERVISOR_PROMPT = load_prompts("team")["supervisor_prompt"]
REPLY = "compose"          # where every finished turn goes: v8's compose node


@lru_cache
def _model():
    return build_model()


def team() -> dict[str, str]:
    """The specialists on duty, with their remits. The market analyst is [ADV]."""
    return {name: remit for name, remit in SPECIALISTS.items()
            if name != "market" or settings.MARKET_ANALYST}


def intake(state: TeamState) -> dict:
    """v8's intake, plus a clean slate for routing."""
    return v8_intake(state) | {"refunds": RESET, "route_log": RESET, "hops": 0,
                               "task": None, "handoff": None}


def split(state: TeamState, routing: str | None = None) -> dict:
    """v8's split, which the code route reads. Pure supervisor routing does not
    need it, so it costs nothing there."""
    if (routing or settings.ROUTING) == "supervisor":
        return {"sub_issues": []}
    return v8_split(state)


def task_for(question: str, position: int, order_id: str | None = None,
             category: str | None = None) -> dict:
    return {"question": question, "order_id": order_id, "position": position, "category": category}


def done_so_far(state: TeamState) -> str:
    """What the supervisor reads about this turn's progress: each specialist
    run, in order, with what it asked and what came back."""
    lines = []
    for f in sorted(state.get("findings") or [], key=lambda f: f["position"]):
        lines.append(f"- {f['desk']} was asked: {f['question']}\n  and found: "
                     f"{' '.join(str(f['note']).split())}")
    for r in state.get("refunds") or []:
        lines.append(f"- refund outcome on {r.get('order_id', '?')}: {r.get('status', 'issued')}")
    return "\n".join(lines) or "(nothing yet)"


def finish(reason: str, by: str = "supervisor") -> Command:
    return Command(goto=REPLY, update={"route_log": [{"by": by, "to": FINISH, "reason": reason}]})


def supervisor(state: TeamState, routing: str | None = None) -> Command:
    routing = routing or settings.ROUTING
    log = state.get("route_log") or []
    subs = state.get("sub_issues") or []

    # 1. a code-routed turn is finished once its specialists report back
    if any(entry["by"] == "code" for entry in log):
        return finish("every issue went to its specialist by category", by="code")

    # 2. the code route: no model call
    clear = len(subs) == 1 and subs[0].category in CLEAR_CATEGORIES
    if not log and subs and (routing == "code" or (routing == "hybrid" and clear)):
        sends, entries = [], []
        for i, sub in enumerate(subs):
            target = SPECIALIST_FOR[sub.category]
            sends.append(Send(target, {"task": task_for(sub.question, i, sub.order_id, sub.category),
                                       "ticket": state["ticket"]}))
            entries.append({"by": "code", "to": target, "task": sub.question,
                            "reason": f"category {sub.category}"})
        progress("supervisor", "Routed by category: " + ", ".join(e["to"] for e in entries))
        return Command(goto=sends, update={"route_log": entries})

    # 3. the hop cap
    hops = state.get("hops", 0)
    if hops >= settings.MAX_HOPS:
        return finish(f"hop cap: {hops} specialists called this turn", by="cap")

    # 4. the model decides
    on_duty = team()
    prompt = SUPERVISOR_PROMPT.format(
        specialists="\n".join(f"- {name}: {remit}" for name, remit in on_duty.items()),
        ticket=as_block(reading(state)), progress=done_so_far(state),
        steps_left=settings.MAX_HOPS - hops)
    route, _ = ask_json(_model(), Route, [("human", prompt)])
    if route.next == FINISH:
        return finish(route.reason)
    if route.next not in on_duty:
        # The prompt never offered it (the market analyst, switched off). Finish
        # with what we have rather than call a specialist that is not on duty.
        return finish(f"{route.next} is not on duty; finishing ({route.reason})")
    progress("supervisor", f"Asking {route.next}: {route.reason}")
    return Command(goto=route.next, update={
        "task": task_for(route.task, hops, route.order_id),
        "hops": hops + 1,
        "route_log": [{"by": "supervisor", "to": route.next, "task": route.task,
                       "reason": route.reason}]})
