"""supervisor: route the turn to a specialist, by code or by model, or finish."""
from __future__ import annotations

from langgraph.types import Command, Send
from shopwise.core.config import settings
from shopwise.core.constants import CLEAR_CATEGORIES, FINISH, SPECIALIST_FOR
from shopwise.llm.models import ask_json
from shopwise.llm.schemas import Route
from shopwise.llm.workflows.inbox.nodes import as_block, progress, reading
from shopwise.llm.workflows.supervisor.state import TeamState
from shopwise.llm.workflows.supervisor.nodes._common import SUPERVISOR_PROMPT, _model, done_so_far, finish, task_for, team


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
