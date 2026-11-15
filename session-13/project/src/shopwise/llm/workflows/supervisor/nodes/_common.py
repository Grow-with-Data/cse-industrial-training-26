"""What the supervisor's nodes share: its prompt, its model, and how a task is written."""
from __future__ import annotations

from functools import lru_cache
from langgraph.types import Command
from shopwise.core.config import settings
from shopwise.core.constants import FINISH, SPECIALISTS
from shopwise.llm.models import build_model
from shopwise.llm.prompts.loader import load_prompts
from shopwise.llm.workflows.supervisor.state import TeamState


SUPERVISOR_PROMPT = load_prompts("team")["supervisor_prompt"]


REPLY = "compose"          # where every finished turn goes: v8's compose node


@lru_cache
def _model():
    return build_model()


def team() -> dict[str, str]:
    """The specialists on duty, with their remits. The market analyst is [ADV]."""
    return {name: remit for name, remit in SPECIALISTS.items()
            if name != "market" or settings.MARKET_ANALYST}


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
