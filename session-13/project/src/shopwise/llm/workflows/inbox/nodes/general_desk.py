"""The general desk: an agent for anything that is not orders, billing or policy."""
from __future__ import annotations

from langgraph.runtime import Runtime
from shopwise.llm.middleware import ShopContext
from shopwise.llm.workflows.inbox.state import DeskTask
from shopwise.llm.workflows.inbox.nodes._desks import run_desk


def general_desk(task: DeskTask, runtime: Runtime[ShopContext]) -> dict:
    return run_desk("general", task, runtime)
