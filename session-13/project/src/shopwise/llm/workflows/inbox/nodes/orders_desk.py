"""The orders desk: an agent with the order and shipment tools."""
from __future__ import annotations

from langgraph.runtime import Runtime
from shopwise.llm.middleware import ShopContext
from shopwise.llm.workflows.inbox.state import DeskTask
from shopwise.llm.workflows.inbox.nodes._desks import run_desk


def orders_desk(task: DeskTask, runtime: Runtime[ShopContext]) -> dict:
    return run_desk("orders", task, runtime)
