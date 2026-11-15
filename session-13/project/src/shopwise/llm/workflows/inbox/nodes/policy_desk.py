"""The policy desk: facts fetched in code, then the policy graph answers."""
from __future__ import annotations

from langgraph.runtime import Runtime
from shopwise.llm.middleware import ShopContext
from shopwise.llm.workflows.inbox.state import DeskTask
from shopwise.llm.workflows.inbox.nodes._common import progress, who
from shopwise.llm.workflows.inbox.nodes._desks import finding, order_facts, policy_graph


def policy_desk(task: DeskTask, runtime: Runtime[ShopContext]) -> dict:
    """Return and warranty questions: the order's facts, then the cited rule."""
    sub = task["sub_issue"]
    facts = None
    if sub.order_id:
        facts = order_facts(sub.order_id, str(task["ticket"]["received"]),
                            who(task["ticket"], runtime))
    progress("policy", "Searching the policies…")
    result = policy_graph().invoke({"question": sub.question})
    progress("policy", f"Found {len(result.get('documents', []))} policy passages")
    return finding("policy", task, note=result["answer"], facts=facts)
