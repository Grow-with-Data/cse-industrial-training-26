"""What every specialist shares: the brief it reads, the report it writes, the agent loop.

Three functions, generalised from v8's desks (workflows/inbox/nodes.py):

    brief(task, customer)     v8's brief(), from a supervisor task instead of a SubIssue
    report(name, task, ...)   v8's finding(), the dict compose reads
    run_agent(name, agent, …) v8's run_desk(), for any agent, not only a desk's
"""
from __future__ import annotations

from langchain.messages import HumanMessage

from middleware import Customer
from models import stamp
from shopwise_data import CUSTOMERS
from workflows.inbox.nodes import emit


def brief(task: dict, customer: Customer) -> str:
    """The specialist's one message. Who the customer is comes from the runtime
    context; what to do comes from the supervisor's task. Never the transcript."""
    name = CUSTOMERS[customer.email]["name"]
    return (f"Customer: {name}, customer_id {customer.customer_id}\n"
            f"Order named for this task: {task.get('order_id') or 'none'}\n"
            f"Task: {task['question']}")


def report(name: str, task: dict, note: str, facts: dict | None = None,
           proposal: dict | None = None) -> dict:
    """One finding, in the shape v8's compose and render_findings read."""
    return {"findings": [{"position": task.get("position", 0), "desk": name,
                          "category": task.get("category"), "question": task["question"],
                          "facts": facts, "note": note, "proposal": proposal}]}


def run_agent(name: str, agent, message: str, context, prompt_name: str, prompt: str,
              seen: list | None = None) -> str:
    """Stream one specialist agent's run, emitting a `tool` event per call and
    per result (v8's run_desk), and return its final answer. Pass a list as
    `seen` to collect the tool results too (the knowledge specialist keeps the
    passages it was given, for the scorecard).

    A handoff tool returns Command(graph=Command.PARENT): LangGraph raises it out
    of this loop and applies it to the parent graph, so this function simply
    never returns in that case. That is the handoff."""
    reply = ""
    for update in agent.stream({"messages": [HumanMessage(message)]},
                               config={"run_name": name, **stamp(prompt_name, prompt)},
                               context=context, stream_mode="updates"):
        for delta in update.values():
            messages = delta.get("messages", []) if isinstance(delta, dict) else []
            for m in messages if isinstance(messages, list) else [messages]:
                if m.type == "ai":
                    for call in m.tool_calls:
                        emit({"type": "tool", "node": name, "name": call["name"], "status": "started"})
                    if not m.tool_calls:
                        reply = m.text
                elif m.type == "tool":
                    emit({"type": "tool", "node": name, "name": m.name, "status": "finished"})
                    if seen is not None:
                        seen.append(m)
    return reply
