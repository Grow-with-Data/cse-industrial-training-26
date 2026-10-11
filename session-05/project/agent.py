"""The ShopWise agent — project version v2. The assistant's second entry point.

triage.py classifies a ticket. This answers a customer's question from ShopWise's
records. Two jobs, two entry points (ADR-0007).

create_agent is the loop you write by hand in section 4 of the notebook:

    for step in range(budget):
        ai = model_with_tools.invoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            return ai                                  # an answer, not a request
        for call in ai.tool_calls:
            messages.append(tools_by_name[call["name"]].invoke(call))

It forgets: every answer() starts from an empty message list. Session 6 gives it
a memory.

    uv run python agent.py                                 # the three demo questions
    uv run python agent.py "Where is my order ORD-5003?"   # your own
"""

from __future__ import annotations

import sys

from langchain.agents import create_agent
from langchain_core.tracers.langchain import wait_for_all_tracers

from config import settings
from tools import TOOLS
from triage import PROMPTS
from utils import build_model

# create_agent allows 9,999 steps unless you say otherwise, which is no budget at all.
# One lookup costs two steps (the model asks, the tools run), so 12 allows five
# lookups and the answer. Session 7 replaces this with call-limit middleware.
MAX_STEPS = 12

DEMO = [
    "Hi, I'm tanvir@example.com. What did I order most recently, and when will it arrive?",
    "I'm kamal@example.com. Can you list my orders?",
    "Is the BrewMaster kettle in stock, and what warranty does it have?",
]


def build_agent():
    """The model, the six read-only tools, and the grounded prompt. issue_refund isn't one of them."""
    return create_agent(model=build_model(), tools=TOOLS, system_prompt=PROMPTS["agent_prompt"])


def answer(agent, question: str) -> dict:
    """Answer one question. Returns the whole result, because result["messages"] is the transcript."""
    return agent.invoke({"messages": [{"role": "user", "content": question}]},
                        config={"recursion_limit": MAX_STEPS, "run_name": "agent"})


def transcript(result: dict) -> str:
    """The run as a readable conversation: who asked for what, and what came back."""
    lines = []
    for message in result["messages"]:
        if message.type == "human":
            lines.append(f"CUSTOMER   {message.text}")
        elif message.type == "tool":
            lines.append(f"  RESULT   {message.name} -> {message.content}")
        elif message.tool_calls:
            for call in message.tool_calls:
                lines.append(f"  ASKS     {call['name']}({call['args']})")
        else:
            lines.append(f"ASSISTANT  {message.text}")
    return "\n".join(lines)


def main() -> int:
    if not settings.has_key:
        print("No GEMINI_API_KEY found. Put it in .env and try again.")
        return 1

    agent = build_agent()
    print(f"{settings.APP_NAME} v{settings.VERSION} · {settings.model_id} · "
          f"{len(TOOLS)} tools · budget {MAX_STEPS} steps\n")
    for question in sys.argv[1:] or DEMO:
        print("=" * 78)
        print(transcript(answer(agent, question)))
        print()
    wait_for_all_tracers()      # send the last traces before the script exits (see tools.py)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
