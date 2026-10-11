"""Score tool selection: does the agent ask for the right tools? — session 5.

    uv run python score_tools.py                       # the configured model
    uv run python score_tools.py --about shipment      # only the shipment questions
    uv run python score_tools.py --provider google_genai:gemini-3.8-flash

A question passes when every tool in its `must` list was called (tool_questions.yml).
Extra calls don't fail it, and neither does the order: the path is the model's
decision. A question with an empty `must` passes only if no tool was called.

The same habit as score.py: before you trust a change to a tool's name, arguments or
docstring, measure it on questions you wrote down beforehand. It lives beside the
notebook, not in project/: it measures the assistant, it is not part of it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "project"))

from langchain.agents import create_agent        # noqa: E402

from agent import MAX_STEPS                       # noqa: E402
from config import settings                       # noqa: E402
from tools import TOOLS                           # noqa: E402
from triage import PROMPTS                        # noqa: E402
from utils import make_model                      # noqa: E402


def load_questions(about: str | None = None) -> list[dict]:
    questions = yaml.safe_load((HERE / "tool_questions.yml").read_text(encoding="utf-8"))
    if about:
        questions = [q for q in questions if q["about"] == about]
    return questions


def tools_called(result: dict) -> list[str]:
    """Every tool the model asked for during one run, in order."""
    names = []
    for message in result["messages"]:
        for call in getattr(message, "tool_calls", None) or []:
            names.append(call["name"])
    return names


def score(agent, questions: list[dict], show: bool = True) -> dict:
    """Run every question once. Returns passed, total, model calls and input tokens."""
    passed = model_calls = input_tokens = 0
    for q in questions:
        try:
            result = agent.invoke({"messages": [{"role": "user", "content": q["question"]}]},
                                  config={"recursion_limit": MAX_STEPS})
        except Exception as err:
            print(f"  FAIL  {q['question'][:58]:58}  {type(err).__name__}")
            continue

        called = tools_called(result)
        if q["must"]:
            ok = set(q["must"]) <= set(called)
        else:
            ok = not called
        passed += ok

        for message in result["messages"]:
            if message.type == "ai":
                model_calls += 1
                input_tokens += (message.usage_metadata or {}).get("input_tokens", 0)
        if show:
            print(f"  {'ok  ' if ok else 'FAIL'}  {q['question'][:58]:58}  {', '.join(called) or '-'}")

    return {"passed": passed, "total": len(questions),
            "model_calls": model_calls, "input_tokens": input_tokens}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", default=settings.model_id, help="a provider string")
    parser.add_argument("--about", help="only questions with this tag, e.g. shipment")
    args = parser.parse_args()

    agent = create_agent(make_model(args.provider), TOOLS, system_prompt=PROMPTS["agent_prompt"])
    questions = load_questions(args.about)
    print(f"{args.provider} · {len(TOOLS)} tools · {len(questions)} questions\n")
    s = score(agent, questions)
    print(f"\ntool selection: {s['passed']}/{s['total']}"
          f" · {s['model_calls']} model calls · {s['input_tokens']:,} input tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
