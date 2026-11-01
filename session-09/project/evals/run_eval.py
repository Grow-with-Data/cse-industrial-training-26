"""Score the assistant's ANSWERS in plain Python: no judge, no LangSmith.

    uv run python evals/run_eval.py                    # 2-step RAG (kb/rag.py), settings' defaults
    uv run python evals/run_eval.py --k 1              # change ONE thing...
    uv run python evals/run_eval.py --mode hybrid      # ...or the retrieval mode
    uv run python evals/run_eval.py --rerank           # ...or add the re-ranker
    uv run python evals/run_eval.py --target agent     # score the agent instead

Every question in data/kb_questions.yml is asked once, and three numbers come
back. A table, never one accuracy, because one number hides which half broke:

    refusal     did it refuse EXACTLY the questions the corpus cannot answer?
                Refusing an answerable question counts against it as much as
                inventing an answer to an unanswerable one. A string check.
    citation    of the answerable questions it answered, how many cite the
                policy file the answer lives in? A set comparison.
    retrieval   of the answerable questions, how many had the gold chunk among
                the passages the model was given? Session 8's hit-rate, re-run
                here as a regression check: generation cannot fix what this missed.

Groundedness (is every claim in the passages?) is not here: it needs a model to
read the answer, so it lives in evals/langsmith_eval.py.

Kept as a script, not notebook cells, so every later session can re-run it after
a change and see whether anything got worse.

Calls the model once per question (19 by default). Pass --limit to spend less.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import count
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

from config import settings  # noqa: E402
from kb import rag  # noqa: E402
from kb.retrieval_eval import rank_of  # noqa: E402

Target = Callable[[str], dict]      # question -> {"answer": str, "documents": [Document]}


def questions() -> list[dict]:
    return yaml.safe_load((settings.DATA_DIR / "kb_questions.yml").read_text(encoding="utf-8"))


def two_step(**options) -> Target:
    """kb/rag.py's pipeline as a target. options: k, mode, rerank (None = settings)."""
    options = {key: v for key, v in options.items() if v is not None}
    return lambda q: rag.answer_with_rag(q, **options)


def agent_target(email: str = "priya@example.com") -> Target:
    """The v6 agent as a target: logged in as one customer, a fresh ticket per
    question. Its documents are whatever its search_policy calls returned, so an
    agent that skipped the search scores a retrieval miss, which is right."""
    from langgraph.checkpoint.memory import InMemorySaver

    from agent import answer, build_agent
    from middleware import customer_for

    agent, tickets, customer = build_agent(InMemorySaver()), count(1), customer_for(email)

    def run(q: str) -> dict:
        result = answer(agent, q, f"EVAL-{next(tickets)}", customer)
        docs = [Document(page_content=p["content"], metadata=p)
                for m in result["messages"] if m.type == "tool" and m.name == "search_policy"
                for p in json.loads(m.content)]
        return {"answer": result["messages"][-1].text, "documents": docs}
    return run


def score_row(q: dict, answer: str, documents: list[Document]) -> dict:
    """One question's verdicts, from its answer and the passages behind it."""
    row = {"id": q["id"], "answerable": q["answerable"], "refused": rag.refused(answer),
           "cited": sorted(rag.cited_sources(answer)), "answer": answer}
    if q["answerable"]:
        row["retrieved"] = rank_of(q, documents) is not None
        row["cited_right"] = q["source"] in row["cited"]
    return row


def summarise(rows: list[dict]) -> dict:
    """The three metrics over scored rows."""
    answerable = [r for r in rows if r["answerable"]]
    answered = [r for r in answerable if not r["refused"]]
    return {
        "rows": rows,
        "refusal": sum(r["refused"] == (not r["answerable"]) for r in rows) / len(rows),
        "citation": sum(r["cited_right"] for r in answered) / max(len(answered), 1),
        "retrieval": sum(r["retrieved"] for r in answerable) / max(len(answerable), 1),
    }


def score(target: Target, rows: list[dict] | None = None) -> dict:
    """Ask every gold question; return per-question rows and the three metrics."""
    scored = []
    for q in rows or questions():
        result = target(q["question"])
        scored.append(score_row(q, result["answer"], result["documents"]))
    return summarise(scored)


def failures(row: dict) -> list[str]:
    """What went wrong on one row, in words. Empty if nothing did."""
    wrong = []
    if row["refused"] != (not row["answerable"]):
        wrong.append("refused an answerable question" if row["refused"]
                     else "answered instead of refusing")
    elif row["answerable"] and not row["cited_right"]:
        wrong.append(f"cited {row['cited'] or 'nothing'}")
    if row["answerable"] and not row["retrieved"]:
        wrong.append("gold chunk not retrieved")
    return wrong


def report(result: dict, label: str) -> None:
    print(f"{label}\n  refusal {result['refusal']:.2f}   citation {result['citation']:.2f}"
          f"   retrieval {result['retrieval']:.2f}")
    for row in result["rows"]:
        if wrong := failures(row):
            print(f"    {row['id']}: {'; '.join(wrong)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", choices=["two-step", "agent"], default="two-step")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--mode", choices=["dense", "keyword", "hybrid"], default=None)
    ap.add_argument("--rerank", action="store_true", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions")
    args = ap.parse_args()

    target = (agent_target() if args.target == "agent"
              else two_step(k=args.k, mode=args.mode, rerank=args.rerank))
    result = score(target, questions()[: args.limit])
    report(result, f"{args.target} · k={args.k or settings.RAG_K}"
                   f" · {args.mode or settings.RETRIEVAL_MODE}"
                   f"{' · rerank' if args.rerank or settings.RERANK else ''}"
                   f" · {settings.model_id} · {settings.collection}")
