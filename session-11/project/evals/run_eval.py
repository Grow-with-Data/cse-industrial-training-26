"""Score the assistant's ANSWERS in plain Python: no judge, no LangSmith.

    uv run python evals/run_eval.py                    # 2-step RAG (kb/rag.py), settings' defaults
    uv run python evals/run_eval.py --k 1              # change ONE thing...
    uv run python evals/run_eval.py --mode hybrid      # ...or the retrieval mode
    uv run python evals/run_eval.py --rerank           # ...or add the re-ranker
    uv run python evals/run_eval.py --target agent     # score the agent instead

    uv run python evals/run_eval.py --target graph --fusion   # v7: the policy graph,
    uv run python evals/run_eval.py --target graph --grade    #     one technique at a time
    uv run python evals/run_eval.py --arms                    # v7: every arm, one table
    uv run python evals/run_eval.py --followups               # v8: conversations

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

v7 adds two more columns, because every retrieval technique is paid for in
model calls: CALLS (model calls per question, on average) and SECONDS (wall
clock per question). A technique that moves no accuracy number and adds calls
is a cost, and the table has to show it. `--arms` scores straight / fusion /
grade / both / hyde on the same questions, one variable at a time.

v8 adds --followups: evals/conversational.yml's two-turn conversations, where
the second question means nothing on its own ("and for the watch?"). Three arms,
on the policy graph's straight shape so only the question changes:

    follow-ups · as typed     the follow-up searched as written: the bug
    follow-ups · rewritten    kb/query.rewrite_query first: the fix
    standalone · rewritten    data/kb_questions.yml's answerable questions, each
                              asked after an unrelated opening exchange and
                              rewritten: the fix must not damage questions that
                              never needed it

Groundedness (is every claim in the passages?) is not here: it needs a model to
read the answer, so it lives in evals/langsmith_eval.py.

Kept as a script, not notebook cells, so every later session can re-run it after
a change and see whether anything got worse.

Calls the model once per question (19 by default) for two-step; the graph's
arms cost 2 to 6 calls per question each, and --arms runs five of them. Pass
--limit to spend less.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from itertools import count
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402
from langchain_core.callbacks import BaseCallbackHandler  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

from config import settings  # noqa: E402
from kb import rag  # noqa: E402
from kb.retrieval_eval import rank_of  # noqa: E402

# question -> {"answer": str, "documents": [Document], "calls": int (optional)}
Target = Callable[[str], dict]


class CallCounter(BaseCallbackHandler):
    """Counts chat-model calls in one run (new in v7).

    Passed as a callback when the graph is invoked, it is inherited by every
    model call inside every node, including nodes running in parallel threads
    (RAG-Fusion's searches), hence the lock. Embedding calls are not chat
    calls and are not counted: the column is about what a technique adds."""

    def __init__(self):
        self.calls = 0
        self._lock = threading.Lock()

    def on_chat_model_start(self, *args, **kwargs):
        with self._lock:
            self.calls += 1


def questions() -> list[dict]:
    return yaml.safe_load((settings.DATA_DIR / "kb_questions.yml").read_text(encoding="utf-8"))


def two_step(**options) -> Target:
    """kb/rag.py's pipeline as a target. options: k, mode, rerank (None = settings)."""
    options = {key: v for key, v in options.items() if v is not None}
    return lambda q: rag.answer_with_rag(q, **options) | {"calls": 1}   # generate, once


def graph_target(grade: bool = False, fusion: bool = False, hyde: bool = False,
                 k: int | None = None) -> Target:
    """v7's policy graph as a target, built once with the arm's switches.

    Its documents are the final state's: after a rewrite, the second search's.
    Every switch is passed explicitly, so an arm never inherits a setting from
    labs/.env by accident."""
    from workflows.corrective_rag import build_corrective

    graph = build_corrective(grade=grade, fusion=fusion, hyde=hyde)

    def run(q: str) -> dict:
        counter = CallCounter()
        state = {"question": q} | ({"k": k} if k else {})
        out = graph.invoke(state, config={"callbacks": [counter]})
        return {"answer": out["answer"], "documents": out["documents"], "calls": counter.calls}
    return run


# The arms --arms compares. One variable per arm against `straight`, then the
# two that could combine. HyDE is the contrast arm for fusion: both change what
# we search WITH, so they are alternatives, never stacked.
ARMS = {
    "straight": {},
    "fusion": {"fusion": True},
    "grade": {"grade": True},
    "fusion+grade": {"fusion": True, "grade": True},
    "hyde": {"hyde": True},
}


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
        calls = sum(m.type == "ai" for m in result["messages"])    # one AI message per call
        return {"answer": result["messages"][-1].text, "documents": docs, "calls": calls}
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
    timed = [r for r in rows if "seconds" in r]
    counted = [r for r in rows if r.get("calls") is not None]
    return {
        "rows": rows,
        "refusal": sum(r["refused"] == (not r["answerable"]) for r in rows) / len(rows),
        "citation": sum(r["cited_right"] for r in answered) / max(len(answered), 1),
        "retrieval": sum(r["retrieved"] for r in answerable) / max(len(answerable), 1),
        "calls": sum(r["calls"] for r in counted) / len(counted) if counted else None,
        "seconds": sum(r["seconds"] for r in timed) / len(timed) if timed else None,
    }


def score(target: Target, rows: list[dict] | None = None) -> dict:
    """Ask every gold question; return per-question rows and the three metrics."""
    scored = []
    for q in rows or questions():
        started = time.perf_counter()
        result = target(q["question"])
        row = score_row(q, result["answer"], result["documents"])
        row |= {"calls": result.get("calls"), "seconds": time.perf_counter() - started}
        scored.append(row)
    return summarise(scored)


def compare(arms: dict[str, dict] | None = None, rows: list[dict] | None = None,
            k: int | None = None) -> dict[str, dict]:
    """Score several graph arms on the same questions. {arm: result}."""
    rows = rows or questions()
    return {name: score(graph_target(**switches, k=k), rows)
            for name, switches in (arms or ARMS).items()}


# ── v8: conversations ────────────────────────────────────────────────────────

def conversations() -> list[dict]:
    return yaml.safe_load((Path(__file__).with_name("conversational.yml")).read_text(encoding="utf-8"))


# An opening exchange about nothing in particular, for the regression arm: the
# standalone questions get real history, so the rewrite runs, and has nothing to
# resolve. It should hand each question back unchanged.
OPENING = [("human", "Hi, I bought a few things from you last month."),
           ("ai", "Hello! Happy to help. What can I do for you today?")]


def conversation_target(rewrite: bool, unrelated: bool = False):
    """One conversational row -> {"answer", "documents", "calls", "asked"}.

    The row's history is its turn1 and reply1 (or OPENING, for the regression
    arm). The model calls are counted with the same CallCounter as --arms: the
    work runs inside a RunnableLambda invoked with the counter as a callback,
    and every model call inside it, the rewrite included, inherits it."""
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.runnables import RunnableLambda

    from kb.query import rewrite_query
    from workflows.corrective_rag import build_corrective

    graph = build_corrective(grade=False, fusion=False, hyde=False)

    def answer(row: dict) -> dict:
        if unrelated:
            history = [HumanMessage(OPENING[0][1]), AIMessage(OPENING[1][1])]
        else:
            history = [HumanMessage(row["turn1"]), AIMessage(row["reply1"])]
        message = row.get("followup") or row["question"]
        asked = rewrite_query(history, message) if rewrite else message
        out = graph.invoke({"question": asked})
        return {"answer": out["answer"], "documents": out["documents"], "asked": asked}

    def run(row: dict) -> dict:
        counter = CallCounter()
        out = RunnableLambda(answer).invoke(row, config={"callbacks": [counter]})
        return out | {"calls": counter.calls}
    return run


def score_conversations(limit: int | None = None) -> dict[str, dict]:
    """The three --followups arms, each scored like any other arm."""
    pairs = conversations()[:limit]
    standalone = [q for q in questions() if q["answerable"]][:limit]
    arms = {"follow-ups · as typed": (conversation_target(rewrite=False), pairs),
            "follow-ups · rewritten": (conversation_target(rewrite=True), pairs),
            "standalone · rewritten": (conversation_target(rewrite=True, unrelated=True), standalone)}
    results = {}
    for name, (target, rows) in arms.items():
        scored = []
        for row in rows:
            started = time.perf_counter()
            out = target(row)
            scored.append(score_row(row, out["answer"], out["documents"])
                          | {"calls": out["calls"], "seconds": time.perf_counter() - started,
                             "asked": out["asked"]})
        results[name] = summarise(scored)
    return results


def table(results: dict[str, dict]) -> None:
    """One row per arm: the three accuracy numbers, then what they cost."""
    width = max(14, *(len(name) for name in results))
    print(f"{'arm':{width}} {'refusal':>8} {'citation':>9} {'retrieval':>10} {'calls/q':>8} {'s/q':>6}")
    for name, r in results.items():
        calls = f"{r['calls']:.1f}" if r["calls"] is not None else "-"
        print(f"{name:{width}} {r['refusal']:8.2f} {r['citation']:9.2f} {r['retrieval']:10.2f}"
              f" {calls:>8} {r['seconds']:6.1f}")


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
    cost = (f"   calls/q {result['calls']:.1f}" if result.get("calls") is not None else "")
    print(f"{label}\n  refusal {result['refusal']:.2f}   citation {result['citation']:.2f}"
          f"   retrieval {result['retrieval']:.2f}{cost}   s/q {result['seconds']:.1f}")
    for row in result["rows"]:
        if wrong := failures(row):
            print(f"    {row['id']}: {'; '.join(wrong)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", choices=["two-step", "agent", "graph"], default="two-step")
    ap.add_argument("--fusion", action="store_true", help="graph: RAG-Fusion")
    ap.add_argument("--hyde", action="store_true", help="graph: HyDE")
    ap.add_argument("--grade", action="store_true", help="graph: corrective loop")
    ap.add_argument("--arms", action="store_true", help="score every graph arm, one table")
    ap.add_argument("--followups", action="store_true",
                    help="v8: conversational follow-ups, as typed vs rewritten")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--mode", choices=["dense", "keyword", "hybrid"], default=None)
    ap.add_argument("--rerank", action="store_true", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions")
    args = ap.parse_args()

    rows = questions()[: args.limit]
    if args.followups:
        print(f"conversations · policy graph (straight) · k={settings.RAG_K} · {settings.model_id}")
        results = score_conversations(args.limit)
        table(results)
        for row in results["follow-ups · rewritten"]["rows"]:
            print(f"  {row['id']}: {row['asked']}")
        sys.exit(0)
    if args.arms:
        print(f"policy graph · k={args.k or settings.RAG_K} · {settings.model_id} · {len(rows)} questions")
        table(compare(rows=rows, k=args.k))
        sys.exit(0)
    if args.target == "graph":
        target = graph_target(grade=args.grade, fusion=args.fusion, hyde=args.hyde, k=args.k)
    elif args.target == "agent":
        target = agent_target()
    else:
        target = two_step(k=args.k, mode=args.mode, rerank=args.rerank)
    result = score(target, rows)
    arm = "+".join(n for n in ("fusion", "hyde", "grade") if getattr(args, n)) or "straight"
    report(result, f"{args.target}{' · ' + arm if args.target == 'graph' else ''}"
                   f" · k={args.k or settings.RAG_K}"
                   f" · {args.mode or settings.RETRIEVAL_MODE}"
                   f"{' · rerank' if args.rerank or settings.RERANK else ''}"
                   f" · {settings.model_id} · {settings.collection}")
