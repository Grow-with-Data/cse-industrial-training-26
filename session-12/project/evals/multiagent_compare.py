"""The multi-agent comparison: the same eight tickets through four systems. New in v9.

    uv run python evals/multiagent_compare.py              # all four, writes docs/multiagent-comparison.md
    uv run python evals/multiagent_compare.py --only hybrid --limit 2

THE FOUR SYSTEMS
    v8 static router    workflows/inbox: split, then each issue to its desk by
                        category, in parallel. Code routes; 0 routing calls.
    subagents           workflows/subagents.py: one agent whose tools are the
                        specialists. The model routes, inside its own loop.
    supervisor          workflows/supervisor with ROUTING=supervisor: a
                        supervisor node routes every turn, one hop at a time.
    hybrid (shipped)    the same graph with ROUTING=hybrid: one clear issue
                        goes by code, everything else to the supervisor.

WHAT IS MEASURED, PER TICKET
    calls      chat-model calls (a callback counts every one, inside every agent)
    tokens     input + output tokens, from each call's usage metadata
    seconds    wall clock for the whole turn
    correct    judged on the REPLY, the same way for every system: every
               `mention` group present and every `cite` source cited
               (evals/team_tickets.yml). Who handled it is shown, not scored.

No memory: every graph is built without a Store, so recall and remember cost
nothing, for all four alike. Each ticket gets a fresh thread.

Cost: about 4 to 10 calls per ticket per system, so roughly 200 calls for the
whole table. Pass --limit and --only to spend less.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402
from langchain_core.callbacks import BaseCallbackHandler  # noqa: E402
from langchain_core.tracers.langchain import wait_for_all_tracers  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from core.config import settings  # noqa: E402
from kb import rag  # noqa: E402
from middleware import context_for  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
REPORT = PROJECT / "docs" / "multiagent-comparison.md"


class Meter(BaseCallbackHandler):
    """Counts chat-model calls and the tokens they used, across every nested run."""

    def __init__(self):
        self.calls = self.tokens = 0
        self._lock = threading.Lock()

    def on_chat_model_start(self, *args, **kwargs):
        with self._lock:
            self.calls += 1

    def on_llm_end(self, response, **kwargs):
        used = 0
        for generation in (g for gens in response.generations for g in gens):
            usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
            used += usage.get("total_tokens", 0)
        with self._lock:
            self.tokens += used


def tickets() -> list[dict]:
    return yaml.safe_load(Path(__file__).with_name("team_tickets.yml").read_text(encoding="utf-8"))


def as_ticket(t: dict, run: int) -> dict:
    return {"id": f"{t['id']}-{run}", "from": t["from"], "subject": "", "body": t["body"],
            "received": settings.today}


def correct(t: dict, reply: str) -> tuple[bool, list[str]]:
    """Every mention group present, every cite source cited. And what was missing."""
    missing = []
    text = reply.lower()
    for group in t.get("mention", []):
        if not any(p.lower() in text for p in group):
            missing.append(f"mention {group[0]!r}")
    cited = rag.cited_sources(reply)
    missing += [f"cite {s}" for s in t.get("cite", []) if s not in cited]
    return not missing, missing


# ── the four systems: ticket -> {"reply", "routes"} ──────────────────────────

def graph_system(build):
    """A LangGraph system (v8's inbox or v9's team), built once, fresh thread per ticket."""
    from workflows.inbox.graph import checkpoint_serde
    graph = build(InMemorySaver(serde=checkpoint_serde()))

    def run(ticket: dict, meter: Meter) -> dict:
        config = {"configurable": {"thread_id": ticket["id"]}, "callbacks": [meter],
                  "run_name": "compare", "metadata": {"thread_id": ticket["id"]}}
        out = graph.invoke({"ticket": ticket}, config, context=context_for(ticket["from"]))
        if "route_log" in out:
            routes = [e["to"] for e in out["route_log"] if e["to"] != "FINISH"]
        else:
            routes = [f["desk"] for f in out.get("findings", [])]
        return {"reply": out.get("reply", ""), "routes": routes}
    return run


def subagents_system():
    from workflows.subagents import answer_with_team

    def run(ticket: dict, meter: Meter) -> dict:
        from langchain_core.runnables import RunnableLambda
        out = RunnableLambda(lambda t: answer_with_team(t["body"], context_for(t["from"]))).invoke(
            ticket, config={"callbacks": [meter]})
        names = {"ask_orders_expert": "orders", "ask_policy_expert": "knowledge",
                 "ask_refunds_expert": "refunds", "load_skill": "skill"}
        return {"reply": out["reply"], "routes": [names.get(c, c) for c in out["consulted"]]}
    return run


def systems() -> dict:
    from workflows.inbox import build_inbox
    from workflows.supervisor import build_team
    return {
        "v8 static router": lambda: graph_system(lambda cp: build_inbox(checkpointer=cp)),
        "subagents": subagents_system,
        "supervisor": lambda: graph_system(lambda cp: build_team(checkpointer=cp, routing="supervisor")),
        "hybrid (shipped)": lambda: graph_system(lambda cp: build_team(checkpointer=cp, routing="hybrid")),
    }


def compare(only: list[str] | None = None, limit: int | None = None) -> dict[str, list[dict]]:
    """{system: [one row per ticket]}. Each row: id, shape, calls, tokens,
    seconds, correct, missing, routes, reply."""
    rows_by_system = {}
    for name, make in systems().items():
        if only and name.split()[0] not in only:
            continue
        system, rows = make(), []
        for n, t in enumerate(tickets()[:limit]):
            meter, started = Meter(), time.perf_counter()
            try:
                out = system(as_ticket(t, n), meter)
            except Exception as exc:                       # noqa: BLE001 — a row, not a crash
                out = {"reply": f"[error] {type(exc).__name__}: {exc}", "routes": []}
            ok, missing = correct(t, out["reply"])
            rows.append({"id": t["id"], "shape": t["shape"], "calls": meter.calls,
                         "tokens": meter.tokens, "seconds": time.perf_counter() - started,
                         "correct": ok, "missing": missing, "routes": out["routes"],
                         "reply": out["reply"]})
            print(f"  {name:17} {t['id']:8} {meter.calls:>2} calls {meter.tokens:>6} tok "
                  f"{rows[-1]['seconds']:5.1f}s  {'ok ' if ok else 'MISS'} {' → '.join(out['routes'])}",
                  flush=True)
        rows_by_system[name] = rows
    return rows_by_system


def summary(rows: list[dict]) -> dict:
    return {"n": len(rows),
            "calls": statistics.mean(r["calls"] for r in rows) if rows else 0,
            "tokens": statistics.mean(r["tokens"] for r in rows) if rows else 0,
            "p50": statistics.median(r["seconds"] for r in rows) if rows else 0,
            "correct": sum(r["correct"] for r in rows)}


def markdown(results: dict[str, list[dict]]) -> str:
    """The committed report: one summary table per ticket shape, then every row."""
    lines = ["# Multi-agent comparison", "",
             f"Generated by `uv run python evals/multiagent_compare.py` on {settings.model_id}, "
             f"AS_OF {settings.today}. Re-run it before changing `ROUTING` in `core/config.py`.", "",
             "Calls and tokens are per ticket, averaged; seconds is the median (p50). "
             "*Correct* is judged on the reply (evals/team_tickets.yml).", ""]
    for shape in ("single", "multi"):
        lines += [f"## {shape}-issue tickets", "",
                  "| system | calls | tokens | p50 s | correct |", "|---|---:|---:|---:|---:|"]
        for name, rows in results.items():
            s = summary([r for r in rows if r["shape"] == shape])
            lines.append(f"| {name} | {s['calls']:.1f} | {s['tokens']:.0f} | {s['p50']:.1f} | "
                         f"{s['correct']}/{s['n']} |")
        lines.append("")
    lines += ["## Every ticket", "", "| system | ticket | calls | tokens | s | correct | routed to |",
              "|---|---|---:|---:|---:|---|---|"]
    for name, rows in results.items():
        for r in rows:
            verdict = "yes" if r["correct"] else "no: " + "; ".join(r["missing"])
            lines.append(f"| {name} | {r['id']} | {r['calls']} | {r['tokens']} | {r['seconds']:.1f} | "
                         f"{verdict} | {' → '.join(r['routes']) or '-'} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", choices=["v8", "subagents", "supervisor", "hybrid"])
    ap.add_argument("--limit", type=int, default=None, help="only the first N tickets")
    ap.add_argument("--no-write", action="store_true", help="print, do not write the report")
    args = ap.parse_args()
    if not settings.has_key:
        sys.exit("No model key in labs/.env.")
    results = compare(args.only, args.limit)
    report = markdown(results)
    print("\n" + report)
    if not args.no_write and not args.only and not args.limit:
        REPORT.parent.mkdir(exist_ok=True)
        REPORT.write_text(report, encoding="utf-8")
        print(f"wrote {REPORT.relative_to(PROJECT)}")
    wait_for_all_tracers()
