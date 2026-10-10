"""Score the assistant against what we know is true — session 4 version.

    uv run python score.py                                     # the configured model
    uv run python score.py --provider groq:openai/gpt-oss-20b  # any provider string
    uv run python score.py --provider groq:openai/gpt-oss-20b --runs 2

Same two numbers as session 3:

  triage accuracy   agreement with the hand labels in tickets.yml (OUR convention)
  extraction        order numbers compared with a regex over the ticket text:
                    found / missed / invented. No judgment involved.

New in session 4: --provider. Before you switch vendor, or trust a fallback,
score it here. The score always names ONE model, so no fallback is attached.

It lives beside the notebook, not in project/: it measures the assistant, it is
not part of it.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "project"))        # reuse the snapshot's code

import triage as project                               # noqa: E402
from config import settings                            # noqa: E402
from schemas import BANGLA_DIGITS, TriageResult        # noqa: E402
from utils import ask_json, load_inbox, make_model     # noqa: E402


def score_triage(model, tickets):
    """Category accuracy against the hand labels. Returns (hits, misses, output tokens)."""
    hits, misses, out_tokens = 0, [], 0
    for ticket in tickets:
        messages = [("system", project.triage_system()), ("human", ticket.as_block())]
        try:
            result, usage = ask_json(model, TriageResult, messages)
        except Exception as err:
            misses.append((ticket.id, ticket.category, f"FAILED {type(err).__name__}"))
            continue
        out_tokens += usage["output_tokens"]
        if result.category == ticket.category:
            hits += 1
        else:
            misses.append((ticket.id, ticket.category, result.category))
    return hits, misses, out_tokens


def true_order_ids(ticket) -> set[str]:
    """Ground truth: every ORD-number written in the ticket, found by a regex."""
    text = f"{ticket.subject} {ticket.body}".translate(BANGLA_DIGITS)
    return {f"ORD-{n}" for n in re.findall(r"ORD-?(\d+)", text, re.I)}


def score_extraction(model, tickets):
    """Order-number extraction. Returns (found, missed, invented, rejected)."""
    found = missed = invented = rejected = 0
    for ticket in tickets:
        truth = true_order_ids(ticket)
        try:
            facts, _ = project.extract(model, ticket)
        except Exception:
            rejected += 1          # the validator refused it, even after one repair
            missed += len(truth)
            continue
        got = set(facts.order_ids)
        found += len(got & truth)
        missed += len(truth - got)
        invented += len(got - truth)
    return found, missed, invented, rejected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", default=settings.model_id,
                        help="provider string to score, e.g. groq:openai/gpt-oss-20b")
    parser.add_argument("--runs", type=int, default=1, help="how many times to score it")
    args = parser.parse_args()

    model = make_model(args.provider)
    tickets = load_inbox()
    n = len(tickets)
    print(f"scoring {args.provider} on {n} tickets, {args.runs} run(s)\n")

    for run in range(1, args.runs + 1):
        hits, misses, out_tokens = score_triage(model, tickets)
        found, missed, invented, rejected = score_extraction(model, tickets)
        print(f"run {run}: triage {hits}/{n} · {out_tokens} output tokens · order numbers "
              f"{found} found, {missed} missed, {invented} invented, {rejected} rejected")
        for tid, truth, said in misses:
            print(f"    {tid}  labelled {truth:<17} model said {said}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
