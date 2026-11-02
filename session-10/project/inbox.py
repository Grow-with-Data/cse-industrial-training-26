"""Answer inbound tickets with the v7 graph: one ticket in, one reply out.

    uv run python inbox.py TCK-M01            # one ticket, the reply
    uv run python inbox.py TCK-M01 --steps    # ...and every node's update as it lands
    uv run python inbox.py TCK-005            # single-issue tickets work too
    uv run python inbox.py --list             # every ticket it can read

NEW IN v7, AND NOT A REPLACEMENT FOR chat.py
    chat.py is a conversation: a logged-in customer, turn after turn, saved to
    disk, with a manager approving refunds. This is the inbox: an email arrives
    whole and is answered whole. The graph has no memory and cannot pause for a
    person yet, so it has no business behind a chat window; that is session 11.
    Until then chat.py and agent.py stay exactly as v6 left them.

WHO THE CUSTOMER IS
    The ticket's sender. `customer_for(ticket["from"])` builds the same Customer
    context chat.py builds from `--as`, so the desks' authorize rail holds here
    too: a ticket asking about someone else's order gets a polite no.
"""
from __future__ import annotations

import argparse
import sys

import yaml
from langchain_core.tracers.langchain import wait_for_all_tracers

from config import KEY_NAMES, settings
from middleware import customer_for
from workflows.inbox import build_inbox


def load_all_tickets() -> dict[str, dict]:
    """Both inboxes by id, as plain dicts: the twenty labelled tickets and the
    multi-issue ones. Plain dicts rather than schemas.Ticket, because a
    multi-issue ticket has no single category to validate."""
    tickets = []
    for name in ("tickets.yml", "multi_issue_tickets.yml"):
        tickets += yaml.safe_load((settings.DATA_DIR / name).read_text(encoding="utf-8"))
    return {t["id"]: t for t in tickets}


def show_step(update: dict) -> None:
    """One line per node update: what stream_mode="updates" yields."""
    for node, delta in update.items():
        if node == "triage":
            t = delta["triage"]
            print(f"  · triage    {t.category}, urgency {t.urgency}, needs_human={t.needs_human}")
        elif node == "split":
            for sub in delta["sub_issues"]:
                print(f"  · split     [{sub.category}] {sub.question}")
        elif node == "compose":
            print("  · compose   reply written")
        else:
            for f in delta["findings"]:
                print(f"  · {node:9} #{f['position'] + 1} answered")


def answer_ticket(graph, ticket: dict, steps: bool = False) -> dict:
    """Run the graph on one ticket. Returns the final state."""
    run = {"ticket": ticket}
    context = customer_for(ticket["from"])
    if not steps:
        return graph.invoke(run, context=context)
    final = {"findings": []}
    for update in graph.stream(run, context=context, stream_mode="updates"):
        show_step(update)
        for delta in update.values():
            delta = delta or {}
            final["findings"] += delta.get("findings", [])   # the reducer, by hand
            final |= {k: v for k, v in delta.items() if k != "findings"}
    return final


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ticket_id", nargs="?")
    ap.add_argument("--steps", action="store_true", help="print each node's update as it lands")
    ap.add_argument("--list", action="store_true", help="list the tickets and exit")
    args = ap.parse_args(argv)

    tickets = load_all_tickets()
    if args.list or not args.ticket_id:
        for tid, t in tickets.items():
            print(f"{tid}  {t['from']:<20} {t['subject']}")
        return 0
    if args.ticket_id not in tickets:
        print(f"No ticket {args.ticket_id}. Try --list.")
        return 1
    if not settings.has_key:
        print(f"No {KEY_NAMES[settings.PROVIDER]} found. Put it in labs/.env and try again.")
        return 1

    ticket = tickets[args.ticket_id]
    print(f"{ticket['id']} from {ticket['from']}: {ticket['subject']}\n")
    final = answer_ticket(build_inbox(), ticket, steps=args.steps)
    t = final["triage"]
    print(f"\nQueue: {t.category} · urgency {t.urgency}"
          f"{' · a person must handle the money' if t.needs_human else ''}\n")
    print(final["reply"])
    wait_for_all_tracers()      # send the last traces before the script exits
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
