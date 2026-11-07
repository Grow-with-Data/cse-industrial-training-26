"""Answer inbound tickets with the v9 team: one email in, one reply out.

    uv run python inbox.py TCK-M01            # one ticket, the reply
    uv run python inbox.py TCK-M01 --steps    # ...and every progress and tool event
    uv run python inbox.py TCK-M05            # a refund: pauses for a manager
    uv run python inbox.py TCK-M05 --fresh    # forget the saved thread and run again
    uv run python inbox.py --list             # every ticket it can read

v9: THE TEAM ANSWERS
    The supervisor graph replaces the inbox graph here, through one import, as
    in chat.py. --steps now also shows the routing: the code route or each of
    the supervisor's choices, then every specialist's tool calls. --graph v8
    runs the old inbox graph instead, which is still in the snapshot.

THE ONE-SHOT ENTRY, NOW SAVED
    chat.py is a conversation; this is the inbox: an email arrives whole and is
    answered whole. From v8 both run the same graph through the same
    workflows/events.stream_ticket(), and both save to CHECKPOINT_DB. Each
    ticket's thread id is its ticket id, so running a ticket twice prints the
    saved reply rather than paying to answer it again (--fresh starts over).

A REFUND PAUSES
    TCK-M05 asks for a refund. The billing desk works out what is due and stops
    for a manager. This program does not decide it. The ticket waits, saved, and
    a manager answers it from the chat, as the customer's ticket:

        uv run python chat.py --as ayesha@example.com --thread TCK-M05
        customer  > /approve

WHO THE CUSTOMER IS
    The ticket's sender: context_for(ticket["from"]), the same ShopContext
    chat.py builds from --as.
"""
from __future__ import annotations

import argparse
import sys

import yaml
from langchain_core.tracers.langchain import wait_for_all_tracers

from core.config import KEY_NAMES, settings
from middleware import context_for
from workflows.events import paused_on, run_config, stream_ticket
from workflows.inbox import open_inbox
from workflows.supervisor import open_team


def load_all_tickets() -> dict[str, dict]:
    """Both inboxes by id, as plain dicts: the twenty labelled tickets and the
    multi-issue ones (session 11's stories are TCK-M05 to M09)."""
    tickets = []
    for name in ("tickets.yml", "multi_issue_tickets.yml"):
        tickets += yaml.safe_load((settings.DATA_DIR / name).read_text(encoding="utf-8"))
    return {t["id"]: {**t, "received": str(t["received"])} for t in tickets}


def answer_ticket(graph, ticket: dict, steps: bool = False) -> dict:
    """Run one ticket through stream_ticket(). Returns {"reply", "paused"}."""
    result = {"reply": "", "paused": None}
    for event in stream_ticket(graph, ticket["id"], ticket, context_for(ticket["from"])):
        if event.type == "progress" and steps:
            print(f"  · {event.node:9} {event.text}")
        elif event.type == "tool" and steps and event.status == "started":
            print(f"  · {event.node:9} calls {event.name}")
        elif event.type == "interrupt":
            result["paused"] = event.payload
        elif event.type == "done":
            result["reply"] = event.reply
        elif event.type == "error":
            result["reply"] = f"[error] {event.message}"
    return result


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ticket_id", nargs="?")
    ap.add_argument("--steps", action="store_true", help="print progress and tool events")
    ap.add_argument("--fresh", action="store_true", help="delete the saved thread first")
    ap.add_argument("--list", action="store_true", help="list the tickets and exit")
    ap.add_argument("--graph", choices=["v9", "v8"], default="v9",
                    help="v9: the team (default); v8: the inbox graph, for comparison")
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
    with (open_team() if args.graph == "v9" else open_inbox()) as graph:
        if args.fresh:
            graph.checkpointer.delete_thread(ticket["id"])
        saved = graph.get_state(run_config(ticket["id"])).values
        if pending := paused_on(graph, ticket["id"]):
            print(f"Waiting for a manager: refund ${pending[0].value['amount']:.2f} on "
                  f"{pending[0].value['order_id']}. Answer it with:\n  uv run python chat.py "
                  f"--as {ticket['from']} --thread {ticket['id']}")
            return 0
        if saved.get("reply"):
            print(f"(already answered: the saved reply. --fresh to run it again)\n\n{saved['reply']}")
            return 0

        result = answer_ticket(graph, ticket, steps=args.steps)
        if result["paused"]:
            p = result["paused"]
            print(f"\n⏸  Paused for a manager: refund ${p['amount']:.2f} on {p['order_id']} "
                  f"({p['why']}; recommend {p['recommendation']}).\n   Answer it with:  "
                  f"uv run python chat.py --as {ticket['from']} --thread {ticket['id']}")
        else:
            t = graph.get_state(run_config(ticket["id"])).values["triage"]
            print(f"\nQueue: {t.category} · urgency {t.urgency}\n\n{result['reply']}")
    wait_for_all_tracers()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
