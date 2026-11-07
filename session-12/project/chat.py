"""ShopWise support chat — v9. The team, one conversation at a time.

    uv run python chat.py --as ayesha@example.com                      # a new ticket
    uv run python chat.py --as ayesha@example.com --thread CHAT-1A2B3C  # reopen one

Commands inside the chat:

    /new                       start a fresh ticket
    /thread                    this ticket's id, to reopen it later with --thread
    /history                   the conversation as the graph has it saved
    /quit                      leave (Ctrl-D works too)

  manager only, when the ticket is paused on a refund:
    /approve                   pay what was asked
    /edit <amount> <reason>    pay a different amount, and say why
    /reject <reason>           pay nothing, and say why

WHAT CHANGED FROM v8: CHAT RUNS ON THE TEAM
    One import. chat.py now opens the supervisor graph (workflows/supervisor/)
    instead of the inbox graph, and nothing else changed, because it only ever
    spoke to stream_ticket() and its seven events. That is the event
    vocabulary paying off: the graph behind it was replaced and the client did
    not notice. The supervisor's routing shows up as dim progress lines
    ("Asking knowledge: …").

WHAT CHANGED FROM v7: CHAT RUNS ON THE GRAPH
    Until v7 chat.py drove agent.py, session 6's create_agent. From v8 it is a
    client of workflows/events.stream_ticket(): it sends a line, and prints the
    events that come back. It knows nothing about nodes, state or LangGraph's
    streaming API, which is exactly what session 13's web API will do with the
    same generator, and session 15's React page after it.

    token      printed as it arrives (compose's tokens only)
    progress   a dim status line from a node, e.g. "Checking refund eligibility…"
    tool       a dim "checking get_shipment…" when a desk calls a tool
    interrupt  the approval form, and the manager's three commands
    citation   the reply's sources, listed under it
    error      printed as is: it is already safe to show

TICKETS SURVIVE THE PROCESS
    Every turn is saved to CHECKPOINT_DB under the ticket's thread id. Quit, come
    back tomorrow, pass --thread, and the conversation (and any paused refund) is
    exactly where you left it.

THE MANAGER
    In a real helpdesk the manager is a different person on a different screen,
    maybe on Monday. Here they share your terminal, but not your identity: the
    approval commands resume the graph with role="manager" in the runtime
    context (middleware.ShopContext). A real app takes the role from the login.
"""

from __future__ import annotations

import argparse
import sys
import uuid

from langchain_core.tracers.langchain import wait_for_all_tracers
from langgraph.types import Command

from core.config import KEY_NAMES, settings
from middleware import context_for
from workflows.events import paused_on, run_config, stream_ticket
from workflows.supervisor import open_team

# Dim status lines on a terminal; plain text when piped into a notebook or a log.
DIM, RESET = ("\033[2m", "\033[0m") if sys.stdout.isatty() else ("", "")


def new_thread_id() -> str:
    return f"CHAT-{uuid.uuid4().hex[:6].upper()}"


def show_form(payload: dict) -> None:
    """The approval form, as a manager reads it."""
    print(f"\n⏸  This refund is waiting for a ShopWise manager.")
    print(f"{DIM}   [manager] {payload['order_id']}: refund ${payload['amount']:.2f} "
          f"(paid ${payload['max_amount']:.2f}) · {payload['why']}")
    print(f"   [manager] recommendation: {payload['recommendation']}")
    if payload.get("error"):
        print(f"   [manager] last answer refused: {payload['error']}")
    print(f"   [manager] /approve · /edit <amount> <reason> · /reject <reason>{RESET}")


def render(events) -> None:
    """Print one turn's events. The whole of chat.py's knowledge of the graph."""
    print("assistant > ", end="", flush=True)
    sources = []
    for event in events:
        if event.type == "token":
            print(event.text, end="", flush=True)
        elif event.type == "progress":
            print(f"{DIM}· {event.text}{RESET} ", end="", flush=True)
        elif event.type == "tool" and event.status == "started":
            print(f"{DIM}· checking {event.name}…{RESET} ", end="", flush=True)
        elif event.type == "citation":
            sources.append(event.label)
        elif event.type == "interrupt":
            show_form(event.payload)
        elif event.type == "error":
            print(f"\n— {event.message}")
    print()
    if sources:
        print(f"{DIM}  sources: {' · '.join(f'[{s}]' for s in sources)}{RESET}")


def decision(command: str, rest: str) -> dict | str:
    """A manager command as the resume payload, or a message saying what is wrong.
    The graph validates it again: this only catches typos before a round trip."""
    if command == "/approve":
        return {"type": "approve"}
    if command == "/reject":
        return {"type": "reject", "reason": rest.strip()} if rest.strip() else \
            "a rejection needs a reason: the customer will be told it"
    amount, _, reason = rest.strip().partition(" ")
    try:
        return {"type": "edit", "amount": float(amount.lstrip("$")), "reason": reason.strip()}
    except ValueError:
        return "usage: /edit <amount> <reason>, e.g. /edit 99 one earcup only"


def owner(graph, thread_id: str) -> str | None:
    """Whose ticket a thread is, from its saved state. None for a new thread."""
    values = graph.get_state(run_config(thread_id)).values
    return values["ticket"]["from"] if values.get("ticket") else None


def history(graph, thread_id: str) -> str:
    values = graph.get_state(run_config(thread_id)).values
    lines = [f"(summary) {values['summary']}"] if values.get("summary") else []
    for m in values.get("messages", []):
        lines.append(f"{'customer ' if m.type == 'human' else 'assistant'} > {m.text}")
    return "\n".join(lines) or "(nothing on this ticket yet)"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="ShopWise support chat (v9)")
    parser.add_argument("--as", dest="email", required=True,
                        help="the customer you are logged in as, e.g. ayesha@example.com")
    parser.add_argument("--thread", help="reopen a saved ticket by its id")
    args = parser.parse_args(argv[1:])

    if not settings.has_key:
        print(f"No {KEY_NAMES[settings.PROVIDER]} found. Put it in labs/.env and try again.")
        return 1
    try:
        customer, manager = context_for(args.email), context_for(args.email, role="manager")
    except KeyError:
        print(f"No ShopWise customer with email {args.email!r}.")
        return 1

    echo = not sys.stdin.isatty()        # piped input: show what was "typed"
    with open_team() as graph:
        thread = args.thread or new_thread_id()
        if args.thread and owner(graph, thread) not in (None, customer.email):
            # A thread id is not a secret. Ownership is checked against the saved
            # ticket, never taken on trust from the command line.
            print(f"Ticket {thread} is not yours.")
            return 1

        print(f"{settings.APP_NAME} v{settings.VERSION} · {settings.model_id} · "
              f"logged in as {customer.email} · ticket {thread}")
        print("commands: /new /thread /history /quit · manager: /approve /edit /reject\n")
        if pending := paused_on(graph, thread):
            show_form(pending[0].value)
            print()

        while True:
            try:
                line = input("customer  > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if echo:
                print(line)
            if not line:
                continue
            command, _, rest = line.partition(" ")

            if command == "/quit":
                break
            elif command == "/new":
                thread = new_thread_id()
                print(f"— new ticket {thread}\n")
            elif command == "/thread":
                print(f"— {thread}. Reopen with: uv run python chat.py --as {customer.email} "
                      f"--thread {thread}\n")
            elif command == "/history":
                print(history(graph, thread) + "\n")
            elif command in ("/approve", "/edit", "/reject"):
                if not paused_on(graph, thread):
                    print("— nothing on this ticket is waiting for a manager.\n")
                    continue
                answer = decision(command, rest)
                if isinstance(answer, str):
                    print(f"— {answer}\n")
                    continue
                render(stream_ticket(graph, thread, Command(resume=answer), manager))
                print()
            elif command.startswith("/"):
                print(f"— unknown command {command}\n")
            else:
                try:
                    render(stream_ticket(graph, thread, line, customer))
                except KeyboardInterrupt:
                    print(f"\n{DIM}(interrupted){RESET}")
                print()

    print(f"Ticket {thread} is saved. Reopen it with:  "
          f"uv run python chat.py --as {customer.email} --thread {thread}")
    wait_for_all_tracers()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
