"""ShopWise support chat — v4. Logged in, guarded, and refunds wait for a manager.

    uv run python chat.py --as ayesha@example.com

Commands inside the chat:

    /new              start a fresh ticket (a new thread id: a stranger again)
    /tickets          list this run's tickets, * marks the current one
    /switch <id>      go back to one of them
    /history          the current ticket as the agent has it saved
    /cost             tokens this ticket has used, and how much the next turn resends
    /quit             leave (Ctrl-D works too)

  manager only, when a ticket is paused on a refund:
    /approve          the refund runs, and the assistant tells the customer
    /reject <reason>  it doesn't, and the assistant passes the reason on

WHAT CHANGED FROM v3
    --as <email>   stands in for a login. It becomes the runtime context
                   (middleware.Customer) on every call, and middleware.authorize
                   uses it to keep this customer inside their own orders. A real
                   app takes it from the login session, never from the chat.
    a pause        when the assistant asks to issue a refund, the run stops
                   before the tool runs, and the ticket waits. Customer messages
                   are held until a manager decides.
    the manager    in a real helpdesk the manager is a different person on a
                   different screen, maybe hours later. Here they share your
                   terminal, because v4's tickets live in this process's memory:
                   quit, and a paused refund is gone with it. Session 11 saves
                   tickets to disk, so the pause can wait for a manager.
    no word-by-word every model step here has tools available, and with v4's
                   rails, streaming those steps sometimes delivered gpt-oss's
                   tool calls with EMPTY arguments (lookup_customer({}): the
                   notebook, section 11). So the agent is built with
                   stream_tool_steps=False: the status lines still appear as
                   each lookup happens, and each reply arrives whole.
"""

from __future__ import annotations

import argparse
import sys
import warnings

# stream_events(version="v3") is marked beta in langgraph 1.2 and warns on every
# call. We use it on purpose, so silence that one warning and nothing else.
from langchain_core._api import LangChainBetaWarning

warnings.filterwarnings("ignore", category=LangChainBetaWarning)

from langchain_core.callbacks import get_usage_metadata_callback  # noqa: E402
from langchain_core.tracers.langchain import wait_for_all_tracers  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from agent import (  # noqa: E402
    build_agent, decision, new_ticket_id, pending_refunds, run_config, transcript,
)
from config import KEY_NAMES, settings  # noqa: E402
from middleware import customer_for  # noqa: E402

# Dim status lines on a terminal; plain text when piped into a notebook or a log.
DIM, RESET = ("\033[2m", "\033[0m") if sys.stdout.isatty() else ("", "")


def stream_reply(agent, payload, thread_id: str, customer) -> dict:
    """Run one turn, print the reply as it's generated, return the token usage.

    payload is the customer's message, or a Command carrying a manager's decision:
    a resume streams exactly like a new message. As in v3, stream.messages yields
    one item per model call; new in v4, the run may end PAUSED on a refund.
    """
    with get_usage_metadata_callback() as usage:
        stream = agent.stream_events(payload, config=run_config(thread_id),
                                     context=customer, version="v3")
        print("assistant > ", end="", flush=True)
        for message in stream.messages:
            for delta in message.text:
                print(delta, end="", flush=True)      # flush, or the terminal holds it back
            # A message a rail wrote (the content filter, the blocked-reply guard) has no
            # tool-call projection at all: .get() returns None, not [].
            for call in message.tool_calls.get() or []:
                print(f"{DIM}· checking {call['name']}…{RESET} ", end="", flush=True)
        _ = stream.output                             # drive the run to the end (v3's lazy-stream rule)
    if stream.interrupted:
        for request in pending_refunds(agent, thread_id):
            args = request["args"]
            print(f"\n⏸  Your refund request is waiting for a ShopWise manager."
                  f"\n{DIM}   [manager] {request['name']}: {args.get('amount')} on "
                  f"{args.get('order_id')} — {args.get('reason')}"
                  f"\n   [manager] /approve  or  /reject <reason>{RESET}")
    print()
    return usage.usage_metadata


def add_usage(totals: dict, usage: dict) -> None:
    """Fold one turn's usage into a ticket's running totals (unchanged from v3)."""
    for counts in usage.values():
        totals["input"] += counts.get("input_tokens", 0)
        totals["output"] += counts.get("output_tokens", 0)
        totals["cached"] += counts.get("input_token_details", {}).get("cache_read", 0)


def resend_size(agent, thread_id: str) -> int:
    """Roughly how many tokens of history the next turn sends (unchanged from v3)."""
    messages = agent.get_state(run_config(thread_id)).values.get("messages", [])
    for message in reversed(messages):
        if message.type == "ai" and message.usage_metadata:
            return message.usage_metadata["input_tokens"] + message.usage_metadata["output_tokens"]
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="ShopWise support chat (v4)")
    parser.add_argument("--as", dest="email", required=True,
                        help="the customer you are logged in as, e.g. ayesha@example.com")
    args = parser.parse_args(argv[1:])

    if not settings.has_key:
        print(f"No {KEY_NAMES[settings.PROVIDER]} found. Put it in .env and try again.")
        return 1
    try:
        customer = customer_for(args.email)
    except KeyError:
        print(f"No ShopWise customer with email {args.email!r}.")
        return 1

    agent = build_agent(InMemorySaver(), stream_tool_steps=False)   # see the docstring
    ticket = new_ticket_id()
    usage: dict[str, dict] = {}          # ticket id -> running token totals
    echo = not sys.stdin.isatty()        # piped input: show what was "typed"

    print(f"{settings.APP_NAME} v{settings.VERSION} · {settings.model_id} · "
          f"logged in as {customer.email} · ticket {ticket}")
    print("commands: /new /tickets /switch <id> /history /cost /quit · manager: /approve /reject <reason>\n")

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
        paused = bool(pending_refunds(agent, ticket))

        if command == "/quit":
            break
        elif command == "/new":
            ticket = new_ticket_id()
            print(f"— new ticket {ticket}. The assistant has never met you.\n")
        elif command == "/tickets":
            for known in usage:
                waiting = " (waiting for a manager)" if pending_refunds(agent, known) else ""
                print(("* " if known == ticket else "  ") + known + waiting)
            print()
        elif command == "/switch":
            if rest.strip() not in usage:
                print(f"— no ticket {rest.strip()!r} in this run. Try /tickets.\n")
                continue
            ticket = rest.strip()
            print(f"— back on {ticket}\n")
        elif command == "/history":
            state = agent.get_state(run_config(ticket)).values
            print(transcript(state) if state else "(nothing on this ticket yet)")
            print()
        elif command == "/cost":
            totals = usage.get(ticket)
            if not totals:
                print("(nothing on this ticket yet)\n")
                continue
            print(f"{ticket}: {totals['input']:,} tokens in "
                  f"({totals['cached']:,} from cache) · {totals['output']:,} out")
            print(f"the next turn resends about {resend_size(agent, ticket):,} tokens "
                  "before the customer's new message\n")
        elif command in ("/approve", "/reject"):
            if not paused:
                print("— nothing on this ticket is waiting for a manager.\n")
                continue
            if command == "/reject" and not rest.strip():
                print("— a rejection needs a reason: the customer will be told it.\n")
                continue
            resume = Command(resume={"decisions": [decision(command == "/approve", rest.strip())]})
            add_usage(usage[ticket], stream_reply(agent, resume, ticket, customer))
            print()
        elif command.startswith("/"):
            print(f"— unknown command {command}\n")
        elif paused:
            print("— this ticket is waiting for a manager's decision. Start a /new one meanwhile.\n")
        else:
            totals = usage.setdefault(ticket, {"input": 0, "output": 0, "cached": 0})
            try:
                add_usage(totals, stream_reply(
                    agent, {"messages": [{"role": "user", "content": line}]}, ticket, customer))
            except KeyboardInterrupt:
                print(f"\n{DIM}(interrupted){RESET}")
            print()

    print(f"{len(usage)} ticket(s) lived in this process's memory, and they're gone now, "
          "paused refunds included. Session 11 keeps them on disk.")
    wait_for_all_tracers()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
