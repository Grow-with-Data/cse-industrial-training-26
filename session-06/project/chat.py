"""ShopWise support chat — v3's new entry point. Multi-turn, streaming, bounded.

    uv run python chat.py

Commands inside the chat:

    /new            start a fresh ticket (a new thread id: a stranger again)
    /tickets        list this run's tickets, * marks the current one
    /switch <id>    go back to one of them
    /history        the current ticket as the agent has it saved
    /cost           tokens this ticket has used, and how much the next turn resends
    /quit           leave (Ctrl-D works too)

WHY THIS IS A SEPARATE FILE FROM agent.py
    agent.py says what the assistant is; this file is one way of talking to it.
    Session 13 adds another, an HTTP API over the same agent, and neither should
    need to know about the other.

SESSION 6, AS YOU USE IT
    memory      each turn sends only the new message and the ticket id; the
                checkpointer supplies the rest. Tickets live in this process's
                memory, so /quit loses them all. Session 11 puts them on disk.
    streaming   the reply prints as it's generated (stream_events, version="v3"),
                and every lookup shows a status line, because five silent
                seconds while the agent chains three lookups reads as a hang.
    bounding    agent.py's SummarizationMiddleware keeps a long ticket under
                budget. /history shows a SUMMARY line once it has fired, and
                /cost shows the resend shrinking.
"""

from __future__ import annotations

import sys
import warnings

# stream_events(version="v3") is marked beta in langgraph 1.2 and warns on every
# call. We use it on purpose, so silence that one warning and nothing else.
from langchain_core._api import LangChainBetaWarning

warnings.filterwarnings("ignore", category=LangChainBetaWarning)

from langchain_core.callbacks import get_usage_metadata_callback  # noqa: E402
from langchain_core.tracers.langchain import wait_for_all_tracers  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from agent import build_agent, new_ticket_id, run_config, transcript  # noqa: E402
from config import KEY_NAMES, settings  # noqa: E402

# Dim status lines on a terminal; plain text when piped into a notebook or a log.
DIM, RESET = ("\033[2m", "\033[0m") if sys.stdout.isatty() else ("", "")


def stream_reply(agent, text: str, thread_id: str) -> dict:
    """Send one customer message, print the reply as it's generated, return the token usage.

    stream.messages yields one item per MODEL CALL in the run. A question that
    needs three lookups makes four: three that only ask for tools (no text) and
    the one that answers. Print each one's text as it arrives, then a status
    line for any tools it asked for.
    """
    with get_usage_metadata_callback() as usage:
        stream = agent.stream_events(
            {"messages": [{"role": "user", "content": text}]},
            config=run_config(thread_id),
            version="v3",
        )
        print("assistant > ", end="", flush=True)
        for message in stream.messages:
            for delta in message.text:
                print(delta, end="", flush=True)      # flush, or the terminal holds it back
            for call in message.tool_calls.get():
                print(f"{DIM}· checking {call['name']}…{RESET} ", end="", flush=True)
        # The stream is lazy: nothing runs until something reads it, and the last
        # checkpoint is written when the run finishes. Reading .output drives it
        # to the end even if the loop above was left early.
        _ = stream.output
    print()
    return usage.usage_metadata


def add_usage(totals: dict, usage: dict) -> None:
    """Fold one turn's usage into a ticket's running totals.

    The callback reports per model name, summed over every call that model made,
    including the summarizer's.
    """
    for counts in usage.values():
        totals["input"] += counts.get("input_tokens", 0)
        totals["output"] += counts.get("output_tokens", 0)
        totals["cached"] += counts.get("input_token_details", {}).get("cache_read", 0)


def resend_size(agent, thread_id: str) -> int:
    """Roughly how many tokens of history the next turn sends before the customer types a word.

    The last model call's input plus its output is the history as it now stands;
    the next call sends all of it again.
    """
    messages = agent.get_state(run_config(thread_id)).values.get("messages", [])
    for message in reversed(messages):
        if message.type == "ai" and message.usage_metadata:
            return message.usage_metadata["input_tokens"] + message.usage_metadata["output_tokens"]
    return 0


def main() -> int:
    if not settings.has_key:
        print(f"No {KEY_NAMES[settings.PROVIDER]} found. Put it in .env and try again.")
        return 1

    agent = build_agent(InMemorySaver())
    ticket = new_ticket_id()
    usage: dict[str, dict] = {}          # ticket id -> running token totals
    echo = not sys.stdin.isatty()        # piped input: show what was "typed"

    print(f"{settings.APP_NAME} v{settings.VERSION} · {settings.model_id} · ticket {ticket}")
    print("commands: /new /tickets /switch <id> /history /cost /quit\n")

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
            ticket = new_ticket_id()
            print(f"— new ticket {ticket}. The assistant has never met you.\n")
        elif command == "/tickets":
            for known in usage:
                print(("* " if known == ticket else "  ") + known)
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
        elif command.startswith("/"):
            print(f"— unknown command {command}\n")
        else:
            totals = usage.setdefault(ticket, {"input": 0, "output": 0, "cached": 0})
            try:
                add_usage(totals, stream_reply(agent, line, ticket))
            except KeyboardInterrupt:
                print(f"\n{DIM}(interrupted){RESET}")
            print()

    print(f"{len(usage)} ticket(s) lived in this process's memory, and they're gone now. "
          "Session 11 keeps them on disk.")
    wait_for_all_tracers()
    return 0


if __name__ == "__main__":
    sys.exit(main())
