"""ShopWise support assistant — project version v1.

    uv run python triage.py --limit 3            # three tickets, printed
    uv run python triage.py --csv triage.csv     # the whole inbox, plus a spreadsheet

The same pipeline as v0.2:

    ticket -> triage()  -> TriageResult   what kind of ticket, how urgent, needs a person?
           -> extract() -> TicketFacts    order numbers, product, days since delivery
           -> return_window(facts)        decided in code, not by the model
           -> draft_reply()               the reply, steered by the triage

What v1 changes: the vendor is one string in .env, a fallback takes over
when that vendor fails, tickets run several at a time under a rate limit, and
with LANGSMITH_TRACING=true every call is recorded without any tracing code here.
"""

from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path

import yaml
from langchain_core.runnables import RunnableLambda

from config import settings
from policy import return_window
from schemas import Ticket, TicketFacts, TriageResult
from utils import ask_json, build_model, load_inbox

HERE = Path(__file__).resolve().parent
PROMPTS = yaml.safe_load((HERE / "prompts.yml").read_text(encoding="utf-8"))


def triage_system(with_examples: bool | None = None) -> str:
    """The triage prompt, plus the few-shot examples when they are switched on."""
    if with_examples is None:
        with_examples = settings.USE_FEW_SHOT
    prompt = PROMPTS["triage_prompt"]
    if with_examples:
        prompt += "\nExamples of how ShopWise labels the awkward cases:\n"
        for ex in PROMPTS["few_shot_examples"]:
            prompt += (f"\n<ticket>\nSubject: {ex['subject']}\n{ex['body']}\n</ticket>\n"
                       f"category: {ex['category']}\n")
    return prompt


def triage(model, ticket: Ticket):
    """Decide what the ticket is. Returns (TriageResult, usage)."""
    messages = [("system", triage_system()), ("human", ticket.as_block())]
    return ask_json(model, TriageResult, messages)


def extract(model, ticket: Ticket):
    """Copy the facts out of the ticket. Returns (TicketFacts, usage).

    The ticket text goes in as validation context, so every order number the model
    returns is checked against what the customer actually wrote.
    """
    messages = [("system", PROMPTS["facts_prompt"]), ("human", ticket.as_block())]
    return ask_json(model, TicketFacts, messages,
                    context={"ticket": f"{ticket.subject}\n{ticket.body}"})


def reply_system(today: date) -> str:
    """v0.1's system slot, unchanged: today's date first, then the rules, then examples."""
    examples = ""
    for ex in PROMPTS["examples"]:
        examples += (f"<example>\n<ticket>\n{ex['ticket'].strip()}\n</ticket>\n"
                     f"<reply>\n{ex['reply'].strip()}\n</reply>\n</example>\n")
    return (f"Today's date is {today.isoformat()}.\n\n"
            f"{PROMPTS['system_prompt'].strip()}\n\n## Examples\n{examples}")


def draft_reply(model, ticket: Ticket, result: TriageResult) -> str:
    """Write the reply, now that triage has decided what kind of ticket this is."""
    steer = (f"\nThis ticket was triaged as {result.category}, urgency {result.urgency}; "
             f"the customer sounds {result.sentiment}.")
    if result.needs_human:
        steer += ("\nA colleague must make the actual decision. Don't state or hint at an "
                  "outcome: say who will follow up and what happens next.")

    user_turn = PROMPTS["ticket_template"].format(
        id=ticket.id, subject=ticket.subject, body=ticket.body)
    messages = [("system", reply_system(date.today()) + steer), ("human", user_turn)]
    reply = model.invoke(messages, config={"run_name": "draft_reply"}, stop=["</reply>"])
    return reply.text.split("<reply>", 1)[-1].strip()


def handle_ticket(model, ticket: Ticket, with_reply: bool = True) -> dict:
    """One ticket, start to finish."""
    result, _ = triage(model, ticket)
    facts, _ = extract(model, ticket)
    reply = draft_reply(model, ticket, result) if with_reply else ""
    return {"result": result, "facts": facts, "window": return_window(facts), "reply": reply}


def main() -> int:
    parser = argparse.ArgumentParser(description="Triage and answer the ShopWise inbox.")
    parser.add_argument("--limit", type=int, help="only the first N tickets")
    parser.add_argument("--csv", type=Path, help="also write one row per ticket to this file")
    parser.add_argument("--no-reply", action="store_true", help="skip drafting replies")
    args = parser.parse_args()

    if not settings.has_key:
        print("No GEMINI_API_KEY found. Put it in .env and try again.")
        return 1

    model = build_model()
    tickets = load_inbox()[: args.limit]
    tracing = f"tracing to {settings.LANGSMITH_PROJECT}" if settings.LANGSMITH_TRACING else "tracing off"
    print(f"{settings.APP_NAME} v{settings.VERSION} · {settings.model_id}"
          f" · fallback {settings.FALLBACK or 'off'} · {tracing} · {len(tickets)} ticket(s)\n")

    def one(ticket: Ticket) -> dict:
        return handle_ticket(model, ticket, with_reply=not args.no_reply)

    # Several tickets at once. Each gets its own config so its run in LangSmith is
    # named after the ticket. return_exceptions=True: one failed ticket comes back
    # as its exception instead of stopping the other nineteen.
    configs = [{"run_name": t.id, "max_concurrency": settings.MAX_CONCURRENCY} for t in tickets]
    outcomes = RunnableLambda(one).batch(tickets, configs, return_exceptions=True)

    rows = []
    for ticket, outcome in zip(tickets, outcomes):
        if isinstance(outcome, Exception):
            # One branch for every failure. Vendors raise different exception
            # classes, so the class name is the most useful thing to print.
            print(f"! {ticket.id}: {type(outcome).__name__}: {str(outcome).splitlines()[0][:150]}\n")
            continue

        result, facts = outcome["result"], outcome["facts"]
        print("=" * 76)
        print(f"{ticket.id}  {result.category} | {result.urgency} | {result.sentiment}"
              f" | needs_human={result.needs_human}")
        print(f"  {result.summary}")
        print(f"  orders={facts.order_ids or '-'}  days={facts.days_since_delivery}"
              f"  return window: {outcome['window']}")
        if outcome["reply"]:
            print("-" * 76)
            print(outcome["reply"])
        print()

        rows.append({"id": ticket.id, **result.model_dump(),
                     "order_ids": " ".join(facts.order_ids),
                     "days_since_delivery": facts.days_since_delivery,
                     "return_window": outcome["window"]})

    if args.csv and rows:
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows)} rows to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
