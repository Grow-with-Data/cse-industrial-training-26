import os
import re
from pathlib import Path
import yaml
from dotenv import load_dotenv
from google import genai

load_dotenv(Path.cwd() / ".env")
client = genai.Client()
MODEL = "gemini-3.5-flash-lite"
TICKETS = yaml.safe_load((Path.cwd() / "data" / "tickets.yml").read_text(encoding="utf-8"))
by_id = {t["id"]: t for t in TICKETS}

def as_email(ticket_id):
    t = by_id[ticket_id]
    return f"Subject: {t['subject']}\n{t['body'].strip()}"

def ask(system, user, model=MODEL, **config):
    if system:
        config["system_instruction"] = system
    r = client.models.generate_content(model=model, contents=user, config=config or None)
    return r.text or ""

def grade(system, user, check, n=3):
    replies = [ask(system, user) for _ in range(n)]
    passed = sum(bool(check(r)) for r in replies)
    print(replies[0].strip(), "\n" + "-" * 72)
    print(f"passed {passed}/{n}")
    return replies

STEP3 = """You are the customer support assistant for ShopWise, an online electronics store in Bangladesh.
You write the reply email the customer receives. Tone: warm, plain-spoken, professional. Never blame the customer.

Output format: a greeting line ("Hello" unless you know their name), two to four short paragraphs of
plain text, then sign off as "ShopWise Support". No markdown, no subject line.

Constraints, and why they exist:
- You have no access to orders, payments or accounts, so never say you have checked, found or confirmed anything.
- Only the billing team can approve refunds, so never promise, approve or start one.
- Never invent timeframes, amounts or policy details; a wrong promise costs more than no promise.
When the customer needs something you cannot do: say so plainly, say a colleague on the right team will
follow up, and tell them what happens next."""

angry = as_email("TCK-003")
acknowledges = lambda r: re.search(r"(sorry|apolog|frustrat|understand)", r.split(".")[0], re.I) is not None
hands_over_with_next_step = lambda r: re.search(r"(colleague|team|specialist|human)", r, re.I) is not None and re.search(r"(will (reply|contact|follow|email|be in touch)|within|next)", r, re.I) is not None

STEP4 = STEP3 + "\nReply in no more than 3 sentences."
print("--- Old STEP3 ---")
grade(STEP3, angry, acknowledges, n=1)
grade(STEP3, angry, hands_over_with_next_step, n=1)

print("\n--- New STEP4 ---")
grade(STEP4, angry, acknowledges, n=1)
grade(STEP4, angry, hands_over_with_next_step, n=1)
