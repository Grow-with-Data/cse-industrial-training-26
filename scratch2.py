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
    return f"Subject: {by_id[ticket_id]['subject']}\n{by_id[ticket_id]['body'].strip()}"

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

SYS = ("You are ShopWise customer support. Write the reply email the customer receives. Greet the "
       "customer, acknowledge the issue, say a colleague will follow up, and sign off as 'ShopWise Support'.")
shaped = lambda r: r.strip().startswith("Hello,") and "Subject" not in r and r.strip().endswith("ShopWise Support")

MY_EXAMPLE = """
<example>
Hello,

I am so sorry to hear that your kettle is out of stock. I have forwarded this to our fulfillment team, and a colleague will follow up with you shortly.

ShopWise Support
</example>
"""

print("--- TCK-015 ---")
grade(SYS + MY_EXAMPLE, as_email("TCK-015"), shaped, n=1)

print("\n--- TCK-013 ---")
grade(SYS + MY_EXAMPLE, as_email("TCK-013"), shaped, n=1)
