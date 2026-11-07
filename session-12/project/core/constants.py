"""Fixed facts of ShopWise's domain, in one place. New in v9.

A constant belongs here when more than one feature reads it and it is NOT a
setting: nobody should change it from labs/.env. The return window is a policy
fact (returns-policy 1.1), not a knob; the specialists' names are the vocabulary
the supervisor's structured output, the code route, the comparison and the trace
all share, so a typo in one of them must be impossible rather than unlikely.

Settings change per deployment (config.py). Constants change only when the shop
or the code changes, and then in a reviewed diff.
"""
from __future__ import annotations

RETURN_WINDOW_DAYS = 30          # returns-policy 1.1. The manager applies the exceptions.

# ── The team (workflows/supervisor/) ─────────────────────────────────────────
# One line each, written for the supervisor model: the supervisor's prompt lists
# exactly these, so the remit IS the routing logic, the way a tool's docstring
# is (session 5). Change a remit and you change where tickets go.
SPECIALISTS = {
    "orders": "Looks up THIS customer's records: orders, shipments and tracking, the status "
              "of a refund already requested, products, prices and stock. Never answers "
              "questions about rules.",
    "knowledge": "Answers questions about rules, from ShopWise's policies and the law, and "
                 "says how similar past cases were handled. Cites its sources.",
    "refunds": "Works out a request for money back on ONE named order or refund id, and puts "
               "it to a ShopWise manager. Answers no questions.",
    "market": "[ADV] Compares a ShopWise product's price with alternatives on the open web, "
              "with links. Never invents a price.",
}
FINISH = "FINISH"

# The code route (ROUTING=hybrid or code): which specialist a sub-issue goes to
# by its category, with no model call. v8's DESK_FOR, renamed for the team: the
# orders and billing desks became the orders specialist, the policy desk became
# knowledge, billing's refund path became refunds.
SPECIALIST_FOR = {
    "order_status": "orders",
    "shipping": "orders",
    "billing": "orders",
    "product_question": "orders",
    "refund": "refunds",
    "return": "knowledge",
    "warranty": "knowledge",
    "complaint": "orders",
    "other": "orders",
}
# Hybrid routing takes the code route only for these. A complaint or "other" is
# not a clear category: the right specialist depends on what it is about, which is
# exactly the judgement the supervisor is paid for.
CLEAR_CATEGORIES = {"order_status", "shipping", "billing", "product_question",
                    "refund", "return", "warranty"}

# ── Metadata the knowledge base carries (kb/) ────────────────────────────────
# The closed vocabularies a self-query filter may use. backend.yml's product_type
# values, and the topics the policy files cover.
PRODUCT_TYPES = ("audio", "wearables", "home", "outdoor")
TOPIC_SOURCES = {
    "returns": ["returns-policy"],
    "shipping": ["shipping-policy", "ftc-prompt-delivery"],
    "warranty": ["warranty-policy", "cfr-16-part-700"],
}
