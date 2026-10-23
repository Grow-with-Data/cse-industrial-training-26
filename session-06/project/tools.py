"""The seven things the assistant may ask ShopWise's backend to do — project version v2.

A tool is a function the model may ASK your code to run. The model never sees the
body. It reads four things and nothing else:

    name          a verb: get_order, not order_handler
    arguments     typed, because the type hints become the JSON schema it fills in
    docstring     a one-line summary of what it does and what comes back, when to use
                  it, and an Args: line per argument. parse_docstring=True moves each
                  Args: line into the schema beside its argument. A Returns: section
                  would be dropped, so what comes back is said in the summary.
    return        plain data that survives being turned into text

Errors are RETURNED as {"error": "..."} when the model can do something about them
(ask for a corrected number, try another tool), and each message says what would
work instead. A bug in our own code still raises: the model can't fix that.

The seven signatures are fixed for the rest of the course (PROJECT-SPINE §3).

    uv run python tools.py      # every tool called directly: no model, no key
"""

from __future__ import annotations

from langchain.tools import tool
from langchain_core.tracers.langchain import wait_for_all_tracers

from shopwise_data import CUSTOMERS, ORDERS, PRODUCTS, REFUNDS, SHIPMENTS


@tool(parse_docstring=True)
def lookup_customer(email: str) -> dict:
    """Find a customer by email address: their customer_id, name, tier and join date.

    Call this first when a customer gives their email, because list_orders needs the
    customer_id it returns. Returns {"error": ...} if no customer has that email.

    Args:
        email: The customer's email address, as they wrote it.
    """
    email = email.strip().lower()
    customer = CUSTOMERS.get(email)
    if customer is None:
        return {"error": f"no customer with email {email}. Ask the customer to check the address."}
    return {"email": email, **customer}


@tool(parse_docstring=True)
def get_order(order_id: str) -> dict:
    """Get one order by its ID: status, order and delivery dates, total, and the products in it.

    Use when the customer names an order. Returns {"error": ...} if no order has that ID.

    Args:
        order_id: An order ID like "ORD-5001".
    """
    order_id = order_id.strip().upper()
    order = ORDERS.get(order_id)
    if order is None:
        return {"error": f"no order with ID {order_id}. Order IDs look like ORD-5001."}

    # The stored order has SKUs only. Add the product names here, so the model never
    # has to guess what SKU-100 is.
    items = []
    for item in order["items"]:
        product = PRODUCTS[item["sku"]]
        items.append({"sku": item["sku"], "name": product["name"], "qty": item["qty"],
                      "unit_price": product["price"]})
    return {"order_id": order_id, **order, "items": items}


@tool(parse_docstring=True)
def list_orders(customer_id: str) -> list[dict]:
    """List a customer's orders, newest first: ID, status, order date and total for each.

    Takes the customer_id that lookup_customer returns, not an email. An empty list
    means the customer has never ordered anything.

    Args:
        customer_id: A customer ID like "C-1001".
    """
    customer_id = customer_id.strip().upper()
    found = []
    for order_id, order in ORDERS.items():
        if order["customer_id"] == customer_id:
            found.append({"order_id": order_id, "status": order["status"],
                          "order_date": order["order_date"], "total": order["total"]})
    return sorted(found, key=lambda o: o["order_date"], reverse=True)


@tool(parse_docstring=True)
def get_shipment(order_id: str) -> dict:
    """Track a shipped order: carrier, tracking ID, delivery status and estimated arrival date.

    Use for "where is my parcel?" questions. Returns {"error": ...} if there is no
    tracking for that order, which is a normal answer for an order that hasn't shipped.

    Args:
        order_id: An order ID like "ORD-5003".
    """
    order_id = order_id.strip().upper()
    if order_id not in ORDERS:
        return {"error": f"no order with ID {order_id}. Order IDs look like ORD-5001."}
    shipment = SHIPMENTS.get(order_id)
    if shipment is None:
        return {"error": f"no tracking for {order_id}. get_order shows its current status."}
    return {"order_id": order_id, **shipment}


@tool(parse_docstring=True)
def get_product(sku_or_name: str) -> dict:
    """Look up a product by SKU or name: price, stock and warranty.

    stock 0 means out of stock; warranty_months 0 means no warranty. If nothing
    matches, returns {"error": ...} listing the products that exist; if the name fits
    several, {"error": ...} listing the candidates.

    Args:
        sku_or_name: A SKU like "SKU-300", or part of a product name as the customer said it, like "kettle".
    """
    query = sku_or_name.strip()
    if query.upper() in PRODUCTS:
        return {"sku": query.upper(), **PRODUCTS[query.upper()]}

    # Customers say "the kettle", the catalogue says "BrewMaster Electric Kettle".
    # A case-insensitive substring match is the whole algorithm, so you can predict it.
    matches = []
    for sku, product in PRODUCTS.items():
        if query.lower() in product["name"].lower():
            matches.append(sku)

    if len(matches) == 1:
        return {"sku": matches[0], **PRODUCTS[matches[0]]}
    if len(matches) > 1:
        return {"error": f"{query!r} matches more than one product. Ask which one.",
                "candidates": [{"sku": s, "name": PRODUCTS[s]["name"]} for s in matches]}
    return {"error": f"no product matches {query!r}.",
            "available": [{"sku": s, "name": p["name"]} for s, p in PRODUCTS.items()]}


@tool(parse_docstring=True)
def check_refund_status(refund_id: str) -> dict:
    """Check a refund the customer has already requested: status, amount, order and reason.

    Status "requested" means it is logged but no money has moved yet. Returns
    {"error": ...} if there is no refund with that ID.

    Args:
        refund_id: A refund ID like "REF-9001".
    """
    refund_id = refund_id.strip().upper()
    refund = REFUNDS.get(refund_id)
    if refund is None:
        return {"error": f"no refund with ID {refund_id}. Refund IDs look like REF-9001."}
    return {"refund_id": refund_id, **refund}


@tool(parse_docstring=True)
def issue_refund(order_id: str, amount: float, reason: str) -> dict:
    """Refund money to a customer for an order. This moves real money.

    Only use when a refund has been approved. Returns the new refund record, or
    {"error": ...} if the order doesn't exist or the amount isn't between 0 and the
    order total.

    Args:
        order_id: An order ID like "ORD-5001".
        amount: How much to refund. More than 0, at most the order total.
        reason: Why the refund is being issued, in one sentence.
    """
    # These checks are code, not prompt. A rule written only in a prompt is a rule
    # you have chosen to enforce some of the time.
    order_id = order_id.strip().upper()
    order = ORDERS.get(order_id)
    if order is None:
        return {"error": f"no order with ID {order_id}."}
    if not 0 < amount <= order["total"]:
        return {"error": f"amount must be more than 0 and at most the order total, {order['total']}."}

    refund_id = f"REF-{9001 + len(REFUNDS)}"
    REFUNDS[refund_id] = {"order_id": order_id, "amount": amount, "status": "processed",
                          "reason": reason, "requested_date": order["order_date"]}
    return {"refund_id": refund_id, **REFUNDS[refund_id]}


# What the agent is given. issue_refund works (the notebook calls it) and is left out
# on purpose: these six only read, and that one moves money. Session 7 gives it to the
# agent, with a person approving every call.
TOOLS = [lookup_customer, get_order, list_orders, get_shipment, get_product, check_refund_status]


if __name__ == "__main__":
    print("given to the agent:", [t.name for t in TOOLS])
    print("withheld:          ", issue_refund.name, "\n")
    print(lookup_customer.invoke({"email": "tanvir@example.com"}))
    print(list_orders.invoke({"customer_id": "C-1002"}))
    print(get_order.invoke({"order_id": "ORD-5003"}))
    print(get_shipment.invoke({"order_id": "ORD-5003"}))
    print(get_shipment.invoke({"order_id": "ORD-5002"}))       # not shipped: an honest error
    print(get_product.invoke({"sku_or_name": "kettle"}))
    print(check_refund_status.invoke({"refund_id": "REF-9001"}))
    # With LangSmith tracing on, a script this short can exit before the trace is sent,
    # and LangSmith prints a thread traceback. Wait for it first.
    wait_for_all_tracers()
