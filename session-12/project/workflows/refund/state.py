"""What the refund graph remembers while it works on ONE refund request.

ITS OWN SCHEMA, ON PURPOSE
    The supervisor's state (workflows/supervisor/state.py) is about a turn: a
    conversation, a route log, many findings. This graph is about one request
    for money back on one order. None of its keys has the same name and meaning
    in the parent, so it is NOT added to the supervisor graph as a node. The
    refunds specialist CALLS it inside a node and translates in and out
    (workflows/specialists/refunds.py): the use-subgraphs rule "different
    schema, call it and translate".

    Who the customer is travels as two plain strings, not a Customer object: a
    paused refund is checkpointed, and a checkpoint stores plain data.
"""
from __future__ import annotations

from typing import TypedDict


class RefundState(TypedDict, total=False):
    # in: what the specialist hands over
    ticket: dict              # the ticket the request came from (id, received, ...)
    order_id: str
    reason: str               # the request in words: becomes the refund's reason
    position: int             # where the customer raised it: compose matches outcomes by it
    email: str                # whose ticket (runtime context, flattened)
    customer_id: str

    # worked out inside
    note: str                 # what the reply should say about it
    facts: dict | None        # the order's facts, from v8's order_facts
    proposal: dict | None     # the approval form; None when a hard rule refused

    # out
    decision: dict | None     # the manager's answer, with the proposal it answered
    outcome: dict | None      # what settle() did: paid, already paid, or declined
