"""The errors ShopWise raises on purpose. New in v9.

    ShopWiseError           the base: "this is ours, and we meant it"
    ├── RoutingError        the supervisor named a specialist that does not exist
    ├── HandoffError        a specialist handed over without what the next one needs
    ├── KnowledgeBaseMissing  a collection was searched before kb/ingest.py built it
    └── MCPUnavailable      the MCP server could not be reached (also a ConnectionError)

WHY A BASE CLASS
    With more than one agent, a failure needs a name. "KeyError: 'refunds_2'" in a
    trace tells you a dict lookup failed somewhere; RoutingError tells you the
    supervisor chose a desk that does not exist, and where to look. One base means
    one `except ShopWiseError` can tell our deliberate failures from bugs, which
    keep raising as the built-in errors they are. Session 13 maps each subclass to
    an HTTP status; that is why each one is specific.

MCPUnavailable IS ALSO A ConnectionError
    middleware.RETRY retries TRANSIENT = (TimeoutError, ConnectionError), and
    middleware.TOOL_ERRORS turns a spent retry into a safe message for the model.
    Subclassing ConnectionError puts an MCP outage on that same path without
    touching the rails: the orders specialist retries, then says it could not check.
"""
from __future__ import annotations


class ShopWiseError(Exception):
    """Base for every error ShopWise raises deliberately."""


class RoutingError(ShopWiseError):
    """The supervisor chose a specialist that is not on the team (or is switched off)."""


class HandoffError(ShopWiseError):
    """A handoff arrived without what the receiving specialist needs."""


class KnowledgeBaseMissing(ShopWiseError):
    """A collection was searched that kb/ingest.py has not built."""


class MCPUnavailable(ShopWiseError, ConnectionError):
    """The MCP server did not start, did not answer, or went away."""
