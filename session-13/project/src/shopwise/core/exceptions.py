"""The errors ShopWise raises on purpose. New in v9; v10 gives each one an HTTP meaning.

    ShopWiseError                 the base: "this is ours, and we meant it"   500
    ├── RoutingError              the supervisor named a specialist that does not exist
    ├── HandoffError              a specialist handed over without what the next one needs
    ├── KnowledgeBaseMissing      a collection was searched before ingest built it   503
    ├── MCPUnavailable            the MCP server could not be reached              503
    │                             (also a ConnectionError)
    │   v10, raised by the services and turned into HTTP responses by api/errors.py:
    ├── CustomerNotFound          no ShopWise customer with that email             422
    ├── ThreadNotFound            no thread with that id                           404
    ├── ThreadBusy                the thread is already answering a message        409
    ├── ThreadAwaitingApproval    the thread is paused for a manager               409
    ├── ThreadNotPaused           a decision was sent to a thread nothing waits on 409
    ├── InvalidDecision           the manager's decision does not fit the request  422
    ├── TurnFailed                the assistant could not finish the turn          502
    ├── IdempotencyKeyRequired    /resume was called without an Idempotency-Key    400
    ├── IdempotencyKeyReused      the same key arrived with a different body       422
    └── IdempotencyKeyInProgress  the first request with this key is still running 409

WHY A BASE CLASS
    With more than one agent, a failure needs a name. "KeyError: 'refunds_2'" in a
    trace tells you a dict lookup failed somewhere; RoutingError tells you the
    supervisor chose a desk that does not exist, and where to look. One base means
    one `except ShopWiseError` can tell our deliberate failures from bugs, which
    keep raising as the built-in errors they are.

v10: EVERY ERROR CARRIES ITS OWN HTTP ANSWER
    Each class says three things a client needs: `status_code` (404, 409...),
    `error_type` (a stable name such as "thread-not-found") and `title` (one
    human sentence). api/errors.py turns any ShopWiseError into a problem-details
    response (RFC 9457) from those three, so a service never imports FastAPI and
    a route never builds an error by hand. A client branches on `error_type`,
    which never changes, and never on the English `detail`, which may.

MCPUnavailable IS ALSO A ConnectionError
    middleware.RETRY retries TRANSIENT = (TimeoutError, ConnectionError), and
    middleware.TOOL_ERRORS turns a spent retry into a safe message for the model.
    Subclassing ConnectionError puts an MCP outage on that same path without
    touching the rails: the orders specialist retries, then says it could not check.
"""
from __future__ import annotations


class ShopWiseError(Exception):
    """Base for every error ShopWise raises deliberately."""

    status_code: int = 500
    error_type: str = "internal-error"
    title: str = "Something went wrong on our side."

    def __init__(self, detail: str | None = None) -> None:
        # `detail` is the specific sentence for THIS occurrence; `title` is the
        # same for every occurrence of the class.
        self.detail = detail or self.title
        super().__init__(self.detail)


# ── v9: failures inside the assistant ───────────────────────────────────────

class RoutingError(ShopWiseError):
    """The supervisor chose a specialist that is not on the team (or is switched off)."""


class HandoffError(ShopWiseError):
    """A handoff arrived without what the receiving specialist needs."""


class KnowledgeBaseMissing(ShopWiseError):
    """A collection was searched that the ingest script has not built."""

    status_code = 503
    error_type = "knowledge-base-missing"
    title = "The knowledge base has not been built yet."


class MCPUnavailable(ShopWiseError, ConnectionError):
    """The MCP server did not start, did not answer, or went away."""

    status_code = 503
    error_type = "records-unavailable"
    title = "ShopWise's records could not be reached."


# ── v10: errors a client of the API can cause or run into ────────────────────

class CustomerNotFound(ShopWiseError):
    status_code = 422
    error_type = "customer-not-found"
    title = "There is no ShopWise customer with that email."


class ThreadNotFound(ShopWiseError):
    status_code = 404
    error_type = "thread-not-found"
    title = "There is no thread with that id."


class ThreadBusy(ShopWiseError):
    status_code = 409
    error_type = "thread-busy"
    title = "This thread is still answering the previous message."


class ThreadAwaitingApproval(ShopWiseError):
    status_code = 409
    error_type = "thread-awaiting-approval"
    title = "This thread is waiting for a manager's decision."


class ThreadNotPaused(ShopWiseError):
    status_code = 409
    error_type = "thread-not-paused"
    title = "Nothing on this thread is waiting for a decision."


class InvalidDecision(ShopWiseError):
    status_code = 422
    error_type = "invalid-decision"
    title = "The decision does not fit the request it answers."


class TurnFailed(ShopWiseError):
    status_code = 502
    error_type = "turn-failed"
    title = "The assistant could not finish this turn."


class IdempotencyKeyRequired(ShopWiseError):
    status_code = 400
    error_type = "idempotency-key-required"
    title = "This request needs an Idempotency-Key header."


class IdempotencyKeyReused(ShopWiseError):
    status_code = 422
    error_type = "idempotency-key-reused"
    title = "This Idempotency-Key was already used with a different request body."


class IdempotencyKeyInProgress(ShopWiseError):
    status_code = 409
    error_type = "idempotency-key-in-progress"
    title = "The first request with this Idempotency-Key is still running."
