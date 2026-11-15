"""Make "press approve twice" a non-event: the Idempotency-Key pattern. New in v10.

    result = idem.run(key, body, lambda: approvals.resume(...))

THE BUG THIS PREVENTS
    The manager taps Approve. The request reaches the server, the refund is
    paid, and the network drops before the response arrives. The app, seeing a
    timeout, retries. Without protection, the server resumes the thread again.

THE PATTERN (as Stripe and most payment APIs use it)
    The client makes up a key (a UUID) per decision and sends it in the
    `Idempotency-Key` header, the same key on every retry of that decision.
    The server remembers key -> (a hash of the body, the response it gave):

      new key                  run the request, store the response, return it
      known key, same body     DO NOT run it again: return the stored response
      known key, other body    422: the client reused a key for a different decision
      known key, still running 409: the first attempt has not finished; try again soon

    A failed request releases its key, so the client can retry the same decision.

TWO LINES OF DEFENCE
    Session 11 made issue_refund itself idempotent (one refund per order, checked
    against the ledger). That protects the money. This protects the WORKFLOW:
    the resume, the reply, the trace. Want both.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable

from pydantic import BaseModel

from shopwise.core.exceptions import IdempotencyKeyInProgress, IdempotencyKeyReused, IdempotencyKeyRequired
from shopwise.core.logging_config import get_logger
from shopwise.db.idempotency import IdempotencyRepository

log = get_logger(__name__)


def body_hash(scope: str, body: BaseModel) -> str:
    """The same decision always hashes the same: keys sorted, no whitespace.
    `scope` (the thread id) is part of it, so a key reused on another thread
    counts as a different request."""
    canonical = json.dumps({"scope": scope, "body": body.model_dump(mode="json")},
                           sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class IdempotencyService:
    def __init__(self, keys: IdempotencyRepository) -> None:
        self.keys = keys

    def run[T: BaseModel](self, key: str | None, scope: str, body: BaseModel,
                          action: Callable[[], T], model: type[T]) -> tuple[T, bool]:
        """Run `action` at most once per key. Returns (result, replayed)."""
        if not key:
            raise IdempotencyKeyRequired()
        digest = body_hash(scope, body)
        if not self.keys.start(key, digest):
            row = self.keys.get(key)
            if row is None:                              # expired between start() and get()
                return self.run(key, scope, body, action, model)
            if row.request_hash != digest:
                raise IdempotencyKeyReused()
            if row.status == "in_progress":
                raise IdempotencyKeyInProgress()
            log.info("idempotent replay for key %s", key)
            return model.model_validate_json(row.response_body), True
        try:
            result = action()
        except Exception:
            self.keys.release(key)          # nothing happened: let the client retry
            raise
        self.keys.finish(key, 200, result.model_dump_json())
        return result, False
