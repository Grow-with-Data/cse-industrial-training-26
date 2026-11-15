"""Code that runs around every request: the request id, and CORS. New in v10.

MIDDLEWARE, IN ONE SENTENCE
    A function that sees every request before the route and every response
    after it. Good for things that apply to ALL routes (ids, timing, CORS);
    wrong for things that apply to some (those are dependencies).

THE REQUEST ID
    Read from the X-Request-ID header if a proxy already set one (session 16's
    Nginx can), otherwise made up here. It goes into a context variable (so every
    log line carries it), into the problem-details body of any error, and back to
    the client in the X-Request-ID response header.

CORS
    A browser refuses to let a page on one origin (http://localhost:5173, the
    React dev server in session 15) read responses from another
    (http://127.0.0.1:8000) unless the server says it may. CORS_ORIGINS lists who
    may. Never allow_origins=["*"] together with allow_credentials=True: that
    lets any website on the internet call the API with the user's cookies.
"""
from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from shopwise.core.config import settings
from shopwise.core.logging_config import get_logger, request_id_var

log = get_logger("shopwise.access")


async def request_id_middleware(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    request_id_var.set(rid)           # for log lines written while this request runs
    request.state.request_id = rid    # for the error handlers (api/errors.py)
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    # For a streamed response this is the time to the first byte, not the whole stream.
    log.info("%s %s -> %s in %.0f ms [%s]", request.method, request.url.path,
             response.status_code, (time.perf_counter() - start) * 1000, rid)
    return response


def register_middleware(app: FastAPI) -> None:
    app.middleware("http")(request_id_middleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=False,             # no cookies yet; session 14 uses an API key header
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "Idempotency-Key", "X-Request-ID"],
        expose_headers=["X-Request-ID", "Location", "Idempotent-Replayed"],
    )
