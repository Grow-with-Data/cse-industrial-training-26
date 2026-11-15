"""Every error a client can see, as RFC 9457 problem details. New in v10.

    HTTP/1.1 404 Not Found
    Content-Type: application/problem+json

    {"type": "https://shopwise.example/problems/thread-not-found",
     "title": "There is no thread with that id.", "status": 404,
     "detail": "No thread 3f2b….", "instance": "/v1/threads/3f2b…",
     "request_id": "9c1e0b7a4d"}

WHY A STANDARD SHAPE
    Without one, every endpoint invents its own: {"error": "..."} here,
    {"message": "..."} there, a bare string from the framework's 404. A client has
    to special-case each. RFC 9457 is the IETF's answer: one shape, one media type.
    `type` is the field a program branches on; it never changes once published.

FOUR HANDLERS COVER EVERYTHING
    ShopWiseError             our deliberate errors: status, type and title come
                              from the class (core/exceptions.py)
    RequestValidationError    FastAPI's 422 for a bad body or parameter, reshaped:
                              `errors` lists each bad field and why
    StarletteHTTPException    the framework's own 404 (no such route), 405...
    Exception                 a bug. Log the traceback HERE, on the server, and
                              send the client a generic 500 with the request id.
                              Never the traceback: it can carry file paths,
                              hosts, keys, customer data.
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from shopwise.core.exceptions import ShopWiseError
from shopwise.core.logging_config import get_logger, request_id_var

log = get_logger(__name__)

PROBLEM_BASE = "https://shopwise.example/problems/"
HTTP_TITLES = {404: "Not found.", 405: "Method not allowed.", 415: "Unsupported media type."}


def problem(request: Request, status: int, error_type: str, title: str, detail: str,
            errors: list[dict] | None = None, headers: dict | None = None) -> JSONResponse:
    body = {"type": PROBLEM_BASE + error_type, "title": title, "status": status,
            "detail": detail, "instance": request.url.path,
            "request_id": getattr(request.state, "request_id", request_id_var.get())}
    if errors is not None:
        body["errors"] = errors
    return JSONResponse(body, status_code=status, media_type="application/problem+json",
                        headers=headers)


async def shopwise_error(request: Request, exc: ShopWiseError) -> JSONResponse:
    if exc.status_code >= 500:
        log.error("%s: %s", type(exc).__name__, exc.detail)
    return problem(request, exc.status_code, exc.error_type, exc.title, exc.detail)


async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = [{"field": ".".join(str(p) for p in e["loc"]), "message": e["msg"]}
              for e in exc.errors()]
    return problem(request, 422, "validation-error", "The request is not valid.",
                   f"{len(errors)} field(s) failed validation.", errors=errors)


async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    title = HTTP_TITLES.get(exc.status_code, "HTTP error.")
    detail = exc.detail if isinstance(exc.detail, str) else title
    return problem(request, exc.status_code, f"http-{exc.status_code}", title, detail,
                   headers=getattr(exc, "headers", None))


async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled error on %s %s", request.method, request.url.path)
    return problem(request, 500, "internal-error", "Something went wrong on our side.",
                   "The error was logged. Quote the request_id if you report it.")


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ShopWiseError, shopwise_error)
    app.add_exception_handler(RequestValidationError, validation_error)
    app.add_exception_handler(StarletteHTTPException, http_error)
    app.add_exception_handler(Exception, unhandled_error)
