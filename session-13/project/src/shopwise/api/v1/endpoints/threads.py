"""/v1/threads — a conversation is a resource: create it, read it, list it, delete it.

Every route here is a plain `def`, not `async def`. FastAPI runs a `def` route
in a worker thread, so its blocking SQLite reads cannot stall the event loop.
Use `async def` only when everything inside it is awaited (see messages.py).

Each route does three things and no more: take the HTTP input (validated by its
type), call one service method, return the API schema. No business logic here.
"""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, Query, Response, status

from shopwise.api.v1.examples import examples
from shopwise.api.v1.schemas.common import problem_responses
from shopwise.api.v1.schemas.threads import ThreadCreate, ThreadHistory, ThreadList, ThreadOut
from shopwise.core.config import settings
from shopwise.core.deps import ThreadServiceDep
from shopwise.core.exceptions import ShopWiseError

router = APIRouter(prefix="/threads", tags=["Threads"])


class BadCursor(ShopWiseError):
    status_code = 422
    error_type = "invalid-cursor"
    title = "The cursor is not one this server issued."


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    summary="Create a thread",
    description="Starts a conversation for one customer. The server mints the id and returns "
                "it in the body and in the `Location` header.",
    responses=problem_responses(422),
    # response_model=ThreadOut,
)
def create_thread(
    body: Annotated[ThreadCreate, Body(openapi_examples=examples("ThreadCreate"))],
    threads: ThreadServiceDep,
    response: Response,
)-> ThreadOut:
    thread = threads.create(body.customer_email)
    response.headers["Location"] = f"{settings.API_PREFIX}/threads/{thread.id}"
    return thread


@router.get(
    "",
    summary="List threads",
    description="Newest activity first. Pass `next_cursor` from one page as `cursor` to get "
                "the next. Cursor pagination does not skip or repeat threads that move while "
                "you page, which offset pagination does.",
    responses=problem_responses(422),
)
def list_threads(
    threads: ThreadServiceDep,
    limit: Annotated[int, Query(ge=1, le=settings.PAGE_SIZE_MAX)] = 20,
    cursor: Annotated[str | None, Query(description="Opaque: from a previous page.")] = None,
) -> ThreadList:
    try:
        return threads.page(limit, cursor)
    except ValueError as exc:
        raise BadCursor() from exc


@router.get(
    "/{thread_id}",
    summary="Get one thread",
    description="Metadata and status: `idle`, `running` or `awaiting_approval`.",
    responses=problem_responses(404),
)
def get_thread(thread_id: str, threads: ThreadServiceDep) -> ThreadOut:
    return threads.get(thread_id)


@router.get(
    "/{thread_id}/messages",
    summary="Get a thread's history",
    description="The conversation so far, customer and assistant turns in order. Also how a "
                "client recovers a reply after its stream was cut off.",
    responses=problem_responses(404),
)
def get_history(thread_id: str, threads: ThreadServiceDep) -> ThreadHistory:
    return threads.history(thread_id)


@router.delete(
    "/{thread_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,          # 204 means "no body": not even an empty JSON one
    summary="Delete a thread",
    description="Removes the thread and every checkpoint it saved. Not the customer's "
                "long-term memories: those belong to the customer, not to one thread.",
    responses=problem_responses(404, 409),
)
def delete_thread(thread_id: str, threads: ThreadServiceDep) -> None:
    threads.delete(thread_id)
