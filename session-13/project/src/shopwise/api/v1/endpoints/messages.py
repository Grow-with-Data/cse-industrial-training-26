"""/v1/threads/{id}/messages — send a message; get the reply whole, or streamed as SSE.

TWO WAYS TO GET THE SAME REPLY
    POST .../messages          waits, then returns one MessageOut. Simple; what
                               scripts, integrations and tests use.
    POST .../messages/stream   Server-Sent Events: the reply arrives token by
                               token, with progress, tool and citation events
                               along the way. What a chat UI uses.

SSE IN ONE PARAGRAPH
    The response stays open and the server writes frames as things happen:

        event: token
        id: 7
        data: {"type":"token","text":"Your order"}

    One direction (server to client), plain HTTP, understood by proxies, with
    reconnect built into the browser's EventSource. A WebSocket is two-way and
    is overkill for "the server talks, the client listens". FastAPI's native SSE
    (`response_class=EventSourceResponse` on a generator route) also sends a
    keep-alive comment every 15 s and the headers proxies need not to buffer.

WHY THE STREAM IS STARTED IN A DEPENDENCY
    Once an SSE response begins, its status is already 200. A 404 or 409
    discovered inside the generator would be too late to send. So the checks
    (does the thread exist, is it busy, is it paused) and the start of the run
    happen in `started_stream`, a dependency, which FastAPI runs BEFORE the
    response begins. A failure there is a normal problem-details error.
"""
from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Body, Depends
from fastapi.sse import EventSourceResponse, ServerSentEvent

from shopwise.api.v1.examples import examples
from shopwise.api.v1.schemas.common import problem_responses
from shopwise.api.v1.schemas.messages import MessageIn, MessageOut
from shopwise.core.deps import ChatServiceDep
from shopwise.utils.streaming.events import Event

router = APIRouter(prefix="/threads", tags=["Messages"])

MessageBody = Annotated[MessageIn, Body(openapi_examples=examples("MessageIn"))]


@router.post(
    "/{thread_id}/messages",
    summary="Send a message and wait for the reply",
    description="Runs one turn and returns the whole reply with its citations. If the turn "
                "asks for a refund, it pauses: `status` is `awaiting_approval` and `interrupt` "
                "holds the form a manager answers with `POST /v1/threads/{id}/resume`.",
    responses=problem_responses(404, 409, 422, 502),
)
def send_message(thread_id: str, body: MessageBody, chat: ChatServiceDep) -> MessageOut:
    # A plain `def`: the turn blocks for seconds, and FastAPI runs a `def` route
    # in a worker thread, so other requests keep being served meanwhile.
    return chat.send(thread_id, body.text)


async def started_stream(thread_id: str, body: MessageBody,
                         chat: ChatServiceDep) -> AsyncIterator[tuple[int, Event]]:
    """Check, claim and START the turn, before any byte of the response is sent."""
    return chat.start_stream(thread_id, body.text)


SSE_DOC = {
    200: {
        "description": "A stream of Server-Sent Events. Each frame's `event:` is one of "
                       "`token`, `progress`, `tool`, `citation`, `interrupt`, `done`, `error`; "
                       "its `data:` is the JSON event (see the `Event` schemas). A turn ends with "
                       "`done`, `interrupt` (paused for a manager) or `error`.",
        "content": {"text/event-stream": {"schema": {"type": "string"}}},
    },
    **problem_responses(404, 409, 422),
}


@router.post(
    "/{thread_id}/messages/stream",
    response_class=EventSourceResponse,
    summary="Send a message and stream the reply (SSE)",
    responses=SSE_DOC,
)
async def stream_message(
    events: Annotated[AsyncIterator[tuple[int, Event]], Depends(started_stream)],
) -> AsyncIterable[ServerSentEvent]:
    async for n, event in events:
        # `id` lets a client say how far it got. We do not replay from it: a
        # client that lost the stream re-fetches GET .../messages instead.
        yield ServerSentEvent(data=event, event=event.type, id=str(n))
