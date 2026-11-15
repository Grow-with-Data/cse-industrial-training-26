"""Run a synchronous generator in a worker thread and read it from async code. New in v10.

    events = start_in_thread(lambda: stream_ticket(graph, tid, text, ctx))   # starts now
    async for event in events:
        yield event

THE PROBLEM
    FastAPI runs every `async def` route on ONE event loop. While a route's code
    is running, no other request on that worker makes progress. Awaiting is fine
    (the loop serves someone else meanwhile); a blocking call is not. The v9
    graph is synchronous: one turn is several seconds of blocking network calls.
    Run it directly inside `async def` and every other user waits for this
    customer's refund.

THE FIX
    Start the generator in its own thread. The thread blocks; the event loop does
    not. Each event the thread produces is handed to the loop through an
    asyncio.Queue (call_soon_threadsafe is the one safe way to touch a loop from
    another thread), and the async side awaits the queue.

    Why one dedicated thread, and not starlette's iterate_in_threadpool? That
    helper runs each next() on whichever pool thread is free. LangGraph keeps
    per-run state in context variables, which belong to the thread (and the
    context) that set them, so a run should stay on one thread from start to end.

WHY IT STARTS AT ONCE
    start_in_thread starts the thread when it is CALLED, not when someone first
    reads from it. The chat service marks a thread "running" before calling it
    and the run clears the mark when it ends. A thread that started lazily, on
    first read, might never start if the client hung up in between, and the mark
    would never clear: the thread would say "running" forever.

IF THE CLIENT GOES AWAY
    The async side stops reading; the thread does NOT stop. The turn finishes,
    the checkpointer saves it, and the client can fetch the reply later from
    GET /v1/threads/{id}/messages. Stopping a run half-way through a refund is
    worse than finishing it with nobody watching.
"""
from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from typing import TypeVar

T = TypeVar("T")
_DONE = object()          # the sentinel that says "the generator has finished"


def start_in_thread(make: Callable[[], Iterator[T]]) -> AsyncIterator[T]:
    """Start `make()` in a new thread now; return an async iterator over what it yields.
    Must be called from async code (it needs the running event loop)."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def put(item) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, item)
        except RuntimeError:
            pass          # the loop is closed (server shutting down): nobody is listening

    def produce() -> None:
        try:
            for item in make():
                put(item)
        except BaseException as exc:  # noqa: BLE001 — re-raised on the async side
            put(exc)
        finally:
            put(_DONE)

    threading.Thread(target=produce, name="graph-run", daemon=True).start()

    async def consume() -> AsyncIterator[T]:
        while True:
            item = await queue.get()
            if item is _DONE:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    return consume()
