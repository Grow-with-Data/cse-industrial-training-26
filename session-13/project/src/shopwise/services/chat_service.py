"""Sending a message: wait for the whole reply, or stream it as it is written. New in v10.

    send(thread_id, text)          -> MessageOut, when the turn is over
    start_stream(thread_id, text)  -> async iterator of (n, Event), starting now

Both run ONE function, run_turn(), which is a plain generator over v9's
stream_ticket(). The difference is only who reads it: send() reads it to the end
in the calling thread (FastAPI already put a `def` route in a worker thread);
start_stream() hands it to a thread of its own and lets async code read it as
events arrive. The graph does not know an HTTP request exists.

THE EVENT VOCABULARY CROSSES THE WIRE UNCHANGED
    stream_ticket() yields session 11's seven events: token, progress, tool,
    citation, interrupt, done, error. The streaming route sends each one as an SSE
    frame whose `event:` is its type; send() folds the same events into one
    MessageOut. Session 15's React page switches on the same seven names.
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Iterator

from langgraph.types import Command

from shopwise.api.v1.schemas.messages import ApprovalRequest, CitationOut, MessageOut
from shopwise.core.exceptions import ThreadAwaitingApproval, TurnFailed
from shopwise.core.logging_config import get_logger
from shopwise.llm.middleware import ShopContext
from shopwise.services.thread_service import ThreadService
from shopwise.utils.streaming.bridge import start_in_thread
from shopwise.utils.streaming.events import Event, paused_on, stream_ticket

log = get_logger(__name__)

TITLE_LENGTH = 60


def title_from(text: str) -> str:
    """A thread's title: its first message, on one line, shortened."""
    line = " ".join(text.split())
    return line if len(line) <= TITLE_LENGTH else line[: TITLE_LENGTH - 1].rstrip() + "…"


class ChatService:
    def __init__(self, threads: ThreadService) -> None:
        self.threads = threads

    @property
    def graph(self):
        return self.threads.graph

    # ── the one generator both paths share ───────────────────────────────────

    def run_turn(self, thread_id: str, turn: str | Command, ctx: ShopContext,
                 title: str | None = None) -> Iterator[Event]:
        """One turn, as events. The caller must have claimed the thread
        (ThreadService.claim); this releases it when the turn ends, however it ends."""
        try:
            yield from stream_ticket(self.graph, thread_id, turn, ctx)
        finally:
            self.threads.threads.touch(thread_id, title=title)
            self.threads.release(thread_id)
            log.info("turn finished on %s", thread_id)

    def prepare(self, thread_id: str) -> ShopContext:
        """Everything that must be true before a customer's message runs. Raises
        the matching ShopWiseError, so the route answers 404 or 409, not a 200
        that turns out to be an error halfway through a stream."""
        row = self.threads.row(thread_id)                       # 404
        if paused_on(self.graph, thread_id):                    # 409
            raise ThreadAwaitingApproval()
        ctx = self.threads.customer_context(row.customer_email)
        self.threads.claim(thread_id)                           # 409 if busy
        return ctx

    # ── wait for the reply ───────────────────────────────────────────────────

    def send(self, thread_id: str, text: str) -> MessageOut:
        ctx = self.prepare(thread_id)
        return self.collect(thread_id, self.run_turn(thread_id, text, ctx, title=title_from(text)))

    def collect(self, thread_id: str, events: Iterable[Event]) -> MessageOut:
        """Fold a turn's events into one response."""
        reply, citations, interrupt = "", [], None
        for event in events:
            match event.type:
                case "citation":
                    citations.append(CitationOut(**event.model_dump(exclude={"type"})))
                case "interrupt" if interrupt is None:
                    interrupt = ApprovalRequest(interrupt_id=event.id, **event.payload)
                case "done":
                    reply = event.reply
                case "error":
                    # Already safe to show (events.py never puts exception text in it).
                    raise TurnFailed(event.message)
        status = "awaiting_approval" if interrupt else "idle"
        return MessageOut(thread_id=thread_id, reply=reply, citations=citations,
                          status=status, interrupt=interrupt)

    # ── stream the reply ─────────────────────────────────────────────────────

    def start_stream(self, thread_id: str, text: str) -> AsyncIterator[tuple[int, Event]]:
        """Check, claim, and start the turn in its own thread NOW. Returns an async
        iterator of (sequence number, event). Call it from async code."""
        ctx = self.prepare(thread_id)
        events = start_in_thread(lambda: self.run_turn(thread_id, text, ctx, title=title_from(text)))
        return self._numbered(events)

    @staticmethod
    async def _numbered(events: AsyncIterator[Event]) -> AsyncIterator[tuple[int, Event]]:
        n = 0
        async for event in events:
            n += 1
            yield n, event
