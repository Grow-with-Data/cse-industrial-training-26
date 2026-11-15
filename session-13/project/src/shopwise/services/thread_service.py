"""Threads as resources: create, read, list, delete, and what state each one is in. New in v10.

A service is one class per area of the product, with one method per thing a
user can do. Routes call services; services call the graph and the repositories.
A service knows nothing about HTTP: it raises ShopWiseError subclasses
(ThreadNotFound...) and api/errors.py turns them into status codes. That is what
lets a test, a CLI or a background job call the same method a route does.

THE SERVER MINTS THE ID
    A thread id is a uuid4 made here. If a client could choose it, a client could
    guess someone else's ("CHAT-0001", "CHAT-0002"...) and read their refund.

"RUNNING" LIVES IN THIS PROCESS
    Which threads are mid-turn is a set in memory, guarded by a lock. Good enough
    for one server process. Session 16 runs several worker processes, each with
    its own set; a shared store (a database row, Redis) would be the fix then.
"""
from __future__ import annotations

import threading
import uuid
from contextlib import contextmanager

from shopwise.api.v1.schemas.threads import HistoryMessage, ThreadHistory, ThreadList, ThreadOut
from shopwise.core.exceptions import CustomerNotFound, ThreadBusy, ThreadNotFound
from shopwise.core.logging_config import get_logger
from shopwise.db.threads import ThreadRepository, ThreadRow
from shopwise.llm.middleware import ShopContext, context_for
from shopwise.utils.streaming.events import paused_on, run_config

log = get_logger(__name__)


class ThreadService:
    def __init__(self, graph, saver, threads: ThreadRepository) -> None:
        self.graph = graph
        self.saver = saver
        self.threads = threads
        self._running: set[str] = set()
        self._lock = threading.Lock()

    # ── the resource ─────────────────────────────────────────────────────────

    def create(self, customer_email: str) -> ThreadOut:
        self.customer_context(customer_email)            # unknown customer -> 422, now
        row = self.threads.create(str(uuid.uuid4()), customer_email.lower())
        log.info("thread created %s for %s", row.id, row.customer_email)
        return self.to_out(row)

    def get(self, thread_id: str) -> ThreadOut:
        return self.to_out(self.row(thread_id))

    def page(self, limit: int, cursor: str | None) -> ThreadList:
        rows, next_cursor = self.threads.page(limit, cursor)
        return ThreadList(items=[self.to_out(r) for r in rows], next_cursor=next_cursor)

    def history(self, thread_id: str) -> ThreadHistory:
        """The conversation, mapped to the API's shape. Never the raw graph state:
        state is internal (it holds triage results, findings, routing logs) and
        will change; HistoryMessage is a promise."""
        self.row(thread_id)
        values = self.graph.get_state(run_config(thread_id)).values
        messages = [HistoryMessage(role="customer" if m.type == "human" else "assistant", text=m.text)
                    for m in values.get("messages", []) if m.type in ("human", "ai") and m.text]
        return ThreadHistory(thread_id=thread_id, summary=values.get("summary") or None,
                             messages=messages)

    def delete(self, thread_id: str) -> None:
        """Remove the thread and every checkpoint it saved: the customer's right to
        be forgotten (session 11). Their long-term memories are a separate store,
        deleted per customer, not per thread."""
        self.row(thread_id)
        if self.is_running(thread_id):
            raise ThreadBusy("Wait for the current turn to finish, then delete the thread.")
        self.saver.delete_thread(thread_id)
        self.threads.delete(thread_id)
        log.info("thread deleted %s", thread_id)

    # ── state ────────────────────────────────────────────────────────────────

    def row(self, thread_id: str) -> ThreadRow:
        row = self.threads.get(thread_id)
        if row is None:
            raise ThreadNotFound(f"No thread {thread_id}.")
        return row

    def status(self, thread_id: str) -> str:
        if self.is_running(thread_id):
            return "running"
        if paused_on(self.graph, thread_id):
            return "awaiting_approval"
        return "idle"

    def to_out(self, row: ThreadRow) -> ThreadOut:
        return ThreadOut(id=row.id, customer_email=row.customer_email, title=row.title,
                         status=self.status(row.id), created_at=row.created_at,
                         updated_at=row.updated_at)

    def is_running(self, thread_id: str) -> bool:
        with self._lock:
            return thread_id in self._running

    def claim(self, thread_id: str) -> None:
        """Mark the thread as mid-turn, or refuse: one turn at a time per thread.
        A second message while the first is being answered would start a second
        run on the same checkpoint (LangGraph calls this "double-texting")."""
        with self._lock:
            if thread_id in self._running:
                raise ThreadBusy()
            self._running.add(thread_id)

    def release(self, thread_id: str) -> None:
        with self._lock:
            self._running.discard(thread_id)

    @contextmanager
    def turn(self, thread_id: str):
        self.claim(thread_id)
        try:
            yield
        finally:
            self.release(thread_id)

    # ── who is acting ────────────────────────────────────────────────────────

    @staticmethod
    def customer_context(email: str, role: str = "customer") -> ShopContext:
        try:
            return context_for(email.lower(), role=role)
        except KeyError as exc:
            raise CustomerNotFound(f"No ShopWise customer with email {email!r}.") from exc
