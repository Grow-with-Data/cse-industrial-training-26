"""The thread table: who owns each conversation, its title, when it last moved. New in v10.

A repository is the only code that writes SQL for one table. Services call its
methods (create, get, page, touch, delete) and never see a cursor or a query,
so swapping SQLite for PostgreSQL later changes this file and nothing above it.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import UTC, datetime

from shopwise.db.db_manager import AppDatabase


def now() -> str:
    """UTC, ISO 8601, to the microsecond: sorts correctly as text."""
    return datetime.now(UTC).isoformat(timespec="microseconds")


@dataclass
class ThreadRow:
    id: str
    customer_email: str
    title: str | None
    created_at: str
    updated_at: str


def encode_cursor(row: ThreadRow) -> str:
    """The position after `row`, as an opaque string. Opaque on purpose: a
    client that cannot read the cursor cannot build one by hand, so the server
    may change what is inside without breaking anyone."""
    return base64.urlsafe_b64encode(f"{row.updated_at}|{row.id}".encode()).decode()


def decode_cursor(cursor: str) -> tuple[str, str]:
    """(updated_at, id) from a cursor. ValueError if it was not one of ours."""
    try:
        updated_at, thread_id = base64.urlsafe_b64decode(cursor.encode()).decode().split("|", 1)
    except Exception as exc:  # noqa: BLE001 — any garbage is the same mistake
        raise ValueError("not a valid cursor") from exc
    return updated_at, thread_id


class ThreadRepository:
    def __init__(self, db: AppDatabase) -> None:
        self.db = db

    def create(self, thread_id: str, customer_email: str) -> ThreadRow:
        stamp = now()
        self.db.execute(
            "INSERT INTO threads (id, customer_email, title, created_at, updated_at) "
            "VALUES (?, ?, NULL, ?, ?)", (thread_id, customer_email, stamp, stamp))
        return ThreadRow(thread_id, customer_email, None, stamp, stamp)

    def get(self, thread_id: str) -> ThreadRow | None:
        row = self.db.execute("SELECT * FROM threads WHERE id = ?", (thread_id,)).fetchone()
        return ThreadRow(**dict(row)) if row else None

    def page(self, limit: int, cursor: str | None = None) -> tuple[list[ThreadRow], str | None]:
        """Newest first, `limit` at a time. Returns (rows, next_cursor or None).

        Keyset pagination: "the rows after this (updated_at, id)", not "skip 40
        rows". With OFFSET, a thread updated while you page moves to the top and
        shifts every row below it, so page 3 repeats a row from page 2 or skips
        one. A cursor names a position, which does not move.
        """
        if cursor:
            updated_at, thread_id = decode_cursor(cursor)
            rows = self.db.execute(
                "SELECT * FROM threads WHERE (updated_at, id) < (?, ?) "
                "ORDER BY updated_at DESC, id DESC LIMIT ?", (updated_at, thread_id, limit + 1)).fetchall()
        else:
            rows = self.db.execute(
                "SELECT * FROM threads ORDER BY updated_at DESC, id DESC LIMIT ?", (limit + 1,)).fetchall()
        items = [ThreadRow(**dict(r)) for r in rows[:limit]]
        # We asked for one row more than the page: if it came back, there is a next page.
        next_cursor = encode_cursor(items[-1]) if len(rows) > limit else None
        return items, next_cursor

    def all(self) -> list[ThreadRow]:
        rows = self.db.execute("SELECT * FROM threads ORDER BY updated_at DESC").fetchall()
        return [ThreadRow(**dict(r)) for r in rows]

    def touch(self, thread_id: str, title: str | None = None) -> None:
        """The thread moved: bump updated_at, and set the title if it has none yet."""
        self.db.execute(
            "UPDATE threads SET updated_at = ?, title = COALESCE(title, ?) WHERE id = ?",
            (now(), title, thread_id))

    def delete(self, thread_id: str) -> None:
        self.db.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
