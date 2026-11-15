"""The idempotency-key table: one row per decision a client has sent. New in v10.

    start(key, hash)      claim the key; False if someone already has it
    get(key)              the stored row: its body hash, status and response
    finish(key, ...)      store the response, so a retry can be answered from it
    release(key)          the request failed: forget the key, the client may retry

services/idempotency_service.py decides what each case MEANS; this file only
stores rows.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass

from shopwise.db.db_manager import AppDatabase


@dataclass
class KeyRow:
    key: str
    request_hash: str
    status: str                      # "in_progress" | "done"
    response_code: int | None
    response_body: str | None
    created_at: float


class IdempotencyRepository:
    def __init__(self, db: AppDatabase, ttl_s: int) -> None:
        self.db = db
        self.ttl_s = ttl_s

    def purge_expired(self) -> None:
        self.db.execute("DELETE FROM idempotency_keys WHERE created_at < ?", (time.time() - self.ttl_s,))

    def get(self, key: str) -> KeyRow | None:
        self.purge_expired()
        row = self.db.execute("SELECT * FROM idempotency_keys WHERE key = ?", (key,)).fetchone()
        return KeyRow(**dict(row)) if row else None

    def start(self, key: str, request_hash: str) -> bool:
        """Claim the key. The PRIMARY KEY makes this atomic: of two requests that
        arrive together with the same key, exactly one INSERT succeeds."""
        try:
            self.db.execute(
                "INSERT INTO idempotency_keys (key, request_hash, status, created_at) "
                "VALUES (?, ?, 'in_progress', ?)", (key, request_hash, time.time()))
            return True
        except sqlite3.IntegrityError:
            return False

    def finish(self, key: str, response_code: int, response_body: str) -> None:
        self.db.execute(
            "UPDATE idempotency_keys SET status = 'done', response_code = ?, response_body = ? "
            "WHERE key = ?", (response_code, response_body, key))

    def release(self, key: str) -> None:
        self.db.execute("DELETE FROM idempotency_keys WHERE key = ?", (key,))
