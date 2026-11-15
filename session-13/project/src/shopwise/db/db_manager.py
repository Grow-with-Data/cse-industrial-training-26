"""Everything that holds a database connection, opened once and closed once. New in v10.

    with open_resources() as res:        # main.py's lifespan does this
        res.graph          # the compiled v9 team, with checkpoints and memory on disk
        res.saver          # LangGraph's checkpointer (CHECKPOINT_DB)
        res.app_db         # the API's own tables (APP_DB): threads, idempotency keys

THREE FILES, THREE JOBS
    CHECKPOINT_DB   LangGraph's: every step of every thread. Written by the graph.
    MEMORY_DB       LangGraph's Store: what ShopWise knows about each customer.
    APP_DB          the API's: a row per thread (owner, title, dates) and a row
                    per idempotency key. Questions a web app asks all the time
                    ("my threads, newest first") that a checkpoint file is not
                    built to answer.

WHY ONCE, AT STARTUP
    A connection is expensive to open and must be closed. Opening the checkpointer
    inside a route would open a file per request; building the graph inside a
    route would rebuild the whole team per request and, worse, a checkpointer
    created and dropped per request remembers nothing between them. So the
    server opens everything here, at startup, and every request borrows it.

WHY SYNC sqlite3, NOT aiosqlite
    The v9 graph is synchronous: its nodes, its tools and its checkpointer.
    Rewriting every node as `async def` would be a large change for no gain to
    a reader learning FastAPI, so v10 keeps the graph synchronous and runs it in
    a worker thread (utils/streaming/bridge.py). check_same_thread=False lets
    those threads share one connection; a lock serialises writes to APP_DB.
"""
from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

from shopwise.core.config import settings
from shopwise.core.logging_config import get_logger

log = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS threads (
    id              TEXT PRIMARY KEY,
    customer_email  TEXT NOT NULL,
    title           TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS threads_by_update ON threads (updated_at DESC, id DESC);

CREATE TABLE IF NOT EXISTS idempotency_keys (
    key           TEXT PRIMARY KEY,
    request_hash  TEXT NOT NULL,
    status        TEXT NOT NULL CHECK (status IN ('in_progress', 'done')),
    response_code INTEGER,
    response_body TEXT,
    created_at    REAL NOT NULL
);
"""


class AppDatabase:
    """The API's own SQLite file. One connection, shared by worker threads."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row          # rows behave like dicts: row["title"]
        self.conn.execute("PRAGMA journal_mode=WAL")  # readers do not block the writer
        self.conn.executescript(SCHEMA)
        self.lock = threading.Lock()

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self.lock:
            return self.conn.execute(sql, params)

    def ping(self) -> bool:
        """Can we still run a query? Used by GET /ready."""
        try:
            self.execute("SELECT 1").fetchone()
            return True
        except sqlite3.Error:
            return False

    def close(self) -> None:
        self.conn.close()


@dataclass
class Resources:
    graph: object            # the compiled supervisor graph (CompiledStateGraph)
    saver: object            # SqliteSaver
    store: object            # SqliteStore
    app_db: AppDatabase


@contextmanager
def open_resources() -> Iterator[Resources]:
    """Open every connection, build the graph once, and close it all on exit.

    ExitStack closes in reverse order, even when something in the middle fails
    to open: a half-started server does not leave files locked behind it.
    """
    # Imported here, not at the top: building the team imports every
    # specialist, which a module that only wants AppDatabase should not pay for.
    from shopwise.llm.workflows.inbox.graph import open_saver
    from shopwise.llm.workflows.memory import open_memory
    from shopwise.llm.workflows.supervisor import build_team

    with ExitStack() as stack:
        saver = stack.enter_context(open_saver())
        store = stack.enter_context(open_memory())
        app_db = AppDatabase(settings.APP_DB)
        stack.callback(app_db.close)
        graph = build_team(checkpointer=saver, store=store)
        log.info("graph built; checkpoints %s, app db %s", settings.CHECKPOINT_DB, settings.APP_DB)
        yield Resources(graph=graph, saver=saver, store=store, app_db=app_db)
    log.info("connections closed")
