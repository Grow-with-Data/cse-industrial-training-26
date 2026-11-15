"""The ShopWise API: where the FastAPI app is created. New in v10.

    uv run fastapi dev                     # http://127.0.0.1:8000, reloads on save
    uv run uvicorn shopwise.main:app --reload    # the same, spelled out

    http://127.0.0.1:8000/scalar    the API reference (Scalar): read and try every endpoint
    http://127.0.0.1:8000/docs      the same, as Swagger UI
    http://127.0.0.1:8000/redoc     the same, as ReDoc
    http://127.0.0.1:8000/health    {"status": "ok"}

WHAT HAPPENS AT STARTUP, IN ORDER
    1. Python imports this module and runs create_app(): routes, error handlers
       and middleware are registered. Nothing is connected yet.
    2. The server calls `lifespan` once, before the first request. It opens the
       databases, builds the graph, builds the services and puts them on
       app.state. A request can only arrive after this.
    3. Every request borrows those objects through core/deps.py.
    4. On shutdown (Ctrl-C), the code after `yield` in lifespan runs: every
       connection is closed.

THE LAYERS (a request goes down, the answer comes back up)
    api/v1/endpoints/   HTTP in and out: parse, validate, status codes
    services/           one method per thing a user can do
    llm/workflows/      the v9 LangGraph team, unchanged in behaviour
    db/                 the API's own tables, and every open connection
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.routing import APIRoute
from scalar_fastapi import get_scalar_api_reference

from shopwise.api.errors import register_exception_handlers
from shopwise.api.middleware import register_middleware
from shopwise.api.v1.endpoints import health
from shopwise.api.v1.router import api_router
from shopwise.core.config import settings
from shopwise.core.logging_config import get_logger, setup_logging
from shopwise.db.db_manager import open_resources
from shopwise.db.idempotency import IdempotencyRepository
from shopwise.db.threads import ThreadRepository
from shopwise.services.approval_service import ApprovalService
from shopwise.services.chat_service import ChatService
from shopwise.services.idempotency_service import IdempotencyService
from shopwise.services.thread_service import ThreadService

log = get_logger(__name__)

DESCRIPTION = """
ShopWise's customer-support assistant as an HTTP API.

A **thread** is one customer's conversation. Send it **messages** and get replies,
whole or streamed as Server-Sent Events. When a reply needs a refund, the thread
pauses; a manager finds it under **approvals** and resumes it with a decision.

Every error is `application/problem+json` (RFC 9457): branch on `type`, show `detail`,
quote `request_id` when reporting a problem.

**Not yet secured.** Anyone who can reach this API can read any thread and approve
any refund. Session 14 adds API keys.
"""

TAGS = [
    {"name": "Threads", "description": "Conversations: create, list, read, delete."},
    {"name": "Messages", "description": "Send a message; get the reply whole or as an SSE stream."},
    {"name": "Approvals", "description": "Refunds waiting for a manager, and the decision that resumes them."},
    {"name": "Health", "description": "Liveness and readiness, for machines. Unversioned."},
]


def operation_id(route: APIRoute) -> str:
    """`threads-create_thread` rather than FastAPI's default
    `create_thread_v1_threads_post`. Session 15 generates a TypeScript client
    from these names, so they are worth keeping readable."""
    return f"{route.tags[0].lower()}-{route.name}" if route.tags else route.name


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    with open_resources() as res:
        threads = ThreadService(res.graph, res.saver, ThreadRepository(res.app_db))
        chat = ChatService(threads)
        app.state.resources = res
        app.state.thread_service = threads
        app.state.chat_service = chat
        app.state.approval_service = ApprovalService(chat)
        app.state.idempotency_service = IdempotencyService(
            IdempotencyRepository(res.app_db, settings.IDEMPOTENCY_TTL_S))
        log.info("%s v%s ready · model %s", settings.APP_NAME, settings.VERSION, settings.model_id)
        yield
    log.info("shut down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="ShopWise API",
        version=settings.VERSION,
        description=DESCRIPTION,
        openapi_tags=TAGS,
        lifespan=lifespan,
        generate_unique_id_function=operation_id,
    )
    register_middleware(app)
    register_exception_handlers(app)
    app.include_router(health.router)                               # /health, /ready
    app.include_router(api_router, prefix=settings.API_PREFIX)      # /v1/...

    @app.get("/scalar", include_in_schema=False)
    def scalar():
        """The API reference, rendered by Scalar from /openapi.json."""
        return get_scalar_api_reference(openapi_url=app.openapi_url, title=app.title)

    return app


app = create_app()