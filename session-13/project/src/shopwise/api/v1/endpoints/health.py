"""/health and /ready — is the process alive, and can it do its job?

Mounted WITHOUT the /v1 prefix: health checks are for machines (a load balancer,
session 16's deploy script, a monitor), not for API clients, and they must not
move when the API's version does.

    /health   liveness: the process answers. Nothing else is checked, so it is
              cheap enough to call every few seconds. If it fails, restart.
    /ready    readiness: the graph is built and the databases answer. If it
              fails, keep the process but send it no traffic yet.

Neither calls a model: a health check that spends tokens is a bill on a timer.
"""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from shopwise.api.v1.schemas.common import problem_responses
from shopwise.api.v1.schemas.health import Health, Readiness
from shopwise.core.config import settings

router = APIRouter(tags=["Health"])


@router.get("/health", summary="Liveness")
def health() -> Health:
    return Health()


@router.get(
    "/ready",
    summary="Readiness",
    responses={503: {"model": Readiness, "description": "Not ready yet."}, **problem_responses(500)},
)
def ready(request: Request) -> Readiness:
    state = request.app.state
    resources = getattr(state, "resources", None)
    checks = {
        "graph": resources is not None and resources.graph is not None,
        "app_db": resources is not None and resources.app_db.ping(),
        "knowledge_base": settings.QDRANT_PATH.exists(),
    }
    body = Readiness(status="ready" if all(checks.values()) else "not_ready",
                     checks=checks, version=settings.VERSION)
    if body.status != "ready":
        return JSONResponse(body.model_dump(), status_code=503)
    return body
