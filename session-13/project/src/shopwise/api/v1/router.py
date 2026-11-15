"""Every /v1 endpoint, registered in one place.

Adding a feature: write api/v1/endpoints/<feature>.py with `router = APIRouter()`,
then add one line here. main.py mounts this router under settings.API_PREFIX.
Health is not here: it is unversioned and mounted by main.py directly.
"""
from __future__ import annotations

from fastapi import APIRouter

from shopwise.api.v1.endpoints import approvals, messages, threads

api_router = APIRouter()
api_router.include_router(threads.router)
api_router.include_router(messages.router)
api_router.include_router(approvals.router)
