"""FastAPI dependencies: how a route gets the things it needs. New in v10.

    @router.get("/{thread_id}")
    def get_thread(thread_id: str, threads: ThreadServiceDep) -> ThreadOut:
        return threads.get(thread_id)

WHAT A DEPENDENCY IS
    A function FastAPI calls for you before your route runs. You declare a
    parameter typed `Annotated[ThreadService, Depends(get_thread_service)]`, and
    FastAPI calls get_thread_service(request) and passes the result in. The route
    never builds a service, opens a connection or reads app.state itself.

    The aliases below (ThreadServiceDep...) are the modern idiom: write the
    Annotated[...] once, here, and every route says just `threads: ThreadServiceDep`.

WHERE THE OBJECTS COME FROM
    main.py's lifespan builds every service once, at startup, and puts it on
    app.state. These functions only hand them out. Nothing is created per request.

WHY THIS MATTERS LATER
    app.dependency_overrides[get_chat_service] = lambda: fake_service
    swaps the real service for a fake in one line, without touching a route.
    Session 14 adds the API-key check as one more dependency, the same way.
"""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from shopwise.core.config import Settings, get_settings
from shopwise.services.approval_service import ApprovalService
from shopwise.services.chat_service import ChatService
from shopwise.services.idempotency_service import IdempotencyService
from shopwise.services.thread_service import ThreadService


def get_thread_service(request: Request) -> ThreadService:
    return request.app.state.thread_service


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service


def get_approval_service(request: Request) -> ApprovalService:
    return request.app.state.approval_service


def get_idempotency_service(request: Request) -> IdempotencyService:
    return request.app.state.idempotency_service


SettingsDep = Annotated[Settings, Depends(get_settings)]
ThreadServiceDep = Annotated[ThreadService, Depends(get_thread_service)]
ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
ApprovalServiceDep = Annotated[ApprovalService, Depends(get_approval_service)]
IdempotencyServiceDep = Annotated[IdempotencyService, Depends(get_idempotency_service)]
