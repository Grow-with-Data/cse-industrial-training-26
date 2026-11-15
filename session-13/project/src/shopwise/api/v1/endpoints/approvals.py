"""/v1/approvals and /v1/threads/{id}/resume — the manager's inbox and the decision.

THE CYCLE
    1. A message asks for a refund. The turn pauses; the thread is awaiting_approval.
    2. GET /v1/approvals lists every paused request, across every thread.
    3. POST /v1/threads/{id}/resume with approve, edit or reject, and an
       Idempotency-Key header. The thread runs on, pays (or not) and replies.

THE IDEMPOTENCY-KEY HEADER
    Required on /resume. Send a fresh UUID per decision and the SAME one on every
    retry of that decision. A retry then gets the first response back, with
    `Idempotent-Replayed: true`, and the thread is not resumed twice.
    services/idempotency_service.py has the four cases.
"""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, Header, Response

from shopwise.api.v1.examples import examples
from shopwise.api.v1.schemas.approvals import ApprovalList, Decision
from shopwise.api.v1.schemas.common import problem_responses
from shopwise.api.v1.schemas.messages import MessageOut
from shopwise.core.deps import ApprovalServiceDep, IdempotencyServiceDep

router = APIRouter(tags=["Approvals"])


@router.get(
    "/approvals",
    summary="List pending approvals",
    description="Every refund waiting for a manager, across all threads.",
)
def list_approvals(approvals: ApprovalServiceDep) -> ApprovalList:
    return approvals.pending()


@router.post(
    "/threads/{thread_id}/resume",
    summary="Approve, edit or reject a paused refund",
    description="Resumes the thread with the manager's decision and returns the reply that "
                "follows. Needs an `Idempotency-Key` header: a retry with the same key and body "
                "returns the stored response without resuming again.",
    responses=problem_responses(400, 404, 409, 422, 502),
)
def resume_thread(
    thread_id: str,
    decision: Annotated[Decision, Body(openapi_examples=examples("Decision"))],
    approvals: ApprovalServiceDep,
    idempotency: IdempotencyServiceDep,
    response: Response,
    idempotency_key: Annotated[str | None, Header(
        alias="Idempotency-Key", min_length=8, max_length=128,
        description="A fresh UUID per decision; the same one on every retry of it.")] = None,
) -> MessageOut:
    result, replayed = idempotency.run(
        idempotency_key, scope=thread_id, body=decision, model=MessageOut,
        action=lambda: approvals.resume(thread_id, decision))
    if replayed:
        response.headers["Idempotent-Replayed"] = "true"
    return result
