"""Shapes every endpoint shares: a thread's status and the problem-details error body."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ThreadStatus = Literal["idle", "running", "awaiting_approval"]
"""idle: ready for a message · running: answering one now ·
awaiting_approval: paused until a manager decides a refund."""


class FieldError(BaseModel):
    """One invalid field in a 422, in a shape a form can show next to the input."""

    field: str = Field(examples=["body.text"])
    message: str = Field(examples=["String should have at most 4000 characters"])


class Problem(BaseModel):
    """An error, as RFC 9457 "problem details" (`application/problem+json`).

    Branch on `type`; it never changes. Show `detail` to a person; its wording may.
    Quote `request_id` when reporting a bug: the server's log has every line that
    request wrote under the same id.
    """

    type: str = Field(description="A stable URI naming the kind of error.",
                      examples=["https://shopwise.example/problems/thread-not-found"])
    title: str = Field(examples=["There is no thread with that id."])
    status: int = Field(examples=[404])
    detail: str = Field(examples=["No thread 3f2b…"])
    instance: str = Field(description="The path that was requested.", examples=["/v1/threads/3f2b…"])
    request_id: str = Field(examples=["9c1e0b7a4d"])
    errors: list[FieldError] | None = Field(default=None, description="Present on 422 only.")


def problem_responses(*codes: int) -> dict[int | str, dict]:
    """The `responses=` entry for a route: every error status it can return,
    documented with the Problem body, so /docs and the client generator know."""
    return {code: {"model": Problem, "content": {"application/problem+json": {}}} for code in codes}
