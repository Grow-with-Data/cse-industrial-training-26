"""HTTP shapes for the manager's side: pending approvals and the decision that resumes a thread."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from shopwise.api.v1.schemas.messages import ApprovalRequest


class PendingApproval(BaseModel):
    thread_id: str
    customer_email: str
    request: ApprovalRequest


class ApprovalList(BaseModel):
    items: list[PendingApproval]


class Decision(BaseModel):
    """A manager's answer. The graph checks it again (an amount above what was
    paid is refused there); this model catches the obvious mistakes first,
    with a 422 that names the field."""

    type: Literal["approve", "edit", "reject"]
    amount: float | None = Field(default=None, gt=0, description="Required for edit: what to pay instead.")
    reason: str | None = Field(default=None, min_length=1, max_length=500,
                               description="Required for edit and reject: the customer is told it.")
    interrupt_id: str | None = Field(
        default=None, description="Which request this answers. Needed only if the thread has two.")

    @model_validator(mode="after")
    def fields_for_type(self) -> Decision:
        if self.type == "edit" and (self.amount is None or not self.reason):
            raise ValueError("an edit needs both amount and reason")
        if self.type == "reject" and not self.reason:
            raise ValueError("a rejection needs a reason: the customer will be told it")
        return self

    def as_resume_value(self) -> dict:
        """The value Command(resume=...) carries: v9's ManagerDecision shape."""
        return self.model_dump(exclude_none=True, exclude={"interrupt_id"})
