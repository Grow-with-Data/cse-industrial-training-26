"""A manager's answer to a paused refund, validated like any other outside input."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field, model_validator


# ── v8: the manager's answer to a paused refund (workflows/inbox/nodes.py) ───
class ManagerDecision(BaseModel):
    """What a manager sends to resume a paused refund: ShopWise's own design.

    This is input from outside the program, like a ticket or a .env file, so it
    is validated like one. The order-total check is not here because it needs
    the order: the billing desk applies it."""

    type: Literal["approve", "edit", "reject"]
    amount: float | None = Field(default=None, gt=0,
                                 description="edit only: the amount to refund instead.")
    reason: str = Field(default="", description="Why. Required to reject or edit; "
                                                "the customer is told it.")

    @model_validator(mode="after")
    def edits_and_rejections_say_why(self) -> ManagerDecision:
        if self.type == "edit" and self.amount is None:
            raise ValueError("an edit needs an amount")
        if self.type in ("edit", "reject") and not self.reason.strip():
            raise ValueError(f"a {self.type} needs a reason: the customer will be told it")
        return self
