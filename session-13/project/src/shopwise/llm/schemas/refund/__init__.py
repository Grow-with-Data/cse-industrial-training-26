"""A manager's answer to a paused refund, validated like any other outside input."""
from __future__ import annotations

from shopwise.llm.schemas.refund.schema import (
    ManagerDecision,
)

__all__ = ["ManagerDecision"]
