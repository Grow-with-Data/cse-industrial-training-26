"""HTTP shapes for the two health checks."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Health(BaseModel):
    status: Literal["ok"] = "ok"


class Readiness(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, bool]
    version: str
