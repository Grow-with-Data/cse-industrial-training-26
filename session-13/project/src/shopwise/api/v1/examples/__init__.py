"""Request examples for the API docs, kept as JSON beside the code. New in v10.

    Body(openapi_examples=examples("MessageIn"))

/docs and /scalar show each one in a drop-down, ready to send. They live in a JSON
file rather than in the route so that the routes stay short and a non-Python
reviewer can add an example. pyproject.toml ships *.json here as package data.
"""
from __future__ import annotations

import json
from functools import cache
from pathlib import Path


@cache
def _all() -> dict:
    return json.loads((Path(__file__).parent / "requests.json").read_text(encoding="utf-8"))


def examples(schema: str) -> dict:
    """The named examples for one request schema, in OpenAPI's shape."""
    return _all()[schema]
