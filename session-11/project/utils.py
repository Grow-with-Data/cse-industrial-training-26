"""Plumbing — project version v8.

v8: everything about the MODEL moved to models.py (build_model, make_model,
has_key, ask_json, ModelReturnedNothing), beside the prompt stamps that left
triage.py. A file called utils is where code goes when nobody has decided where
it belongs; by v7 it held the one function every feature needed, under a name
that said nothing about it. What stays is the inbox loader.

    load_inbox()    tickets.yml, validated
"""

from __future__ import annotations

import yaml
from pydantic import TypeAdapter

from config import settings
from schemas import Ticket


def load_inbox() -> list[Ticket]:
    """Read tickets.yml and validate every ticket in one go."""
    raw = yaml.safe_load((settings.DATA_DIR / "tickets.yml").read_text(encoding="utf-8"))
    return TypeAdapter(list[Ticket]).validate_python(raw)
