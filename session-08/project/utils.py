"""Plumbing every LLM program needs — project version v4.

v4: make_model() and build_model() take stream_tool_steps. chat.py passes False:
with the v4 rails and streaming, gpt-oss's tool calls sometimes arrived with
empty arguments (the notebook, section 11). Steps that don't stream are unaffected.

v3: KEY_NAMES moved to config.py, which needs it for settings.has_key.

v0.2 built a Google client by hand. v1 asks LangChain for a model by provider
string, so nothing in project/ imports a vendor's SDK any more.

    build_model()   the configured model: timeout, retries, rate limit, fallback
    ask_json()      one structured call you can trust, on any provider
    load_inbox()    tickets.yml, validated
"""

from __future__ import annotations

import json
import os

import yaml
from langchain.chat_models import init_chat_model
from langchain_core.rate_limiters import InMemoryRateLimiter
from pydantic import TypeAdapter, ValidationError

from config import KEY_NAMES, settings
from schemas import Ticket


class ModelReturnedNothing(RuntimeError):
    """The call came back without a usable answer (cut off, blocked, empty)."""


def has_key(model_id: str) -> bool:
    """Is the key for this provider string in the environment?"""
    key_name = KEY_NAMES.get(model_id.split(":", 1)[0])
    return key_name is None or bool(os.getenv(key_name))


def make_model(model_id: str, stream_tool_steps: bool = True):
    """One chat model with our timeout, retries, output cap and its own rate limit.

    stream_tool_steps=False sets LangChain's disable_streaming="tool_calling": any
    call made with tools available is sent unstreamed, and its reply arrives whole.

    `max_retries` is the HttpRetryOptions we configured by hand in v0.2: retries
    429s, 5xx and timeouts with backoff. The rate limiter makes each call wait
    for a free slot, so a batch doesn't hit the provider's limit in the first
    place. Each provider gets its own limiter because each has its own quota.
    """
    limiter = InMemoryRateLimiter(requests_per_second=settings.REQUESTS_PER_SECOND)
    timeout, retries = settings.REQUEST_TIMEOUT_S, settings.MAX_RETRIES
    if model_id.startswith("openrouter:"):
        # ChatOpenRouter counts the timeout in milliseconds (every other provider
        # counts seconds), and turns max_retries into a retry window of
        # max_retries x 150 s. OpenRouter already fails over between hosts on
        # its side, so one retry here is enough; four meant up to ten minutes.
        timeout, retries = int(timeout * 1000), 1
    return init_chat_model(
        model_id,
        timeout=timeout,
        max_retries=retries,
        max_tokens=settings.MAX_OUTPUT_TOKENS,
        rate_limiter=limiter,
        disable_streaming=False if stream_tool_steps else "tool_calling",
    )


def build_model(stream_tool_steps: bool = True):
    """The assistant's model: settings.model_id, with the fallback attached.

    If the primary call still fails after its retries (an outage, a spent
    quota), the same request goes to settings.FALLBACK. Nobody decides that in
    the moment, which is why the fallback has to be scored before you trust it.
    """
    model = make_model(settings.model_id, stream_tool_steps)
    if not settings.FALLBACK:
        return model
    if not has_key(settings.FALLBACK):
        print(f"(fallback {settings.FALLBACK} is off: no key for it in .env)")
        return model
    return model.with_fallbacks([make_model(settings.FALLBACK, stream_tool_steps)])


def ask_json(model, schema, messages, context: dict | None = None):
    """One structured call. Returns (validated object, usage_metadata).

    `model.with_structured_output(TicketFacts)` would validate for us, but it has
    no way to pass `context`, so TicketFacts' check that every order number is
    in the ticket would silently never run (session 4, section 4). So we ask for
    plain JSON with a dict schema, which every provider hands back the same way,
    and validate it ourselves. If validation fails, we ask once more, showing
    the model its answer and what was wrong with it.

    method="json_schema" asks every provider for JSON in the reply. Some default
    to a forced tool call instead, and Groq's gpt-oss-20b sometimes calls that
    tool by the wrong name, which Groq rejects with a 400 (9 of 20 tickets).
    """
    structured = model.with_structured_output(
        schema.model_json_schema(), method="json_schema", include_raw=True)
    for attempt in (1, 2):
        out = structured.invoke(messages, config={"run_name": schema.__name__})   # its name in LangSmith
        if out["parsed"] is None:
            raise ModelReturnedNothing(f"no usable answer ({out['parsing_error'] or 'empty reply'})")
        try:
            return schema.model_validate(out["parsed"], context=context), out["raw"].usage_metadata
        except ValidationError as err:
            if attempt == 2:
                raise
            problems = "; ".join(e["msg"] for e in err.errors())
            messages = [*messages,
                        ("ai", json.dumps(out["parsed"])),
                        ("human", f"That answer was rejected: {problems}. Return a corrected answer.")]


def load_inbox() -> list[Ticket]:
    """Read tickets.yml and validate every ticket in one go."""
    raw = yaml.safe_load((settings.DATA_DIR / "tickets.yml").read_text(encoding="utf-8"))
    return TypeAdapter(list[Ticket]).validate_python(raw)
