"""The model, and the helpers every prompt passes through on its way to it. New in v8.

    from models import build_model, ask_json, stamp

    has_key(model_id)     is the key for this provider string in the environment?
    make_model(model_id)  one chat model with our timeout, retries and rate limit
    build_model()         the assistant's model: settings.model_id plus the fallback
    ask_json()            one structured call you can trust, on any provider
    prompt_version()      which wording of ONE prompt this process runs
    prompt_sha()          a content hash of a prompt as assembled
    stamp()               the `config=` keys that label a run with its prompt

WHY THIS FILE EXISTS
    Until v7 these lived in two files named for other things: the model helpers
    in utils.py, the prompt stamps in triage.py, because triage was the first
    program to need them. By v7 agent.py, kb/rag.py, kb/fusion.py and both
    workflows imported triage.py to label a call, so every feature depended on
    the file of one feature, and an edit to triage could break the chat.

    v8 gives them one address named for what they are. Nothing here changed but
    where it lives: every function keeps its name, arguments and behaviour, so
    a diff of this file against v7's utils.py and triage.py shows only moves.
    utils.py keeps load_inbox(); triage.py keeps triage's own prompts.

    uv run python models.py     # which model, which fallback, every prompt's version
"""

from __future__ import annotations

import hashlib
import json
import os

from langchain.chat_models import init_chat_model
from langchain_core.rate_limiters import InMemoryRateLimiter
from pydantic import ValidationError

from config import KEY_NAMES, settings
from prompts.loader import FEATURES, load_prompts, version_in_file


# ── The model (from utils.py) ────────────────────────────────────────────────

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
        print(f"(fallback {settings.FALLBACK} is off: no key for it in labs/.env)")
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
            if out["raw"].response_metadata.get("finish_reason") == "length":
                # A reasoning model can spend the whole budget thinking and write nothing.
                raise ModelReturnedNothing(
                    f"cut off at max_tokens={settings.MAX_OUTPUT_TOKENS} before any answer "
                    f"({out['raw'].usage_metadata.get('output_token_details', {})})")
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


# ── Which wording ran? (from triage.py, new in v6) ───────────────────────────
# Session 9 started changing prompts and comparing scores, so every run says
# which wording produced it. Two halves, because each catches what the other
# cannot (ADR-0005):
#   prompt_version   a name a human chose, kept beside the prompt
#   prompt_sha       a hash of the text as sent: nobody types it, nobody forgets it
# One version with two hashes means the wording changed without anyone saying so.

def prompt_version(name: str) -> str:
    """The version label of ONE prompt: `<name>_version` beside it, in
    prompts/<feature>.yml.

    SHOPWISE_PROMPT_VERSION=rag_prompt:v2-strict overrides it for one run. It
    must name its prompt: an override that relabelled every prompt at once
    would put one wording under two names, the bug a version exists to stop."""
    override = settings.PROMPT_VERSION.strip()
    if override:
        named, sep, label = override.partition(":")
        if not sep:
            raise ValueError("SHOPWISE_PROMPT_VERSION must name its prompt, "
                             f"e.g. {name}:v2 (got {override!r})")
        if named == name:
            return label
    return version_in_file(name)


def prompt_sha(*parts: str) -> str:
    """Twelve hex characters of the prompt's text, as assembled.

    The customer's question is never one of the parts: that is data, and a hash
    that changed with every question could not tell you the WORDING moved."""
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:12]


def stamp(name: str, *parts: str) -> dict:
    """The config= keys that label a run in LangSmith with its prompt.

    `metadata` is what you filter and group by; the `prompt:` tag puts the two
    arms of an experiment one click apart in the run list."""
    version = prompt_version(name)
    return {"metadata": {"prompt": name, "prompt_version": version,
                         "prompt_sha": prompt_sha(*parts)},
            "tags": [f"prompt:{name}:{version}"]}


if __name__ == "__main__":
    print(f"model {settings.model_id} · fallback {settings.FALLBACK or 'off'}")
    for feature in FEATURES:
        names = [k for k in load_prompts(feature) if k.endswith("_prompt")]
        print(f"  prompts/{feature}.yml")
        for name in names:
            print(f"    {name:20} {prompt_version(name)}")
