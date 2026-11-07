"""Load the assistant's prompts: one YAML file per feature, from v8 (v9 adds team.yml).

    from prompts.loader import load_prompts
    PROMPTS = load_prompts("inbox")       # prompts/inbox.yml, as a dict
    PROMPTS["split_prompt"]

    triage.yml   the customer reply (v0.1), triage and its examples (v0.2)
    chat.yml     the chat agent's instructions and its summary prompt (agent.py)
    rag.yml      the grounded answer, RAG-Fusion, HyDE, corrective RAG,
                 and v8's standalone-question rewrite (kb/query.py)
    inbox.yml    the inbox graph: split, desk, compose, and v8's summarize
                 and remember
    team.yml     v9: the supervisor, the orders and knowledge specialists,
                 the subagents-as-tools agent and the [ADV] market analyst

WHY ONE FILE PER FEATURE
    Until v7 every prompt lived in one prompts.yml. By v7 it held seventeen, and
    every feature's work was an edit to the same file: a change to the inbox's
    compose wording sat in the same diff as the chat agent's instructions. One
    file per feature means a pull request touches the feature it is about, and a
    reviewer can see which from the file list.

    Prompt text still lives in YAML and never in Python: it changes most often,
    a non-engineer may need to read it, and an edit to ShopWise's tone should not
    be able to break the program.

NOTHING WAS RENAMED
    Every prompt kept its key and its `<name>_version`, so every LangSmith filter
    written before v8 still finds its runs. A version still belongs to one
    prompt, next to its wording; there is no file-level version.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

PROMPTS_DIR = Path(__file__).resolve().parent
FEATURES = ("triage", "chat", "rag", "inbox", "team")


@lru_cache
def load_prompts(feature: str) -> dict:
    """One feature's prompts, read once per process."""
    if feature not in FEATURES:
        raise KeyError(f"no prompts/{feature}.yml: the features are {', '.join(FEATURES)}")
    return yaml.safe_load((PROMPTS_DIR / f"{feature}.yml").read_text(encoding="utf-8"))


def version_in_file(name: str) -> str:
    """The `<name>_version` written beside one prompt, whichever file holds it.

    "unversioned" for a prompt that has none (the v0.1 and v0.2 prompts, which
    predate versions), exactly as prompts.yml answered until v7."""
    for feature in FEATURES:
        prompts = load_prompts(feature)
        if f"{name}_version" in prompts:
            return str(prompts[f"{name}_version"])
    return "unversioned"
