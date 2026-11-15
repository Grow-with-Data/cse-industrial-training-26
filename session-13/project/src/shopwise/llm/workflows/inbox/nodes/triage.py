"""triage: one structured call on the turn, returning S03's TriageResult."""
from __future__ import annotations

from shopwise.core.config import settings
from shopwise.llm.models import ask_json
from shopwise.llm.prompts.loader import load_prompts
from shopwise.llm.schemas import TriageResult
from shopwise.llm.workflows.inbox.state import InboxState
from shopwise.llm.workflows.inbox.nodes._common import _model, as_block, reading


TRIAGE_PROMPTS = load_prompts("triage")


# v10: was triage.py's, the v1 script; the only part of it the graph still used.
def triage_system(with_examples: bool | None = None) -> str:
    """The triage prompt, plus the few-shot examples when they are switched on."""
    if with_examples is None:
        with_examples = settings.USE_FEW_SHOT
    prompt = TRIAGE_PROMPTS["triage_prompt"]
    if with_examples:
        prompt += "\nExamples of how ShopWise labels the awkward cases:\n"
        for ex in TRIAGE_PROMPTS["few_shot_examples"]:
            prompt += (f"\n<ticket>\nSubject: {ex['subject']}\n{ex['body']}\n</ticket>\n"
                       f"category: {ex['category']}\n")
    return prompt


def triage_node(state: InboxState) -> dict:
    """Session 3's triage, now on the turn as the graph reads it."""
    result, _ = ask_json(_model(), TriageResult,
                         [("system", triage_system()), ("human", as_block(reading(state)))])
    return {"triage": result}
