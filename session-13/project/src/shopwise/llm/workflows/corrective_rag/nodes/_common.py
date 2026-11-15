"""The corrective-RAG nodes' prompts and model."""
from __future__ import annotations

from functools import lru_cache
from shopwise.llm.models import build_model
from shopwise.llm.prompts.loader import load_prompts


PROMPTS = load_prompts("rag")


GRADE_PROMPT = PROMPTS["grade_prompt"]


REWRITE_PROMPT = PROMPTS["rewrite_prompt"]


@lru_cache
def _model():
    return build_model()
