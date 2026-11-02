"""What the policy graph remembers. Raw data only: the prompt text is built
inside the node that needs it.

ONE KEY HAS A REDUCER
    `ranked` is written by every search_one branch of RAG-Fusion, several in the
    same super-step. Two writes to an ordinary key in one step is an
    InvalidUpdateError, so it appends instead. Each entry carries its attempt
    number, because an append reducer never forgets: on the corrective loop's
    second search, the first search's lists are still in it (kb/fusion.fuse).
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.documents import Document


class RAGState(TypedDict, total=False):
    question: str                                   # rewrite overwrites it (graph.py)
    k: int                                          # optional: passages per search
    queries: list[str]                              # RAG-Fusion: what expand wrote
    ranked: Annotated[list[dict], operator.add]     # RAG-Fusion: one list per query
    hypothesis: str                                 # HyDE: the passage searched with
    documents: list[Document]                       # what generate answers from
    relevant: bool                                  # the grader's verdict
    attempts: int                                   # searches so far; absent means one
    answer: str
