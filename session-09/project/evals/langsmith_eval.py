"""One LangSmith EXPERIMENT over the gold questions: plain checks plus two LLM judges.

    uv run python evals/langsmith_eval.py                     # arm A: settings' defaults
    uv run python evals/langsmith_eval.py --k 1               # arm B: one thing changed
    uv run python evals/langsmith_eval.py --mode hybrid       # or the retrieval mode

Run it twice with one variable changed, then open the two experiments side by
side in LangSmith (Datasets & Experiments -> shopwise-kb-questions -> select both
-> Compare). run_eval.py prints three numbers in a terminal; this files the same
run where a team can see it, per question, next to the last one.

Needs LANGSMITH_API_KEY in .env (the free plan is enough). Spends model
calls: one answer per question (19), plus up to two judge calls per question
(groundedness skips refusals, so about 34). Run run_eval.py often; run this on
purpose.

WHAT EACH EVALUATOR READS
    refusal        plain Python: refused exactly when the corpus cannot answer?
    citation       plain Python: cites the source the answer lives in?
    groundedness   JUDGE: is every claim in the answer supported by the passages?
                   Refusals are skipped: a refusal makes no claim to ground, and a
                   judge reading "a member of the team will follow up" finds it in
                   no passage and calls it a hallucination.
    correctness    JUDGE: does the answer agree with kb_questions.yml's reference?

Each judge is openevals' prebuilt prompt on settings.judge_id, which defaults to
the assistant's own model. A judge is a model with the same failure modes as the
one it grades; the notebook (section 16) checks it against a human before
trusting its numbers.

EVERY EXPERIMENT RECORDS WHAT PRODUCED IT
    k, retrieval mode, rerank, model, judge, collection, and rag_prompt's version
    and hash (ADR-0005). An experiment you cannot reproduce is an anecdote.

CUSTOMER DATA STAYS OUT OF THE TRACE
    The client below is built with an anonymizer that rewrites email addresses
    before anything is sent to LangSmith. The gold questions hold none, but the
    agent's lookups do, and a trace store is a copy of your data on someone
    else's server.
"""
from __future__ import annotations

import argparse
import re
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402
from langchain_core.callbacks import get_usage_metadata_callback  # noqa: E402
from langsmith import Client, traceable  # noqa: E402
from langsmith.anonymizer import create_anonymizer  # noqa: E402
from openevals.llm import create_llm_as_judge  # noqa: E402
from openevals.prompts import CORRECTNESS_PROMPT, RAG_GROUNDEDNESS_PROMPT  # noqa: E402

from config import settings  # noqa: E402
from kb import rag  # noqa: E402
from triage import prompt_sha, prompt_version  # noqa: E402
from utils import make_model  # noqa: E402

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
REDACT = create_anonymizer([{"pattern": EMAIL, "replace": "<email>"}])


def client() -> Client:
    """A LangSmith client that never uploads an email address."""
    return Client(anonymizer=REDACT)


def push_dataset(ls: Client) -> str:
    """The gold questions as a LangSmith dataset, created once by name.

    A second run finds it and adds nothing, so every experiment is filed against
    the same examples. Edit kb_questions.yml and you must delete the dataset (or
    rename EVAL_DATASET): experiments on different questions do not compare."""
    name = settings.EVAL_DATASET
    if ls.has_dataset(dataset_name=name):
        return name
    rows = yaml.safe_load((settings.DATA_DIR / "kb_questions.yml").read_text(encoding="utf-8"))
    dataset = ls.create_dataset(dataset_name=name,
                                description="ShopWise gold questions: data/kb_questions.yml")
    ls.create_examples(dataset_id=dataset.id, examples=[
        {"inputs": {"question": q["question"]},
         "outputs": {"answer": q["reference"], "answerable": q["answerable"],
                     "source": q.get("source")},
         "metadata": {"id": q["id"]}}
        for q in rows])
    return name


def target(**options):
    """The system under test: 2-step RAG. Returns the answer, the passages (two
    evaluators read them) and the tokens it spent (the cost of this arm)."""
    options = {key: v for key, v in options.items() if v is not None}

    @traceable(name="rag_bot")
    def rag_bot(inputs: dict) -> dict:
        with get_usage_metadata_callback() as usage:
            state = rag.answer_with_rag(inputs["question"], **options)
        tokens = sum(u.get("total_tokens", 0) for u in usage.usage_metadata.values())
        return {"answer": state["answer"], "tokens": tokens,
                "documents": [rag.format_context([d]) for d in state["documents"]]}
    return rag_bot


# ── plain-Python evaluators: the same checks as run_eval.py ─────────────────
def refusal(outputs: dict, reference_outputs: dict) -> dict:
    expected = not reference_outputs["answerable"]
    return {"key": "refusal", "score": rag.refused(outputs["answer"]) == expected}


def citation(outputs: dict, reference_outputs: dict) -> dict:
    if not reference_outputs["answerable"] or rag.refused(outputs["answer"]):
        return {"key": "citation", "score": None, "comment": "nothing to cite"}
    return {"key": "citation",
            "score": reference_outputs["source"] in rag.cited_sources(outputs["answer"])}


# ── LLM judges ──────────────────────────────────────────────────────────────
@lru_cache
def judge(prompt: str, key: str):
    """An openevals judge on settings.judge_id. make_model, not build_model:
    openevals needs a plain chat model, and a judge silently falling back to a
    different model halfway through an experiment would be measuring two things."""
    return create_llm_as_judge(prompt=prompt, feedback_key=key,
                               judge=make_model(settings.judge_id))


def groundedness(outputs: dict) -> dict:
    if rag.refused(outputs["answer"]):
        return {"key": "groundedness", "score": None, "comment": "refusal: nothing to ground"}
    return judge(RAG_GROUNDEDNESS_PROMPT, "groundedness")(
        context="\n\n".join(outputs["documents"]), outputs=outputs["answer"])


def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    return judge(CORRECTNESS_PROMPT, "correctness")(
        inputs=inputs["question"], outputs=outputs["answer"],
        reference_outputs=reference_outputs["answer"])


def tokens_spent(outputs: list[dict], reference_outputs: list[dict]) -> dict:
    """A summary evaluator: the whole arm's tokens, the cost line of the comparison."""
    return {"key": "tokens_total", "score": sum(o.get("tokens", 0) for o in outputs)}


EVALUATORS = [refusal, citation, groundedness, correctness]


def arm_metadata(**options) -> dict:
    """Everything that produced this experiment, recorded with it."""
    return {"k": options.get("k") or settings.RAG_K,
            "mode": options.get("mode") or settings.RETRIEVAL_MODE,
            "rerank": settings.RERANK if options.get("rerank") is None else options["rerank"],
            "model": settings.model_id, "judge": settings.judge_id,
            "collection": settings.collection,
            "prompt_version": prompt_version("rag_prompt"),
            "prompt_sha": prompt_sha(rag.RAG_PROMPT)}


def run(prefix: str | None = None, **options):
    """One experiment: every example through the target, then every evaluator."""
    ls = client()
    meta = arm_metadata(**options)
    prefix = prefix or f"v6-k{meta['k']}-{meta['mode']}{'-rerank' if meta['rerank'] else ''}"
    return ls.evaluate(target(**options), data=push_dataset(ls), evaluators=EVALUATORS,
                       summary_evaluators=[tokens_spent], experiment_prefix=prefix,
                       metadata=meta, max_concurrency=2)


def means(results) -> dict:
    """The mean of each feedback key over an experiment's rows (None skipped)."""
    frame = results.to_pandas()
    return {key: frame[f"feedback.{key}"].dropna().astype(float).mean()
            for key in ("refusal", "citation", "groundedness", "correctness")
            if f"feedback.{key}" in frame}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--mode", choices=["dense", "keyword", "hybrid"], default=None)
    ap.add_argument("--rerank", action="store_true", default=None)
    ap.add_argument("--prefix", default=None, help="experiment name prefix")
    args = ap.parse_args()
    results = run(args.prefix, k=args.k, mode=args.mode, rerank=args.rerank)
    print(f"experiment: {results.experiment_name}")
    for key, value in means(results).items():
        print(f"  {key:13} {value:.2f}")
    print(f"  tokens        {int(results.to_pandas()['outputs.tokens'].sum())} (answers only, not the judge)")
