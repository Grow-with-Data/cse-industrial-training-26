"""Tests for v6's grounded answers and its scorecard, with no key and no network.

A scripted model (GenericFakeChatModel) stands in for the LLM and a fake
embedding for Gemini, so nothing here tests whether answers are GOOD; that is
the notebook's and evals/run_eval.py's job, with the real model. What it tests
is every promise the code makes around the model: labels on every passage, the
refusal and citation readers, the tool's metadata, the prompt stamp, and the
arithmetic of the three metrics.

    uv run python test_rag.py
"""
from __future__ import annotations

import os

from langchain.messages import AIMessage
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

import tools
import triage
from config import settings
from evals import run_eval
from kb import ingest, rag, store

FAKE = DeterministicFakeEmbedding(size=64)
OPENED = Document(page_content="Opened audio products: 15 days from delivery, for hygiene reasons.",
                  metadata={"source": "returns-policy", "section": "1. The return window",
                            "authority": "shop"})
FTC = Document(page_content="ship within 30 days", metadata={
    "source": "ftc-prompt-delivery", "section": "p.1", "authority": "regulator"})


def scripted(*replies: str) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=iter([AIMessage(content=r) for r in replies]))


def test_every_passage_carries_its_label_and_authority():
    context = rag.format_context([OPENED, FTC])
    assert "[returns-policy · 1. The return window] (authority: shop)" in context
    assert "[ftc-prompt-delivery · p.1] (authority: regulator)" in context


def test_refusal_is_a_pattern_not_one_exact_string():
    assert rag.refused(rag.REFUSAL)
    assert rag.refused("Sorry, our policy documents do not cover that.")
    assert rag.refused("Our policy documents don’t cover that.")   # curly apostrophe
    assert not rag.refused("Opened audio can be returned within 15 days.")


def test_citations_are_read_from_labels():
    answer = ("Opened audio has 15 days [returns-policy · 1. The return window]. "
              "The law allows 30 days [ftc-prompt-delivery · p.1].")
    assert rag.cited_sources(answer) == {"returns-policy", "ftc-prompt-delivery"}
    assert ("returns-policy", "1. The return window") in rag.citations(answer)
    assert rag.cited_sources("No labels here.") == set()


def test_generate_sends_the_passages_and_returns_only_its_key(monkeypatch=None):
    model = scripted("15 days [returns-policy · 1. The return window].")
    rag._model.cache_clear()
    original, rag._model = rag._model, lambda: model
    try:
        out = rag.generate({"question": "opened headphones?", "documents": [OPENED]})
    finally:
        rag._model = original
    assert set(out) == {"answer"}                 # node-shaped: only the key it adds
    assert rag.cited_sources(out["answer"]) == {"returns-policy"}


def test_search_reads_any_collection_through_the_same_code():
    """The notebook's broken-layer demo searches a 200-character collection through
    rag.search(vs=...). This is that path, on a fake embedding."""
    client = store.open_client(":memory:")
    store.ensure_collection(client, FAKE, "t")
    chunks = ingest.chunk(ingest.load_corpus(), 200, 40)
    vs = store.vector_store(client, FAKE, name="t")
    vs.add_documents(chunks, ids=[ingest.chunk_id(c) for c in chunks])
    hits = rag.search("opened headphones", k=3, vs=vs, rerank_on=False)
    assert len(hits) == 3 and all("source" in h.metadata for h in hits)


def test_search_policy_returns_citation_metadata():
    original, rag.search = rag.search, lambda q, *a, **kw: [OPENED, FTC]
    try:
        passages = tools.search_policy.invoke({"query": "opened headphones"})
    finally:
        rag.search = original
    assert passages[0]["label"] == "[returns-policy · 1. The return window]"
    assert {"content", "label", "source", "section", "authority"} <= set(passages[0])
    assert passages[1]["authority"] == "regulator"
    assert "search_policy" in [t.name for t in tools.TOOLS]


def test_prompt_stamp_has_version_and_a_stable_hash():
    s = triage.stamp("rag_prompt", rag.RAG_PROMPT)
    assert s["metadata"]["prompt_version"] == triage.PROMPTS["rag_prompt_version"]
    assert s["metadata"]["prompt_sha"] == triage.prompt_sha(rag.RAG_PROMPT)
    assert len(s["metadata"]["prompt_sha"]) == 12
    assert triage.prompt_sha("a") != triage.prompt_sha("b")


def test_prompt_version_override_must_name_its_prompt():
    settings.PROMPT_VERSION = "rag_prompt:v2-strict"
    try:
        assert triage.prompt_version("rag_prompt") == "v2-strict"
        assert triage.prompt_version("agent_prompt") == triage.PROMPTS["agent_prompt_version"]
        settings.PROMPT_VERSION = "v2-strict"
        try:
            triage.prompt_version("rag_prompt")
            raise AssertionError("a bare override should be rejected")
        except ValueError:
            pass
    finally:
        settings.PROMPT_VERSION = ""


def test_scorecard_arithmetic():
    """A fake target that is perfect on every question but one of each kind."""
    questions = run_eval.questions()
    first_answerable = next(q for q in questions if q["answerable"])
    first_unanswerable = next(q for q in questions if not q["answerable"])

    def target(question: str) -> dict:
        q = next(q for q in questions if q["question"] == question)
        if not q["answerable"]:
            if q is first_unanswerable:
                return {"answer": "We match any price!", "documents": []}       # invents
            return {"answer": rag.REFUSAL, "documents": []}
        gold = Document(page_content=" ".join(q["evidence"]), metadata={"source": q["source"]})
        cite = "warranty-policy" if q is first_answerable else q["source"]     # wrong source
        return {"answer": f"Yes [{cite} · x].", "documents": [gold]}

    result = run_eval.score(target)
    n, answerable = len(questions), sum(q["answerable"] for q in questions)
    assert result["refusal"] == (n - 1) / n
    assert result["citation"] == (answerable - 1) / answerable
    assert result["retrieval"] == 1.0
    bad = {r["id"]: run_eval.failures(r) for r in result["rows"] if run_eval.failures(r)}
    assert bad == {first_answerable["id"]: ["cited ['warranty-policy']"],
                   first_unanswerable["id"]: ["answered instead of refusing"]}


def test_a_refused_answerable_question_is_a_refusal_error_not_a_citation_one():
    q = next(q for q in run_eval.questions() if q["answerable"])
    row = run_eval.score_row(q, rag.REFUSAL, [])
    assert run_eval.failures(row) == ["refused an answerable question", "gold chunk not retrieved"]
    assert run_eval.summarise([row])["citation"] == 0.0        # nothing answered, no crash


def test_the_rails_let_search_policy_through_and_the_eval_can_read_it():
    """A scripted agent: get_order, then search_policy, then a cited answer. The
    rails must not block a public tool, and the tool message must parse the way
    evals/run_eval.agent_target reads it."""
    import json

    from test_rails import ask, call, railed
    from middleware import customer_for

    original, rag.search = rag.search, lambda q, *a, **kw: [OPENED]
    try:
        agent, _ = railed(call("get_order", order_id="ORD-5004"),
                          call("search_policy", query="return window backpack"),
                          AIMessage("The window has passed [returns-policy · 1. The return window]."))
        state = ask(agent, "Can I return the backpack from ORD-5004?",
                    who=customer_for("priya@example.com"))
    finally:
        rag.search = original
    tool_msgs = {m.name: m for m in state["messages"] if m.type == "tool"}
    assert set(tool_msgs) == {"get_order", "search_policy"}
    passages = json.loads(tool_msgs["search_policy"].content)
    assert passages[0]["source"] == "returns-policy"
    assert rag.cited_sources(state["messages"][-1].text) == {"returns-policy"}


def test_judge_defaults_to_the_assistants_model():
    assert settings.JUDGE or settings.judge_id == settings.model_id


if __name__ == "__main__":
    os.environ.setdefault("LANGSMITH_TRACING", "false")
    tests = [(name, fn) for name, fn in list(globals().items()) if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"\n{len(tests)} passed")
