"""Configuration for the ShopWise assistant — project version v6.

WHAT CHANGED FROM v5 — ANSWERS FROM THE KNOWLEDGE BASE, AND A SCORE FOR THEM
    v6 answers policy questions from the passages kb/rag.py retrieves, and
    evals/ scores those answers. Each setting below is one variable the
    session 9 notebook changes and re-scores:

      RAG_K           how many chunks go into the prompt. Too few and the
                      exception clause can miss the cut; too many and the
                      answer is buried in near-misses you pay tokens for.
      RETRIEVAL_MODE  "dense" (meaning), "keyword" (BM25) or "hybrid" (both,
                      fused). The collection has held both kinds of vector
                      since v5, so switching needs no re-ingest.
      RERANK          fetch RERANK_FETCH chunks, re-score each against the
      RERANK_FETCH    question with a small cross-encoder (RERANK_MODEL, on
      RERANK_MODEL    your CPU, no key), keep the best RAG_K.
      PROMPT_VERSION  overrides ONE prompt's version label for one run, as
                      `name:label` (utils.prompt_version explains why).
      JUDGE           the model that grades groundedness in
                      evals/langsmith_eval.py. Empty means the assistant's own
                      model, which is the course default and the reason the
                      notebook checks the judge against a human.
      EVAL_DATASET    the LangSmith dataset name the gold questions go into.

    The defaults are the ones session 9's scorecard chose. Any change is
    `SHOPWISE_RAG_K=1` (or similar) in .env, then evals/run_eval.py.

--- v5 notes follow ---

WHAT CHANGED FROM v4 — THE KNOWLEDGE BASE
    v5 adds a knowledge base: the policy corpus, embedded and kept on disk
    (kb/ingest.py builds it, kb/store.py opens it). Five questions only
    configuration can answer:

      EMBED_MODEL     which embedding model, and through a prefix, which
                      provider. `gemini-embedding-2` is Gemini (the default,
                      free tier, your GEMINI_API_KEY). The chat model stays on
                      OpenRouter: the two are chosen separately.
                      `fastembed:BAAI/bge-small-en-v1.5` is the free CPU model
                      for anyone without a Gemini key.
      EMBED_DIM       how many dimensions Gemini returns. 768, not the default
                      3072: a quarter of the storage, and on this corpus the
                      ranking does not move (session 8, section 8). The
                      fastembed model fixes its own size, so it ignores this.
      QDRANT_PATH     where the knowledge base lives on disk: .state/qdrant,
                      gitignored, so every later snapshot opens the same one.
      CHUNK_SIZE      how big a chunk is, in characters, and how much
      CHUNK_OVERLAP   neighbouring chunks share. Chosen by measurement (session
                      8, section 12 scores three sizes), not copied from a blog.

    And one computed name, `collection`, which is the important one. It is
    DERIVED from the embedding model and dimension, never typed by hand:

        gemini-embedding-2 at 768        -> policies__gemini-embedding-2-768
        fastembed:BAAI/bge-small-en-v1.5 -> policies__bge-small-en-v1.5

    A vector only means something next to vectors from the same model at the
    same size. Switch models and you switch collections, so a query can never be
    compared with vectors from another space by accident.

--- v4 notes follow ---

v4 changes the version number and nothing else. The rails' own numbers (call
limits, retry counts, banned phrases) live in middleware.py as plain constants,
each read in one place, next to the code that uses it.

--- v3 notes follow ---

New in v3, the default provider is OpenRouter: one key and one bill for
hundreds of hosted models, open-weights ones included. The default model is
openai/gpt-oss-20b, an open-weights model, which the session 5 fallback was
already using through Groq. Gemini is one line in .env away:

    SHOPWISE_PROVIDER=openrouter      SHOPWISE_MODEL=google/gemini-3.5-flash-lite
    SHOPWISE_PROVIDER=google_genai    SHOPWISE_MODEL=gemini-3.5-flash-lite

has_key now checks the key of whichever provider is configured, not Gemini's.

Also new in v3, two settings for SummarizationMiddleware (agent.py), which keeps a
long ticket's history inside a budget:

    SUMMARY_TRIGGER_TOKENS   how big a ticket's history may grow before its
                             oldest messages are replaced by one summary
    SUMMARY_KEEP_MESSAGES    how many recent messages survive word for word
                             when that happens

There is no setting for WHERE conversations are saved: in v3 they live in the
process's memory and die with it. Session 11 adds a file on disk. From v2:

    PROVIDER + MODEL   joined into one provider string, openrouter:openai/gpt-oss-20b.
                       Switching vendor is changing these two lines in .env.
    FALLBACK           a second provider string, tried automatically when the
                       first one fails at runtime. Empty turns it off.
    MAX_CONCURRENCY, REQUESTS_PER_SECOND
                       how hard triage.py may push the provider (session 4, section 6).
    LANGSMITH_*        read by LangSmith itself; mirrored here only so the
                       startup line can say whether tracing is on.

Every field is read from `SHOPWISE_<FIELD>`, except the API keys and LANGSMITH_*,
which keep the names their SDKs look for.

    uv run python config.py      # prints what it resolved to
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# session-08/project/config.py -> repo root, the folder holding data/ and .env
LABS = Path(__file__).resolve().parent.parent.parent

# The environment variable each provider's SDK reads its key from.
KEY_NAMES = {
    "openrouter": "OPENROUTER_API_KEY",
    "google_genai": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": None,                       # runs on your machine, no key
}


class Settings(BaseSettings):
    """Everything the assistant can be told that is not code. Read it top to bottom."""

    model_config = SettingsConfigDict(
        env_file=LABS / ".env",
        env_prefix="SHOPWISE_",
        extra="ignore",          # the shared .env also holds keys for later sessions
    )

    APP_NAME: str = "ShopWise Support Assistant"
    VERSION: str = "6.0.0"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"

    # SecretStr prints as '**********', so the key never ends up in a log line,
    # a traceback or a screen share. Reading it takes .get_secret_value().
    GEMINI_API_KEY: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="GEMINI_API_KEY",   # no SHOPWISE_ prefix: the SDK's own name
    )

    # Who serves the model. From v3, OpenRouter: it routes each call to one of
    # several hosts serving the model you named, behind one key. Provider IDs
    # use underscores ("google_genai", not the pip spelling "google-genai").
    PROVIDER: str = "openrouter"

    # Code says settings.model_id; a model ID is never typed at a call site. The
    # other four names are still Gemini's and nothing in v3 uses them; each is
    # revisited in the session that first needs it.
    MODEL: str = "openai/gpt-oss-20b"         # open weights; OpenRouter names are vendor/model
    MODEL_STRONG: str = "gemini-3.8-flash"
    MODEL_SEARCH: str = "gemini-3.5-flash-lite"
    MODEL_PRO: str = "gemini-3.1-pro-preview"   # billed key only
    EMBED_MODEL: str = "gemini-embedding-2"
    """The embedding model that fills, and must forever query, the knowledge base.

    No prefix means Gemini. `fastembed:<model>` means a free model running on
    your CPU through fastembed: no key, no network after the first download.

        SHOPWISE_EMBED_MODEL=fastembed:BAAI/bge-small-en-v1.5

    Switching is deliberate, never automatic: it moves you to another collection
    (see `collection` below), which is empty until you run kb/ingest.py. A quiet
    fallback would let two people query two different spaces through the same
    code, and the only symptom would be bad answers.
    """
    EMBED_DIM: int = Field(
        default=768, ge=128, le=3072,
        description="Gemini output dimensions (768, 1536 or 3072). Changing it means re-ingesting.",
    )

    MAX_OUTPUT_TOKENS: int = Field(default=1024, ge=64, le=8192)
    REQUEST_TIMEOUT_S: float = Field(default=60.0, gt=0)
    MAX_RETRIES: int = Field(default=4, ge=1, le=8)

    # Plan B at runtime: a whole provider string. Groq is free, which is why it's
    # the default. Set SHOPWISE_FALLBACK= (empty) to turn it off.
    FALLBACK: str = "groq:openai/gpt-oss-20b"

    # triage.py runs tickets in parallel. Free tiers allow a few requests a
    # minute (Groq's free gpt-oss-20b: 30 a minute, and only 8,000 tokens a
    # minute), so keep both low unless your key is billed. Per provider.
    MAX_CONCURRENCY: int = Field(default=4, ge=1, le=32)
    REQUESTS_PER_SECOND: float = Field(default=0.25, gt=0)

    # LangSmith reads these itself (no SHOPWISE_ prefix).
    LANGSMITH_TRACING: bool = Field(default=False, validation_alias="LANGSMITH_TRACING")
    LANGSMITH_PROJECT: str = Field(default="default", validation_alias="LANGSMITH_PROJECT")

    # Few-shot examples in the triage prompt. Measured with ../score.py --compare;
    # the result is written in session 3's notebook, section 4. Turn them on for
    # one run with SHOPWISE_USE_FEW_SHOT=true.
    USE_FEW_SHOT: bool = False

    # A label for the wording that produced a run (new in v6). prompts.yml keeps
    # one `<name>_version` beside each prompt; this overrides ONE of them for one
    # run, and must say which: SHOPWISE_PROMPT_VERSION=rag_prompt:v2-strict.
    PROMPT_VERSION: str = ""

    DATA_DIR: Path = LABS / "data"

    # ── Knowledge base (new in v5) ──────────────────────────────────────────
    QDRANT_PATH: Path = LABS / ".state" / "qdrant"
    """Embedded Qdrant keeps the knowledge base in this folder. Persistent, so the
    session 9 snapshot reads what session 8 built. Gitignored: rebuild it any
    time with `uv run python kb/ingest.py`.

    ONE PROCESS AT A TIME: embedded Qdrant locks the folder. A notebook kernel
    holding it makes ingest.py fail with "Storage folder ... is already accessed
    by another instance". Close the client (or the kernel) first."""

    CHUNK_SIZE: int = Field(default=800, ge=100, le=8000,
                            description="Characters per chunk. Chosen by session 8's retrieval score.")
    CHUNK_OVERLAP: int = Field(default=160, ge=0,
                               description="Characters neighbouring chunks share: insurance at the seams.")

    # ── Answering from it (new in v6) ───────────────────────────────────────
    RAG_K: int = Field(default=4, ge=1, le=20, description="Chunks handed to the model per question.")
    RETRIEVAL_MODE: Literal["dense", "keyword", "hybrid"] = "dense"
    RERANK: bool = False
    RERANK_FETCH: int = Field(default=20, ge=2, le=100,
                              description="Chunks fetched for the re-ranker to choose RAG_K from.")
    RERANK_MODEL: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # ── Scoring it (new in v6) ──────────────────────────────────────────────
    JUDGE: str = ""
    """A provider string for the LLM judge. Empty: the assistant's own model_id."""
    EVAL_DATASET: str = "shopwise-kb-questions"

    # A ticket's whole history is resent on every model call, so input tokens
    # grow with every turn. Past SUMMARY_TRIGGER_TOKENS (counted over the
    # messages, approximately), the oldest messages become one summary and the
    # newest SUMMARY_KEEP_MESSAGES stay as they were. The notebook, section 10,
    # measures what each setting costs and what it forgets.
    SUMMARY_TRIGGER_TOKENS: int = Field(default=4000, ge=500)
    SUMMARY_KEEP_MESSAGES: int = Field(default=20, ge=2)

    @field_validator("MODEL", "MODEL_STRONG", "MODEL_SEARCH", "MODEL_PRO", "EMBED_MODEL")
    @classmethod
    def no_blank_model_ids(cls, value: str) -> str:
        """`SHOPWISE_MODEL=` with nothing after it should fail here, not as a 404 later."""
        value = value.strip()
        if not value:
            raise ValueError("must not be empty: delete the line or give it a model ID")
        return value

    @field_validator("PROVIDER")
    @classmethod
    def provider_uses_underscores(cls, value: str) -> str:
        """`google-genai` fails much later with a confusing message. Catch it here."""
        if "-" in value or ":" in value:
            raise ValueError(f"{value!r}: provider IDs use underscores and no colon, "
                             "e.g. google_genai, groq, openai, anthropic, ollama")
        return value

    @model_validator(mode="after")
    def production_needs_a_key(self) -> Settings:
        """You can read the notebooks without a key. Production can't run without one."""
        if self.ENVIRONMENT == "production" and not self.has_key:
            raise ValueError(f"{KEY_NAMES[self.PROVIDER]} is required when ENVIRONMENT=production")
        return self

    @property
    def has_key(self) -> bool:
        """Is the configured provider's key in the environment? (Ollama needs none.)"""
        key_name = KEY_NAMES.get(self.PROVIDER)
        return key_name is None or bool(os.getenv(key_name))

    @property
    def model_id(self) -> str:
        """The provider string the whole program passes to init_chat_model."""
        return f"{self.PROVIDER}:{self.MODEL}"

    @property
    def embed_provider(self) -> str:
        """`fastembed` or `google_genai`, read from EMBED_MODEL's prefix."""
        return "fastembed" if self.EMBED_MODEL.startswith("fastembed:") else "google_genai"

    @property
    def embed_name(self) -> str:
        """EMBED_MODEL without its provider prefix."""
        return self.EMBED_MODEL.removeprefix("fastembed:")

    @property
    def has_embed_key(self) -> bool:
        """Can the knowledge base be embedded? fastembed needs no key; Gemini does."""
        return self.embed_provider == "fastembed" or bool(os.getenv("GEMINI_API_KEY"))

    @property
    def collection(self) -> str:
        """The knowledge base's collection name, derived from its embedding space.

        Never set by hand. The name IS the space, so changing model or dimension
        changes the name, and a query can only ever meet vectors it is
        comparable with."""
        if self.embed_provider == "fastembed":
            return f"policies__{self.embed_name.split('/')[-1].lower()}"
        return f"policies__{self.embed_name}-{self.EMBED_DIM}"

    @property
    def judge_id(self) -> str:
        """The judge's provider string: JUDGE, or the assistant's model if unset."""
        return self.JUDGE.strip() or self.model_id

    @property
    def POLICIES_DIR(self) -> Path:
        return self.DATA_DIR / "policies"


@lru_cache
def get_settings() -> Settings:
    """Build the settings once. Every caller gets the same object.

    load_dotenv is still needed: every vendor's SDK, and LangSmith, read their keys
    from os.environ themselves, so the variables have to be there too.
    """
    load_dotenv(LABS / ".env", override=True)
    return Settings()


settings = get_settings()


if __name__ == "__main__":
    print(f"{settings.APP_NAME} v{settings.VERSION}  [{settings.ENVIRONMENT}]")
    print(f"  model {settings.model_id} · fallback {settings.FALLBACK or 'off'}"
          f" · tracing {'on, project ' + settings.LANGSMITH_PROJECT if settings.LANGSMITH_TRACING else 'off'}")
    for name, value in settings.model_dump().items():
        print(f"  {name:20} {value}")
    print(f"  {'collection':20} {settings.collection}  (computed)")
