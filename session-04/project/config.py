"""Configuration for the ShopWise assistant — project version v1.

Still the Settings class from v0.2. New in v1:

    PROVIDER + MODEL   joined into one provider string, google_genai:gemini-3.5-flash-lite.
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

from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# session-04/project/config.py -> repo root, the folder holding data/ and .env
LABS = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Everything the assistant can be told that is not code. Read it top to bottom."""

    model_config = SettingsConfigDict(
        env_file=LABS / ".env",
        env_prefix="SHOPWISE_",
        extra="ignore",          # the shared .env also holds keys for later sessions
    )

    APP_NAME: str = "ShopWise Support Assistant"
    VERSION: str = "1.0.0"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"

    # SecretStr prints as '**********', so the key never ends up in a log line,
    # a traceback or a screen share. Reading it takes .get_secret_value().
    GEMINI_API_KEY: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="GEMINI_API_KEY",   # no SHOPWISE_ prefix: the SDK's own name
    )

    # Who serves the model. The provider ID uses an underscore: "google-genai" is
    # the pip package's spelling and LangChain rejects it. Leave the prefix off
    # altogether and LangChain picks Vertex AI, a different Google product.
    PROVIDER: str = "google_genai"

    # The same five model names as v0.1. Code says settings.model_id; a model ID is
    # never typed at a call site.
    MODEL: str = "gemini-3.5-flash-lite"
    MODEL_STRONG: str = "gemini-3.8-flash"
    MODEL_SEARCH: str = "gemini-3.5-flash-lite"
    MODEL_PRO: str = "gemini-2.5-pro"
    EMBED_MODEL: str = "gemini-embedding-2"

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

    DATA_DIR: Path = LABS / "data"

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
            raise ValueError("GEMINI_API_KEY is required when ENVIRONMENT=production")
        return self

    @property
    def has_key(self) -> bool:
        return bool(self.GEMINI_API_KEY.get_secret_value())

    @property
    def model_id(self) -> str:
        """The provider string the whole program passes to init_chat_model."""
        return f"{self.PROVIDER}:{self.MODEL}"


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
