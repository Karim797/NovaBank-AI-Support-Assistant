"""Central configuration.

Why this file exists: every tunable (paths, thresholds, provider choice) is read
from the environment in exactly one place. Modules import `settings`, never
`os.getenv`. That makes the running configuration inspectable (`/ready` echoes
it), testable (tests override fields), and impossible to drift between modules.

Common junior mistake: sprinkling `os.getenv("THRESHOLD", 0.5)` through the
codebase, so the value used at inference silently differs from the value used
during evaluation.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", protected_namespaces=()
    )

    app_env: str = "local"
    app_name: str = "novabank-assistant"
    app_version: str = "1.0.0"
    log_level: str = "INFO"

    # --- artifacts ---
    model_path: Path = Path("models/intent_clf_v1.joblib")
    index_path: Path = Path("models/kb_index_v1.joblib")
    knowledge_base_dir: Path = Path("knowledge_base")

    # --- routing ---
    # None = use the threshold stored in the model artifact (the value the
    # evaluation actually validated). An env override exists for incident
    # response — raising it sheds risky traffic to the safe path immediately,
    # without a redeploy — but the default must follow the artifact, otherwise
    # a retrained model silently keeps the old operating point.
    confidence_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    retrieval_top_k: int = Field(4, ge=1, le=20)
    retrieval_min_score: float = Field(0.15, ge=0.0, le=1.0)

    # --- llm ---
    llm_provider: str = "extractive"  # "extractive" | "anthropic"
    llm_model: str = "claude-sonnet-4-5"
    llm_timeout_s: float = 20.0
    llm_max_retries: int = 2
    llm_max_tokens: int = 700
    anthropic_api_key: str = ""
    anthropic_base_url: str = "https://api.anthropic.com"

    # --- persistence ---
    feedback_db_url: str = "sqlite:///feedback.db"

    # --- api protection ---
    api_key: str = ""  # empty disables auth (local dev only)
    rate_limit_per_minute: int = 60
    max_question_chars: int = 500

    def resolve(self, p: Path) -> Path:
        """Make relative artifact paths independent of the working directory."""
        p = Path(p)
        return p if p.is_absolute() else (REPO_ROOT / p)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
