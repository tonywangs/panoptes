"""Runtime configuration: env vars, defaults, paths.

Loaded once at startup via `load_settings()`. Provider API keys are looked up
lazily — a missing key only errors when the corresponding client is constructed,
not at import time, so callers using only one provider don't need to set
unrelated keys just to import the package.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from panoptes.errors import ConfigError


@dataclass(frozen=True, slots=True)
class RateLimit:
    """Per-provider concurrency cap and minimum interval between requests.

    The pipeline uses an `asyncio.Semaphore` sized at `max_concurrency` and an
    optional token-bucket guard sized at `requests_per_minute`. Both default
    to conservative values; tune per provider in your env or pass overrides.
    """

    max_concurrency: int = 8
    requests_per_minute: int = 600


@dataclass(frozen=True, slots=True)
class Settings:
    cache_dir: Path
    runs_dir: Path
    request_timeout_s: float = 60.0
    connect_timeout_s: float = 10.0
    max_retries: int = 5
    backoff_base_s: float = 1.0
    backoff_max_s: float = 30.0
    rate_limits: dict[str, RateLimit] = field(default_factory=dict[str, RateLimit])

    def api_key(self, env_var: str) -> str:
        """Look up an API key from the environment, raising ConfigError if absent."""
        value = os.environ.get(env_var)
        if not value:
            raise ConfigError(
                f"Missing required environment variable {env_var}. "
                "Set it in your environment or a .env file in the project root."
            )
        return value


_DEFAULT_RATE_LIMITS: dict[str, RateLimit] = {
    "anthropic": RateLimit(max_concurrency=8, requests_per_minute=1000),
    "openai": RateLimit(max_concurrency=16, requests_per_minute=3000),
    "google": RateLimit(max_concurrency=8, requests_per_minute=600),
    "openai_compat": RateLimit(max_concurrency=16, requests_per_minute=600),
}


def load_settings(*, env_file: Path | None = None) -> Settings:
    """Build a Settings object from environment variables and a .env file.

    `env_file` defaults to `.env` in the current working directory if it exists.
    Caller-supplied overrides take precedence over .env, which takes precedence
    over the OS environment — same precedence python-dotenv uses by default.
    """
    if env_file is None:
        default = Path.cwd() / ".env"
        if default.exists():
            env_file = default
    if env_file is not None and env_file.exists():
        load_dotenv(dotenv_path=env_file, override=False)

    cache_dir = Path(os.environ.get("PANOPTES_CACHE_DIR", Path.cwd() / ".panoptes-cache"))
    runs_dir = Path(os.environ.get("PANOPTES_RUNS_DIR", Path.cwd() / "runs"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        cache_dir=cache_dir,
        runs_dir=runs_dir,
        request_timeout_s=float(os.environ.get("PANOPTES_REQUEST_TIMEOUT", "60")),
        connect_timeout_s=float(os.environ.get("PANOPTES_CONNECT_TIMEOUT", "10")),
        max_retries=int(os.environ.get("PANOPTES_MAX_RETRIES", "5")),
        backoff_base_s=float(os.environ.get("PANOPTES_BACKOFF_BASE", "1.0")),
        backoff_max_s=float(os.environ.get("PANOPTES_BACKOFF_MAX", "30.0")),
        rate_limits=dict(_DEFAULT_RATE_LIMITS),
    )
