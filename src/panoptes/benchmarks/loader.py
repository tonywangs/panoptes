"""Shared benchmark-data loading utilities.

Benchmarks pull from either a Hugging Face `datasets` mirror or a raw HTTP
JSONL. Either way, we cache the fetched bytes under
`PANOPTES_CACHE_DIR/benchmarks/<sha256[:16]>.<ext>` so that repeated runs are
offline and identical content yields identical cache keys.

The cache is *content-addressed* in the sense that we hash the URL (not the
response body), which means re-fetches will hit the same file but will not
detect upstream content drift. That trade-off is intentional: the public
HumanEval / GSM8K / TruthfulQA artifacts are versioned by repo SHA and
should not change in place. If a benchmark switches its upstream, bump the
version-suffixed file name (e.g. `HumanEval-v2.jsonl`) to invalidate.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx


def content_hashed_cache_path(url: str, *, cache_dir: Path, suffix: str = "") -> Path:
    """Return the cache path for `url`. Does not fetch."""
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    sub = cache_dir / "benchmarks"
    sub.mkdir(parents=True, exist_ok=True)
    return sub / f"{digest}{suffix}"


def http_fetch_cached(
    url: str,
    *,
    cache_dir: Path,
    suffix: str = "",
    timeout_s: float = 60.0,
) -> bytes:
    """Fetch `url` once and cache the bytes under `cache_dir`.

    Subsequent calls read from disk. We use a synchronous fetch here because
    benchmark loading runs once at startup (not in the inner loop), and a
    sync API is simpler and avoids dragging asyncio into the dataset module.
    """
    path = content_hashed_cache_path(url, cache_dir=cache_dir, suffix=suffix)
    if path.exists():
        return path.read_bytes()
    with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        data = response.content
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)
    return data


def iter_jsonl(raw: bytes) -> Iterator[dict[str, Any]]:
    """Yield one parsed JSON object per non-empty line of `raw`."""
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        obj = json.loads(stripped)
        if not isinstance(obj, dict):
            raise ValueError(f"Expected JSON object per JSONL line; got {type(obj).__name__}")
        yield {str(k): v for k, v in obj.items()}  # type: ignore[redundant-cast]
