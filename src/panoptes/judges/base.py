"""Judge Protocol + prompt-template loading.

A `Judge` is anything that, given a `BenchmarkItem` and a model's response,
produces a `JudgeResponse` carrying a `RubricScore`, token usage, cost, and
the hash of the prompt template that generated it. Concrete implementations
(rubric, pairwise) live in sibling modules.

Prompt templates are markdown files with `## System` and `## User` sections.
The split is done by trivial section parsing rather than Jinja or another
heavyweight templating library — eval prompts rarely benefit from control
flow, and a plain `str.format` substitution keeps the on-disk format
diff-friendly.

The content hash is computed over the *raw bytes of the file*, not the
parsed/substituted prompt. This means renaming a variable doesn't invalidate
the cache, but any actual edit to the rubric does — exactly what we want.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from panoptes.schemas import BenchmarkItem, JudgeResponse


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """A parsed rubric template with `system` + `user` sections.

    `content_hash` identifies this template version; it is stored on every
    `JudgeResponse` and used as a duckdb partition key so the prompt-cache
    layer can serve cached rows when the file is unchanged.
    """

    system: str
    user: str
    content_hash: str
    source_path: Path

    def render_user(self, **kwargs: str) -> str:
        """Substitute `{var}` placeholders in the user section.

        Use `str.format_map` with a `_SafeDict` so missing keys are left
        as literal `{var}` instead of crashing — useful for rubrics that
        accept optional variables.
        """
        return self.user.format_map(_SafeDict(kwargs))


class _SafeDict(dict[str, str]):
    """`format_map` helper that leaves unknown placeholders intact."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


_SECTION_RE = re.compile(r"^##\s+(?P<name>[A-Za-z][A-Za-z0-9 _-]*)\s*$", re.MULTILINE)


def load_prompt_template(path: Path) -> PromptTemplate:
    """Read and parse a `.md` rubric template from disk.

    The file must contain at least one `## System` and one `## User` heading.
    Content above the first heading is treated as a title/description and
    ignored. Subsequent section headings (other than System / User) are
    preserved as part of the preceding section's body.
    """
    raw_bytes = path.read_bytes()
    content_hash = hashlib.sha256(raw_bytes).hexdigest()[:16]
    text = raw_bytes.decode("utf-8")

    sections: dict[str, str] = {}
    matches = list(_SECTION_RE.finditer(text))
    for i, match in enumerate(matches):
        name = match.group("name").strip().lower()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        if name not in sections:
            sections[name] = text[start:end].strip()

    if "system" not in sections or "user" not in sections:
        raise ValueError(
            f"Prompt template {path} must contain '## System' and '## User' sections; "
            f"found {sorted(sections)}."
        )

    return PromptTemplate(
        system=sections["system"],
        user=sections["user"],
        content_hash=content_hash,
        source_path=path,
    )


@runtime_checkable
class Judge(Protocol):
    """Protocol every judge implements.

    `judge_id` is a stable identifier of the form
    `provider:model:judge_variant`, e.g.
    `anthropic:claude-sonnet-4-6:rubric-code-v1`. The pipeline uses it as
    a duckdb partition key and as the Thompson-sampling bandit's arm id.
    """

    judge_id: str

    async def evaluate(
        self,
        item: BenchmarkItem,
        model_response: str,
        *,
        sample_index: int = 0,
        temperature: float | None = None,
    ) -> JudgeResponse: ...
