"""Local HF DeBERTa-v3-mnli NLI backend.

Lazy imports `transformers` and `torch` so the rest of PANOPTES doesn't pull
them in. Install via the `providers-hf` extra:

    uv sync --extra providers-hf

Default checkpoint is
`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, a strong public
NLI model with the canonical `[entailment, neutral, contradiction]` label
ordering at indices `[0, 1, 2]`. Different checkpoints can be supplied via
`model_name`; the constructor reads `config.id2label` so the mapping is
auto-detected.

The model is held on CPU by default. Pass `device="cuda"` (or "mps") if
available. Inference batching is supported via `classify_pairs`.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from panoptes.uq.nli.base import NLILabel, NLIScores

if TYPE_CHECKING:  # pragma: no cover
    from transformers import (  # pyright: ignore[reportMissingImports]
        PreTrainedModel,
        PreTrainedTokenizerBase,
    )


_DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"


class DebertaNLIBackend:
    """HF DeBERTa-v3 MNLI backend.

    Construct, optionally specifying a different checkpoint and device.
    The model loads on first call (or eagerly via `prepare()`), keeping
    package import cost low.
    """

    def __init__(
        self,
        *,
        model_name: str = _DEFAULT_MODEL,
        device: str = "cpu",
        max_seq_len: int = 512,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._max_seq_len = max_seq_len
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._label_map: dict[int, NLILabel] | None = None

    async def prepare(self) -> None:
        """Eagerly load tokenizer + weights. Otherwise loaded lazily on first call."""
        if self._model is not None:
            return
        await asyncio.to_thread(self._load)

    def _load(self) -> None:
        try:
            from transformers import (  # pyright: ignore[reportMissingImports]
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "DebertaNLIBackend requires `transformers` and `torch`. "
                "Install via: uv sync --extra providers-hf"
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
        model.to(self._device)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        # id2label is a dict[int, str]; we lower-case and remap to our enum.
        id2label_raw = model.config.id2label
        mapping: dict[int, NLILabel] = {}
        for idx, label in id2label_raw.items():
            normalized = str(label).strip().lower()
            if "entail" in normalized:
                mapping[int(idx)] = NLILabel.ENTAILMENT
            elif "contradict" in normalized:
                mapping[int(idx)] = NLILabel.CONTRADICTION
            else:
                mapping[int(idx)] = NLILabel.NEUTRAL
        self._label_map = mapping

    async def classify_pair(self, premise: str, hypothesis: str) -> NLIScores:
        results = await self.classify_pairs([(premise, hypothesis)])
        return results[0]

    async def classify_pairs(
        self, pairs: list[tuple[str, str]]
    ) -> list[NLIScores]:
        await self.prepare()
        return await asyncio.to_thread(self._infer_sync, pairs)

    def _infer_sync(self, pairs: list[tuple[str, str]]) -> list[NLIScores]:
        # Re-imported under to_thread so the import cost stays off the event loop.
        import torch  # pyright: ignore[reportMissingImports]

        assert self._tokenizer is not None
        assert self._model is not None
        assert self._label_map is not None
        premises = [p for p, _ in pairs]
        hypotheses = [h for _, h in pairs]
        tokenizer: PreTrainedTokenizerBase = self._tokenizer  # type: ignore[assignment]
        model: PreTrainedModel = self._model  # type: ignore[assignment]
        enc = tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=self._max_seq_len,
            return_tensors="pt",
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        results: list[NLIScores] = []
        for row in probs:
            score_by_label: dict[NLILabel, float] = {
                NLILabel.ENTAILMENT: 0.0,
                NLILabel.NEUTRAL: 0.0,
                NLILabel.CONTRADICTION: 0.0,
            }
            for idx, value in enumerate(row):
                score_by_label[self._label_map[idx]] += float(value)
            top = max(score_by_label, key=lambda k: score_by_label[k])
            results.append(
                NLIScores(
                    entailment=score_by_label[NLILabel.ENTAILMENT],
                    neutral=score_by_label[NLILabel.NEUTRAL],
                    contradiction=score_by_label[NLILabel.CONTRADICTION],
                    top=top,
                )
            )
        return results

    async def aclose(self) -> None:
        # Nothing to release; numpy/torch hold the tensors. The HF cache
        # is owned by transformers itself.
        return None
