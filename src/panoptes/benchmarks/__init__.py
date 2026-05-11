"""Benchmark loaders. M1: HumanEval. M2+: MBPP, GSM8K, TruthfulQA, MT-Bench."""

from panoptes.benchmarks.humaneval import load_humaneval
from panoptes.benchmarks.loader import content_hashed_cache_path, http_fetch_cached

__all__ = ["content_hashed_cache_path", "http_fetch_cached", "load_humaneval"]
