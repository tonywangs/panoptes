"""Unit tests for the M5 benchmark loaders + calibration probe.

Avoids live HTTP — we point each loader at a `file://` URL or a
respx-mocked URL. The loaders' core logic is parsing, not networking.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import httpx
import pytest
import respx

from panoptes.benchmarks.calibration_probe import (
    grade_calibration_probe,
    obfuscate_humaneval,
)
from panoptes.benchmarks.gsm8k import load_gsm8k, parse_final_answer
from panoptes.benchmarks.humaneval import load_humaneval
from panoptes.benchmarks.mbpp import load_mbpp
from panoptes.benchmarks.mtbench import load_mtbench
from panoptes.schemas import BenchmarkItem, TaskFamily


def test_parse_gsm8k_final_answer() -> None:
    text = "Step 1 ...\nStep 2 ...\n#### 42"
    assert parse_final_answer(text) == "42"
    assert parse_final_answer("#### 1,234.5") == "1234.5"
    assert parse_final_answer("no marker") is None


@respx.mock
def test_load_gsm8k_parses_jsonl(tmp_path: Path) -> None:
    records = [
        {"question": "Q1?", "answer": "A1\n#### 7"},
        {"question": "Q2?", "answer": "no marker"},
    ]
    payload = "\n".join(json.dumps(r) for r in records).encode("utf-8")
    url = "https://example.test/gsm8k.jsonl"
    respx.get(url).mock(return_value=httpx.Response(200, content=payload))
    items = load_gsm8k(cache_dir=tmp_path, url=url)
    assert len(items) == 2
    assert items[0].task_family is TaskFamily.MATH
    assert items[0].reference == "7"
    assert items[1].reference is None


@respx.mock
def test_load_mbpp_parses_jsonl(tmp_path: Path) -> None:
    payload = json.dumps(
        {
            "task_id": 1,
            "text": "Add two numbers",
            "code": "def add(a, b): return a + b",
            "test_list": ["assert add(1, 2) == 3"],
        }
    ).encode("utf-8")
    url = "https://example.test/mbpp.jsonl"
    respx.get(url).mock(return_value=httpx.Response(200, content=payload))
    items = load_mbpp(cache_dir=tmp_path, url=url)
    assert items[0].item_id == "MBPP/1"
    assert "canonical_solution" in items[0].metadata


@respx.mock
def test_load_mtbench_parses_jsonl(tmp_path: Path) -> None:
    payload = "\n".join(
        json.dumps(r)
        for r in [
            {"question_id": 1, "category": "coding", "turns": ["t1", "t2"]},
            {"question_id": 2, "category": "math", "turns": ["only one"]},
        ]
    ).encode("utf-8")
    url = "https://example.test/mtbench.jsonl"
    respx.get(url).mock(return_value=httpx.Response(200, content=payload))
    items = load_mtbench(cache_dir=tmp_path, url=url)
    assert items[0].prompt == "t1"
    assert items[0].metadata["turn_2"] == "t2"
    assert items[1].prompt == "only one"


@respx.mock
def test_load_humaneval_parses_gzipped(tmp_path: Path) -> None:
    record = json.dumps(
        {
            "task_id": "HumanEval/0",
            "prompt": "def f():\n    pass\n",
            "canonical_solution": "    return 1\n",
            "test": "def check(fn):\n    assert fn() == 1\n",
            "entry_point": "f",
        }
    ).encode("utf-8")
    gz = gzip.compress(record)
    url = "https://example.test/he.jsonl.gz"
    respx.get(url).mock(return_value=httpx.Response(200, content=gz))
    items = load_humaneval(cache_dir=tmp_path, url=url)
    assert len(items) == 1
    assert items[0].metadata["entry_point"] == "f"


def test_obfuscate_humaneval_renames_entry_point_and_passes_under_sandbox() -> None:
    src = BenchmarkItem(
        item_id="HumanEval/0",
        benchmark="humaneval",
        task_family=TaskFamily.CODE,
        prompt='def add(a, b):\n    """Return the sum."""\n',
        reference="    return a + b\n",
        metadata={
            "entry_point": "add",
            "canonical_solution": "    return a + b\n",
            "test": (
                "def check(fn):\n"
                "    assert fn(1, 2) == 3\n"
                "    assert fn(0, 0) == 0\n"
            ),
        },
    )
    probes = obfuscate_humaneval([src])
    assert len(probes) == 1
    probe = probes[0]
    assert probe.original_entry_point == "add"
    assert probe.rewritten_entry_point != "add"
    # The rewritten canonical should grade as passing in the sandbox.
    canonical = probe.item.metadata["canonical_solution"]
    assert isinstance(canonical, str)
    assert grade_calibration_probe(probe, canonical) is True
    # A wrong candidate should not pass.
    wrong = "    return a - b\n"
    assert grade_calibration_probe(probe, wrong) is False


def test_obfuscate_humaneval_skips_incomplete_items() -> None:
    incomplete = BenchmarkItem(
        item_id="HumanEval/x",
        benchmark="humaneval",
        task_family=TaskFamily.CODE,
        prompt="...",
        reference=None,
        metadata={"entry_point": "f"},  # missing canonical_solution + test
    )
    probes = obfuscate_humaneval([incomplete])
    assert probes == []


@pytest.mark.parametrize("dirty", ["def f(): pass", "name with spaces"])
def test_parse_final_answer_handles_dirty_input(dirty: str) -> None:
    # Should not raise even on garbage input.
    parse_final_answer(dirty)
