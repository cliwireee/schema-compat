"""Run the language-neutral case corpus in tests/cases/*.json.

Each case lists the expected repaired schema, the exact set of changes
("code@path"), which of them are lossy, and the unrepairable issues.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from schema_compat import transform_schema

CASES_DIR = Path(__file__).parent / "cases"


def load_cases() -> list[Any]:
    cases = []
    for file in sorted(CASES_DIR.glob("*.json")):
        for case in json.loads(file.read_text(encoding="utf-8")):
            cases.append(pytest.param(case, id=f"{file.stem}: {case['name']}"))
    return cases


def keys(issues: Any) -> set[str]:
    return {f"{i.code}@{i.path}" for i in issues}


@pytest.mark.parametrize("case", load_cases())
def test_case(case: dict[str, Any]) -> None:
    original = copy.deepcopy(case["input"])
    result = transform_schema(case["input"], case["target"])

    assert case["input"] == original, "input must not be mutated"
    assert result.schema == case["output"]
    assert keys(result.changes) == set(case["changes"])
    assert keys(c for c in result.changes if c.lossy) == set(case["lossy"])
    assert keys(result.issues) == set(case["issues"])
    expected_ok = case.get("ok", not case["issues"])
    assert result.ok is expected_ok
