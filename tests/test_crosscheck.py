"""Cross-validation against the providers' own SDK code, plus invariants.

- Gemini: every repaired schema must validate as google-genai ``types.Schema``,
  a pydantic model with ``extra='forbid'`` (unknown keywords are rejected).
- OpenAI strict: every repaired schema must be a fixed point of openai-agents'
  ``ensure_strict_json_schema`` (OpenAI's own strict-mode conversion), i.e. it
  is already strict.
- Every target: transforming twice changes nothing the second time, and the
  input is never mutated.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from schema_compat import TARGETS, transform_schema

CASES_DIR = Path(__file__).parent / "cases"

# Real-world shapes beyond the corpus: pydantic v2 and zod-to-json-schema output.
EXTRA_SCHEMAS: list[Any] = [
    {
        "$defs": {
            "Address": {
                "properties": {
                    "street": {"title": "Street", "type": "string"},
                    "zip": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
                },
                "required": ["street"],
                "title": "Address",
                "type": "object",
            },
            "Color": {"enum": ["red", "green"], "title": "Color", "type": "string"},
        },
        "properties": {
            "name": {"title": "Name", "type": "string", "minLength": 1},
            "age": {"title": "Age", "type": "integer", "exclusiveMinimum": 0},
            "address": {"$ref": "#/$defs/Address"},
            "color": {"$ref": "#/$defs/Color", "default": "red"},
            "tags": {"items": {"type": "string"}, "title": "Tags", "type": "array"},
            "score": {"anyOf": [{"type": "number"}, {"type": "null"}], "default": None},
        },
        "required": ["name", "age", "address"],
        "title": "Person",
        "type": "object",
    },
    {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["create", "update", "delete"]},
            "payload": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "count": {"type": "number"}},
                "required": ["id"],
                "additionalProperties": False,
            },
            "flags": {"type": "array", "items": {"type": "boolean"}},
        },
        "required": ["action"],
        "additionalProperties": False,
        "$schema": "http://json-schema.org/draft-07/schema#",
    },
]


def all_inputs() -> list[Any]:
    inputs = list(EXTRA_SCHEMAS)
    for file in sorted(CASES_DIR.glob("*.json")):
        inputs += [case["input"] for case in json.loads(file.read_text(encoding="utf-8"))]
    return inputs


INPUTS = all_inputs()


@pytest.mark.parametrize("schema", INPUTS)
def test_gemini_output_is_a_valid_google_genai_schema(schema: Any) -> None:
    types = pytest.importorskip("google.genai.types")
    result = transform_schema(schema, "gemini")
    if not result.ok:
        pytest.skip("schema has unrepairable Gemini errors")
    types.Schema.model_validate(result.schema)
    types.FunctionDeclaration.model_validate({"name": "tool", "parameters": result.schema})


def _normalise_required(node: Any) -> Any:
    if isinstance(node, dict):
        out = {k: _normalise_required(v) for k, v in node.items()}
        if isinstance(out.get("required"), list):
            out["required"] = sorted(out["required"])
        return out
    if isinstance(node, list):
        return [_normalise_required(v) for v in node]
    return node


@pytest.mark.parametrize("schema", INPUTS)
def test_openai_strict_output_is_already_strict(schema: Any) -> None:
    strict_schema = pytest.importorskip("agents.strict_schema")
    result = transform_schema(schema, "openai-strict")
    if not result.ok:
        pytest.skip("schema has unrepairable strict-mode errors")
    converted = strict_schema.ensure_strict_json_schema(copy.deepcopy(result.schema))
    assert _normalise_required(converted) == _normalise_required(result.schema)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("schema", INPUTS)
def test_idempotent_and_pure(schema: Any, target: str) -> None:
    before = copy.deepcopy(schema)
    first = transform_schema(schema, target)
    assert schema == before
    second = transform_schema(first.schema, target)
    if not first.ok:
        # Unrepairable input: a second pass may unroll further, but never "succeed".
        assert not second.ok
        return
    # A repaired schema is stable: nothing left to change, same warnings.
    assert second.changes == [], [c.to_dict() for c in second.changes]
    assert second.schema == first.schema
    assert sorted(i.code for i in second.issues) == sorted(i.code for i in first.issues)
