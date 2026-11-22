from __future__ import annotations

import copy

import pytest

from schema_compat import restore_arguments, transform_schema

SCHEMA = {
    "type": "object",
    "properties": {
        "level": {"type": "integer", "enum": [1, 2, 3]},
        "flag": {"const": True},
        "name": {"type": "string"},
        "note": {"type": "string"},
        "maybe": {"type": ["string", "null"]},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"k": {"enum": [10, 20]}, "opt": {"type": "integer"}},
                "required": ["k"],
            },
        },
        "ref": {"$ref": "#/$defs/R"},
        "choice": {"anyOf": [{"type": "integer", "enum": [7]}, {"type": "string"}]},
    },
    "required": ["level", "name"],
    "$defs": {"R": {"type": "object", "properties": {"x": {"type": "number"}}}},
}


def test_gemini_restores_stringified_enums_and_consts() -> None:
    args = {
        "level": "2",
        "flag": "true",
        "name": "2",
        "items": [{"k": "10"}, {"k": "20"}],
        "choice": "7",
    }
    assert restore_arguments(args, SCHEMA, "gemini") == {
        "level": 2,
        "flag": True,
        "name": "2",  # a real string property stays a string
        "items": [{"k": 10}, {"k": 20}],
        "choice": 7,
    }


def test_openai_strict_drops_nulls_for_optional_properties() -> None:
    args = {
        "level": 1,
        "name": "a",
        "note": None,
        "maybe": None,
        "items": [{"k": 10, "opt": None}],
        "ref": None,
        "flag": None,
    }
    assert restore_arguments(args, SCHEMA, "openai-strict") == {
        "level": 1,
        "name": "a",
        "maybe": None,  # the original schema allowed null: kept
        "items": [{"k": 10}],
    }


def test_required_nulls_are_kept_and_input_is_not_mutated() -> None:
    args = {"level": None, "name": None}
    before = copy.deepcopy(args)
    assert restore_arguments(args, SCHEMA, "openai-strict") == {"level": None, "name": None}
    assert args == before


def test_round_trip_with_transform() -> None:
    schema = {"type": "object", "properties": {"mode": {"enum": [0, 1]}, "tag": {"type": "string"}}}
    gemini = transform_schema(schema, "gemini").schema
    assert gemini["properties"]["mode"]["enum"] == ["0", "1"]
    assert restore_arguments({"mode": "1"}, schema, "gemini") == {"mode": 1}
    strict = transform_schema(schema, "openai-strict").schema
    assert strict["required"] == ["mode", "tag"]
    assert restore_arguments({"mode": 0, "tag": None}, schema, "openai-strict") == {"mode": 0}


def test_other_targets_return_arguments_unchanged() -> None:
    args = {"level": "2", "note": None}
    for target in ("anthropic", "anthropic-strict", "openai"):
        assert restore_arguments(args, SCHEMA, target) == args


def test_recursive_schema_terminates() -> None:
    schema = {"$ref": "#/$defs/N", "$defs": {"N": {"properties": {"n": {"$ref": "#/$defs/N"}}}}}
    deep: dict[str, object] = {}
    node = deep
    for _ in range(300):
        node["n"] = {}
        node = node["n"]  # type: ignore[assignment]
    assert restore_arguments(deep, schema, "openai-strict") == deep


def test_unknown_target() -> None:
    with pytest.raises(ValueError, match="unknown target"):
        restore_arguments({}, SCHEMA, "claude")
