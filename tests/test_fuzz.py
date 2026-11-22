"""Randomised schemas through every target (deterministic seeds).

Invariants: no exceptions, input never mutated, accepted (ok) outputs are
stable under a second pass, Gemini ok outputs validate as google-genai
``types.Schema``, OpenAI-strict ok outputs are fixed points of openai-agents'
strict conversion. Set SCHEMA_COMPAT_FUZZ=N to run N schemas instead of 400.
"""

from __future__ import annotations

import copy
import os
import random
from typing import Any

import pytest

from schema_compat import TARGETS, restore_arguments, transform_schema, transform_tools

COUNT = int(os.environ.get("SCHEMA_COMPAT_FUZZ", "400"))

TYPES = ["string", "number", "integer", "boolean", "array", "object", "null"]
SCALARS: list[Any] = ["a", "b", 1, 2, 1.5, True, False, None]


def gen(rng: random.Random, depth: int, defs: list[str]) -> Any:
    if depth > 3 or rng.random() < 0.15:
        return rng.choice(
            [{"type": rng.choice(TYPES[:4])}, {"type": "string", "format": "email"}, True, {}]
        )
    kind = rng.random()
    if kind < 0.3:
        props = {f"p{i}": gen(rng, depth + 1, defs) for i in range(rng.randint(0, 4))}
        node: dict[str, Any] = {"type": "object", "properties": props}
        if props and rng.random() < 0.6:
            node["required"] = rng.sample(sorted(props), rng.randint(0, len(props)))
        if rng.random() < 0.3:
            node["additionalProperties"] = rng.choice(
                [False, True, {"type": "string"}, gen(rng, depth + 1, defs)]
            )
    elif kind < 0.42:
        node = {"type": "array"}
        r = rng.random()
        if r < 0.6:
            node["items"] = gen(rng, depth + 1, defs)
        elif r < 0.8:
            node["prefixItems"] = [gen(rng, depth + 1, defs) for _ in range(rng.randint(1, 3))]
        if rng.random() < 0.4:
            node["minItems"] = rng.randint(0, 3)
        if rng.random() < 0.2:
            node["uniqueItems"] = True
    elif kind < 0.52:
        key = rng.choice(["anyOf", "oneOf", "allOf"])
        node = {key: [gen(rng, depth + 1, defs) for _ in range(rng.randint(1, 3))]}
        if rng.random() < 0.3:
            node["type"] = rng.choice(TYPES)
    elif kind < 0.6 and defs:
        node = {"$ref": "#/$defs/" + rng.choice(defs)}
        if rng.random() < 0.3:
            node["description"] = "ref with sibling"
    elif kind < 0.7:
        values = rng.sample(SCALARS, rng.randint(0, 4))
        node = {"enum": values} if rng.random() < 0.7 else {"const": rng.choice(SCALARS)}
        if rng.random() < 0.5:
            node["type"] = rng.choice(["string", "integer", ["string", "null"]])
    elif kind < 0.8:
        node = {"type": rng.sample(TYPES, rng.randint(1, 3))}
    else:
        node = {"type": rng.choice(["string", "integer", "number"])}
        for key in ("minimum", "exclusiveMaximum", "multipleOf", "minLength", "pattern"):
            if rng.random() < 0.3:
                node[key] = "^(?=a)b$" if key == "pattern" else rng.randint(0, 5)
    if isinstance(node, dict):
        for key in ("description", "default", "examples", "$comment", "title", "not"):
            if rng.random() < 0.08:
                node[key] = {"type": "string"} if key == "not" else "x"
    return node


def gen_root(rng: random.Random) -> Any:
    names = [f"D{i}" for i in range(rng.randint(0, 3))]
    root = gen(rng, 0, names)
    if isinstance(root, dict) and rng.random() < 0.8:
        root = (
            {**root, "type": "object"}
            if "properties" in root
            else {
                "type": "object",
                "properties": {"x": root},
            }
        )
    if names and isinstance(root, dict):
        root["$defs"] = {n: gen(rng, 1, names) for n in names}
    if isinstance(root, dict) and rng.random() < 0.2:
        root["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return root


SCHEMAS = [gen_root(random.Random(seed)) for seed in range(COUNT)]


@pytest.mark.parametrize("index", range(COUNT))
def test_fuzz(index: int) -> None:
    schema = SCHEMAS[index]
    before = copy.deepcopy(schema)
    for target in TARGETS:
        first = transform_schema(schema, target)
        assert schema == before
        second = transform_schema(first.schema, target)
        if first.ok:
            assert second.changes == [], (target, [c.to_dict() for c in second.changes])
            assert second.schema == first.schema
            restore_arguments({"x": "1", "p0": None}, schema, target)
        else:
            assert not second.ok, target

    gemini = transform_schema(schema, "gemini")
    if gemini.ok:
        types = pytest.importorskip("google.genai.types")
        types.Schema.model_validate(gemini.schema)
    strict = transform_schema(schema, "openai-strict")
    if strict.ok:
        strict_schema = pytest.importorskip("agents.strict_schema")
        try:
            converted = strict_schema.ensure_strict_json_schema(copy.deepcopy(strict.schema))
        except Exception as exc:  # pragma: no cover - reported with context below
            pytest.fail(f"openai-agents rejected our strict output: {exc}\n{strict.schema}")
        assert _sorted_required(converted) == _sorted_required(strict.schema)


def _sorted_required(node: Any) -> Any:
    if isinstance(node, dict):
        out = {k: _sorted_required(v) for k, v in node.items()}
        if isinstance(out.get("required"), list):
            out["required"] = sorted(out["required"])
        return out
    if isinstance(node, list):
        return [_sorted_required(v) for v in node]
    return node


# Junk values in every keyword position: nothing may raise.
_JUNK_KEYS = [
    "type",
    "format",
    "enum",
    "const",
    "$ref",
    "items",
    "prefixItems",
    "properties",
    "required",
    "additionalProperties",
    "anyOf",
    "oneOf",
    "allOf",
    "not",
    "minimum",
    "minItems",
    "pattern",
    "nullable",
    "description",
    "$defs",
    "default",
    "examples",
    "title",
]
_JUNK: list[Any] = [
    {},
    [],
    [1],
    {"a": 1},
    1,
    1.5,
    -3,
    True,
    False,
    None,
    "x",
    "",
    "#",
    "#/nope",
    ["string", 1],
    {"$ref": "#"},
    [{}],
    [[]],
    "a\n",
]


def _mutate(node: Any, rng: random.Random, depth: int = 0) -> Any:
    if isinstance(node, dict):
        node = dict(node)
        if rng.random() < 0.35:
            node[rng.choice(_JUNK_KEYS)] = copy.deepcopy(rng.choice(_JUNK))
        for key in list(node):
            if rng.random() < 0.5 and depth < 4:
                node[key] = _mutate(node[key], rng, depth + 1)
    elif isinstance(node, list):
        node = [_mutate(item, rng, depth + 1) for item in node]
    return node


@pytest.mark.parametrize("seed", range(COUNT))
def test_junk_never_raises(seed: int) -> None:
    rng = random.Random(seed + 10**6)
    schema = _mutate(gen_root(rng), rng)
    for target in TARGETS:
        transform_schema(schema, target)
        transform_tools([{"name": "t", "inputSchema": schema}], target)
        restore_arguments({"x": "1", "p0": None, "a": [1, {"b": None}]}, schema, target)
