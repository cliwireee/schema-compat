"""Rules shared by several targets."""

from __future__ import annotations

from typing import Any

from ._util import Report, is_null_schema


def fix_root_object(node: Any, report: Report, family: str) -> Any:
    """The root of a tool's parameters must be ``{"type": "object", ...}``.

    Anthropic: ``input_schema.type`` is ``Required[Literal["object"]]`` in the SDK.
    OpenAI: "schema must be a JSON Schema of 'type: \"object\"'".
    """
    if not isinstance(node, dict):
        report.problem(f"{family}.root_not_object", "", "the parameters schema must be an object")
        return node
    type_value = node.get("type")
    if type_value is None:
        report.fix(f"{family}.root_type", "", 'the root schema needs type: "object"; added')
        return {"type": "object", **node}
    if type_value == "object":
        return node
    if isinstance(type_value, list) and "object" in type_value:
        report.fix(
            f"{family}.root_type",
            "/type",
            f'the root schema must be type: "object", not {type_value}; narrowed',
        )
        return {**node, "type": "object"}
    report.problem(
        f"{family}.root_not_object",
        "/type",
        f'the root schema must be type: "object", got {type_value!r}',
    )
    return node


def allows_null(schema: Any) -> bool:
    """True when ``schema`` visibly accepts ``null`` (conservative)."""
    if not isinstance(schema, dict):
        return schema is True
    type_value = schema.get("type")
    if type_value == "null" or (isinstance(type_value, list) and "null" in type_value):
        return True
    if schema.get("nullable") is True:
        return True
    any_of = schema.get("anyOf")
    if isinstance(any_of, list) and any(is_null_schema(b) for b in any_of):
        return True
    enum = schema.get("enum")
    if isinstance(enum, list) and None in enum:
        return True
    if "const" in schema and schema["const"] is None:
        return True
    constraining = {"type", "enum", "const", "anyOf", "oneOf", "allOf", "$ref", "not"}
    return not (constraining & set(schema))


def make_nullable(schema: Any) -> Any:
    """Return ``schema`` widened to also accept ``null`` (OpenAI's pattern for optional fields)."""
    if not isinstance(schema, dict) or allows_null(schema):
        return schema
    wrap_keys = {"$ref", "const", "allOf", "oneOf", "not"}
    if wrap_keys & set(schema):
        return {"anyOf": [schema, {"type": "null"}]}
    out = dict(schema)
    type_value = out.get("type")
    if isinstance(type_value, str):
        out["type"] = [type_value, "null"]
    elif isinstance(type_value, list):
        out["type"] = [*type_value, "null"]
    if isinstance(out.get("anyOf"), list):
        out["anyOf"] = [*out["anyOf"], {"type": "null"}]
    if isinstance(out.get("enum"), list):
        out["enum"] = [*out["enum"], None]
    return out
