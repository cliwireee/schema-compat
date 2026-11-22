"""Undo the reversible, lossy repairs on the arguments a model sends back."""

from __future__ import annotations

from typing import Any

from ._model import TARGETS
from ._shared import allows_null
from ._util import RefError, json_text, resolve_ref

_MAX_DEPTH = 100


def _deref(schema: Any, root: Any) -> Any:
    seen: set[str] = set()
    while isinstance(schema, dict) and isinstance(schema.get("$ref"), str):
        ref = schema["$ref"]
        if ref in seen:
            return {}
        seen.add(ref)
        try:
            target = resolve_ref(root, ref)
        except RefError:
            return {}
        siblings = {k: v for k, v in schema.items() if k != "$ref"}
        schema = {**target, **siblings} if isinstance(target, dict) else target
    return schema


def _json_type_matches(value: Any, type_name: str) -> bool:
    if type_name == "null":
        return value is None
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "object":
        return isinstance(value, dict)
    return False


def _matches(value: Any, schema: Any, root: Any, target: str) -> bool:
    """Loose shape check used to pick the anyOf/oneOf branch a value belongs to."""
    schema = _deref(schema, root)
    if not isinstance(schema, dict):
        return schema is not False
    literals = list(schema["enum"]) if isinstance(schema.get("enum"), list) else []
    if "const" in schema:
        literals.append(schema["const"])
    if literals:
        if value in literals:
            return True
        return (
            target == "gemini"
            and isinstance(value, str)
            and any(not isinstance(v, str) and json_text(v) == value for v in literals)
        )
    type_value = schema.get("type")
    if isinstance(type_value, str):
        return _json_type_matches(value, type_value)
    if isinstance(type_value, list):
        return any(_json_type_matches(value, t) for t in type_value if isinstance(t, str))
    if isinstance(schema.get("properties"), dict):
        return isinstance(value, dict)
    if "items" in schema or "prefixItems" in schema:
        return isinstance(value, list)
    return True


def _walk(value: Any, schema: Any, root: Any, target: str, depth: int) -> Any:
    if depth > _MAX_DEPTH:
        return value
    schema = _deref(schema, root)
    if not isinstance(schema, dict):
        return value

    for entry in schema.get("allOf", []) if isinstance(schema.get("allOf"), list) else []:
        value = _walk(value, entry, root, target, depth + 1)
    for key in ("anyOf", "oneOf"):
        branches = schema.get(key)
        if isinstance(branches, list):
            for branch in branches:
                if _matches(value, branch, root, target):
                    value = _walk(value, branch, root, target, depth + 1)
                    break

    if target == "gemini" and isinstance(value, str):
        literals = list(schema["enum"]) if isinstance(schema.get("enum"), list) else []
        if "const" in schema:
            literals.append(schema["const"])
        if value not in literals:
            for literal in literals:
                if not isinstance(literal, str) and json_text(literal) == value:
                    return literal

    if isinstance(value, dict):
        raw_props = schema.get("properties")
        props: dict[str, Any] = raw_props if isinstance(raw_props, dict) else {}
        raw_required = schema.get("required")
        required: list[Any] = raw_required if isinstance(raw_required, list) else []
        out: dict[str, Any] = {}
        for key, item in value.items():
            sub = props.get(key)
            if (
                target == "openai-strict"
                and item is None
                and sub is not None
                and key not in required
                and not allows_null(_deref(sub, root))
            ):
                continue  # null sent only because strict mode made the property required
            out[key] = _walk(item, sub, root, target, depth + 1) if sub is not None else item
        return out
    if isinstance(value, list):
        prefix = schema.get("prefixItems")
        if not isinstance(prefix, list) and isinstance(schema.get("items"), list):
            prefix = schema["items"]
        items = schema.get("items") if isinstance(schema.get("items"), dict) else None
        result = []
        for i, item in enumerate(value):
            sub = prefix[i] if isinstance(prefix, list) and i < len(prefix) else items
            result.append(_walk(item, sub, root, target, depth + 1) if sub is not None else item)
        return result
    return value


def restore_arguments(arguments: Any, schema: Any, target: str) -> Any:
    """Map tool-call arguments produced against a transformed schema back to the original.

    ``schema`` is the ORIGINAL schema you passed to :func:`transform_schema`.

    - ``gemini``: enum/const values that were sent as strings (``"1"``,
      ``"true"``) become the original numbers/booleans again.
    - ``openai-strict``: properties that were optional in the original schema and
      came back as ``null`` are removed, unless the original schema itself allowed
      ``null`` for them (then ``null`` is kept: it cannot be told apart).

    Other targets have no reversible repairs; the arguments are returned
    unchanged (as a new object). The input is never mutated.
    """
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; expected one of {', '.join(TARGETS)}")
    return _walk(arguments, schema, schema, target, 0)
