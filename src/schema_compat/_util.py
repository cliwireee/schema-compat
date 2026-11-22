"""Shared machinery: JSON Pointers, local $ref resolution, merging, reporting."""

from __future__ import annotations

import contextlib
import copy
import json
from typing import Any
from urllib.parse import unquote

from ._model import Issue, Severity

# Upper bound on schema nodes processed per schema. Real schemas are far smaller;
# the limit stops pathological $ref fan-out (exponential expansion) from untrusted
# input such as tool schemas advertised by third-party MCP servers.
MAX_NODES = 100_000

# Keywords that describe a value without constraining it.
ANNOTATIONS = frozenset(
    {
        "title",
        "description",
        "$comment",
        "examples",
        "example",
        "default",
        "deprecated",
        "readOnly",
        "writeOnly",
    }
)

# Keywords that hold one subschema / a list of subschemas / a map of subschemas.
SINGLE_SUBSCHEMA_KEYS = (
    "items",
    "additionalProperties",
    "not",
    "contains",
    "propertyNames",
    "if",
    "then",
    "else",
    "unevaluatedProperties",
    "unevaluatedItems",
    "additionalItems",
)
LIST_SUBSCHEMA_KEYS = ("anyOf", "oneOf", "allOf", "prefixItems")
MAP_SUBSCHEMA_KEYS = ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas")


class RefError(ValueError):
    """A $ref that cannot be resolved inside the document."""


def escape(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def child(path: str, *tokens: str | int) -> str:
    return path + "".join("/" + escape(str(t)) for t in tokens)


def resolve_ref(root: Any, ref: str) -> Any:
    """Resolve a local JSON Pointer reference (``#`` or ``#/...``) against ``root``."""
    if ref == "#":
        return root
    if not ref.startswith("#/"):
        raise RefError(f"only local references ('#/...') are supported, got {ref!r}")
    node = root
    for raw in ref[2:].split("/"):
        token = unquote(raw).replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and token in node:
            node = node[token]
        elif isinstance(node, list) and token.isdigit() and int(token) < len(node):
            node = node[int(token)]
        else:
            raise RefError(f"reference {ref!r} does not resolve")
    return node


def resolve_schema(root: Any, ref: str) -> Any:
    """Like :func:`resolve_ref`, but a boolean ``true`` schema becomes ``{}`` (same meaning)."""
    target = resolve_ref(root, ref)
    return {} if target is True else target


def json_text(value: Any) -> str:
    """Deterministic JSON text, identical to the TypeScript port's output."""
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), sort_keys=False)


def fold_into_description(node: dict[str, Any], dropped: dict[str, Any]) -> None:
    """Append dropped constraints to the description so the model can still follow them.

    Same convention as the Anthropic SDK's ``transform_schema``:
    ``"<description>\\n\\n{minimum: 1, maximum: 10}"``.
    """
    if not dropped:
        return
    note = "{" + ", ".join(f"{k}: {json_text(v)}" for k, v in dropped.items()) + "}"
    description = node.get("description")
    if isinstance(description, str) and description:
        node["description"] = description + "\n\n" + note
    else:
        node["description"] = note


_LOWER_BOUNDS = ("minimum", "exclusiveMinimum", "minLength", "minItems", "minProperties")
_UPPER_BOUNDS = ("maximum", "exclusiveMaximum", "maxLength", "maxItems", "maxProperties")


def _types(value: Any) -> set[str] | None:
    if isinstance(value, str):
        return {value}
    if isinstance(value, list) and all(isinstance(t, str) for t in value):
        return set(value)
    return None


def merge_all(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any] | None:
    """Merge two schemas that must BOTH hold (``allOf`` semantics).

    Returns ``None`` when the merge cannot be expressed without changing the
    accepted values. Annotations from ``a`` win over those from ``b``.
    """
    out = dict(a)
    for key, bv in b.items():
        if key not in out:
            out[key] = copy.deepcopy(bv)
            continue
        av = out[key]
        if av == bv or key in ANNOTATIONS:
            continue
        if key == "required" and isinstance(av, list) and isinstance(bv, list):
            out[key] = av + [r for r in bv if r not in av]
        elif key == "properties" and isinstance(av, dict) and isinstance(bv, dict):
            props = dict(av)
            for name, schema in bv.items():
                if name not in props:
                    props[name] = copy.deepcopy(schema)
                elif props[name] != schema:
                    left = {} if props[name] is True else props[name]
                    right = {} if schema is True else schema
                    if not isinstance(left, dict) or not isinstance(right, dict):
                        return None
                    merged = merge_all(left, right)
                    if merged is None:
                        return None
                    props[name] = merged
            out[key] = props
        elif key == "type":
            at, bt = _types(av), _types(bv)
            if at is None or bt is None:
                return None
            both = at & bt
            if "integer" in at and "number" in bt or "number" in at and "integer" in bt:
                both.add("integer")
            if not both:
                return None
            ordered = [t for t in (av if isinstance(av, list) else [av]) if t in both]
            ordered += sorted(both - set(ordered))
            out[key] = ordered[0] if len(ordered) == 1 else ordered
        elif key == "enum" and isinstance(av, list) and isinstance(bv, list):
            both_enum = [v for v in av if v in bv]
            if not both_enum:
                return None
            out[key] = both_enum
        elif key in _LOWER_BOUNDS and _is_number(av) and _is_number(bv):
            out[key] = max(av, bv)
        elif key in _UPPER_BOUNDS and _is_number(av) and _is_number(bv):
            out[key] = min(av, bv)
        elif key == "additionalProperties" and (av is False or bv is False):
            out[key] = False
        else:
            return None
    return out


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_null_schema(schema: Any) -> bool:
    """``{"type": "null"}``, optionally with annotations."""
    return (
        isinstance(schema, dict)
        and schema.get("type") == "null"
        and all(k == "type" or k in ANNOTATIONS for k in schema)
    )


def find_recursive_refs(root: Any) -> list[tuple[str, str]]:
    """Return ``(path, ref)`` for every local $ref that closes a reference cycle."""
    found: dict[str, str] = {}
    clean: set[str] = set()

    def walk(node: Any, path: str, stack: tuple[str, ...]) -> None:
        if isinstance(node, list):
            for i, item in enumerate(node):
                walk(item, child(path, i), stack)
            return
        if not isinstance(node, dict):
            return
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#"):
            if ref in stack:
                found.setdefault(ref, path)
            elif ref not in clean:
                try:
                    target = resolve_ref(root, ref)
                except RefError:
                    target = None
                if target is not None:
                    before = len(found)
                    walk(target, ref[1:], (*stack, ref))
                    if len(found) == before:
                        clean.add(ref)
        for key, value in node.items():
            if key != "$ref":
                walk(value, child(path, key), stack)

    walk(root, "", ())
    return [(path, ref) for ref, path in found.items()]


def _collect_refs(node: Any, out: set[str]) -> None:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#"):
            out.add(ref)
        for key, value in node.items():
            if key != "$ref":
                _collect_refs(value, out)
    elif isinstance(node, list):
        for item in node:
            _collect_refs(item, out)


def cyclic_refs(root: Any) -> frozenset[str]:
    """Local references that can reach themselves through the definitions they point to.

    Merging (allOf) or collapsing such a reference would never terminate, so the
    targets refuse to; every other reference may be expanded as often as it is used.
    """
    refs: set[str] = set()
    _collect_refs(root, refs)
    edges: dict[str, set[str]] = {}
    for ref in refs:
        inner: set[str] = set()
        with contextlib.suppress(RefError):
            _collect_refs(resolve_ref(root, ref), inner)
        edges[ref] = inner
    cyclic: set[str] = set()
    for ref in refs:
        seen: set[str] = set()
        pending = list(edges.get(ref, ()))
        while pending:
            current = pending.pop()
            if current == ref:
                cyclic.add(ref)
                break
            if current not in seen:
                seen.add(current)
                pending.extend(edges.get(current, ()))
    return frozenset(cyclic)


class Report:
    """Collects repairs and unrepairable problems while a target processes a schema."""

    def __init__(self, tool: str | None = None) -> None:
        self.tool = tool
        self.changes: list[Issue] = []
        self.issues: list[Issue] = []
        self.nodes = 0
        self.exhausted = False

    def fix(
        self,
        code: str,
        path: str,
        message: str,
        *,
        lossy: bool = False,
        severity: Severity = "error",
    ) -> None:
        self.changes.append(
            Issue(code, path, message, severity, fixable=True, lossy=lossy, tool=self.tool)
        )

    def problem(self, code: str, path: str, message: str, severity: Severity = "error") -> None:
        self.issues.append(Issue(code, path, message, severity, tool=self.tool))

    def spend(self, path: str) -> bool:
        """Count one processed node; return False once the budget is exhausted."""
        self.nodes += 1
        if self.nodes > MAX_NODES:
            if not self.exhausted:
                self.exhausted = True
                self.problem(
                    "schema.too_large",
                    path,
                    f"schema expands to more than {MAX_NODES} nodes (likely $ref fan-out); "
                    "stopped processing",
                )
            return False
        return True


# Keywords whose boolean value is the idiomatic form; left as written.
_BOOLEAN_KEYWORDS = (
    "additionalProperties",
    "unevaluatedProperties",
    "unevaluatedItems",
    "additionalItems",
)


def true_to_empty(node: Any, path: str, report: Report, code: str) -> Any:
    """Replace every boolean ``true`` subschema with ``{}`` (identical meaning).

    Done once, before a target's rules run, so no merge or collapse step has to
    special-case booleans.
    """
    if node is True:
        report.fix(code, path, "boolean schema `true` replaced with `{}`")
        return {}
    if not isinstance(node, dict):
        return node
    out = dict(node)
    for key in SINGLE_SUBSCHEMA_KEYS:
        if key in out and key not in _BOOLEAN_KEYWORDS and not isinstance(out[key], list):
            out[key] = true_to_empty(out[key], child(path, key), report, code)
    for key in (*LIST_SUBSCHEMA_KEYS, "items"):
        if isinstance(out.get(key), list):
            out[key] = [
                true_to_empty(sub, child(path, key, i), report, code)
                for i, sub in enumerate(out[key])
            ]
    for key in MAP_SUBSCHEMA_KEYS:
        if isinstance(out.get(key), dict):
            out[key] = {
                name: true_to_empty(sub, child(path, key, name), report, code)
                for name, sub in out[key].items()
            }
    return out
