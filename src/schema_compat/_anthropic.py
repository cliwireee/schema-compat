"""Anthropic tool ``input_schema`` and strict tool use (``strict: true``).

Sources:
- https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use
  (name ``^[a-zA-Z0-9_-]{1,128}$``; ``input_schema`` is a JSON Schema object) and
  the SDK's ``InputSchemaTyped`` (``type: Required[Literal["object"]]``).
- https://platform.claude.com/docs/en/build-with-claude/structured-outputs
  ("JSON Schema limitations", "Schema complexity limits"), shared by JSON outputs
  and strict tool use.
"""

from __future__ import annotations

import copy
from typing import Any

from ._shared import fix_root_object
from ._util import (
    ANNOTATIONS,
    RefError,
    Report,
    child,
    cyclic_refs,
    find_recursive_refs,
    fold_into_description,
    merge_all,
    resolve_ref,
    resolve_schema,
    true_to_empty,
)

SUPPORTED_FORMATS = frozenset(
    {"date-time", "time", "date", "duration", "email", "hostname", "uri", "ipv4", "ipv6", "uuid"}
)
# Unsupported constraints. Dropping them loosens validation, so they are appended
# to the description (as the Anthropic SDK's transform_schema does).
NUMERIC_CONSTRAINTS = ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf")
STRING_CONSTRAINTS = ("minLength", "maxLength")
ARRAY_CONSTRAINTS = (
    "maxItems",
    "uniqueItems",
    "contains",
    "minContains",
    "maxContains",
    "additionalItems",
    "unevaluatedItems",
)

# Explicit per-request limits ("Schema complexity limits").
MAX_STRICT_TOOLS = 20
MAX_OPTIONAL_PARAMETERS = 24
MAX_UNION_PARAMETERS = 16


def unsupported_regex_features(pattern: str) -> list[str]:
    """Regex features the docs list as unsupported: backreferences, lookaround, \\b/\\B."""
    found: list[str] = []
    i, n, in_class = 0, len(pattern), False
    while i < n:
        c = pattern[i]
        if c == "\\" and i + 1 < n:
            nxt = pattern[i + 1]
            if not in_class:
                if nxt in "123456789" or nxt == "k" and i + 2 < n and pattern[i + 2] in "<{'":
                    found.append("backreference")
                elif nxt in "bB":
                    found.append("word boundary")
            i += 2
            continue
        if in_class:
            if c == "]":
                in_class = False
            i += 1
            continue
        if c == "[":
            in_class = True
            i += 1
            if i < n and pattern[i] == "^":
                i += 1
            if i < n and pattern[i] == "]":
                i += 1  # a leading "]" is a literal
            continue
        if c == "(" and pattern.startswith(("(?=", "(?!", "(?<=", "(?<!"), i):
            found.append("lookaround")
        i += 1
    return sorted(set(found))


class AnthropicTransformer:
    """Non-strict tools accept JSON Schema (draft 2020-12); only the root is constrained."""

    def __init__(self, schema: Any, report: Report) -> None:
        self.schema = copy.deepcopy(schema)
        self.report = report

    def run(self) -> Any:
        return fix_root_object(self.schema, self.report, "anthropic")


class AnthropicStrictTransformer:
    def __init__(self, schema: Any, report: Report) -> None:
        self.report = report
        self.original = true_to_empty(
            copy.deepcopy(schema), "", report, "anthropic_strict.boolean_schema"
        )
        self.cyclic = cyclic_refs(self.original)

    def run(self) -> Any:
        r = self.report
        root = copy.deepcopy(self.original)
        if isinstance(root, dict) and "type" not in root and isinstance(root.get("anyOf"), list):
            r.problem(
                "anthropic_strict.root_not_object",
                "",
                'the root of input_schema must be type: "object", not anyOf',
            )
            return root
        root = fix_root_object(root, r, "anthropic_strict")
        if not isinstance(root, dict):
            return root
        for path, ref in find_recursive_refs(self.original):
            r.problem(
                "anthropic_strict.recursive_schema",
                path,
                f"recursive schemas are not supported ({ref} refers back to itself)",
            )
        return self.visit(root, "")

    def _deref(self, schema: Any, stack: tuple[str, ...]) -> Any:
        if not isinstance(schema, dict) or "$ref" not in schema:
            return schema
        ref = schema["$ref"]
        if not isinstance(ref, str) or ref in stack:
            return None  # a self-referencing allOf cannot be merged
        try:
            target = resolve_schema(self.original, ref)
        except RefError:
            return None
        if not isinstance(target, dict):
            return None
        siblings = {k: v for k, v in schema.items() if k != "$ref"}
        return merge_all(siblings, copy.deepcopy(target))

    def _probe(self, entries: list[Any], path: str, stack: tuple[str, ...]) -> None:
        """Collect unrepairable problems inside ``entries`` without changing them."""
        real = self.report
        self.report = Report(tool=real.tool)
        self.report.nodes = real.nodes
        try:
            for i, sub in enumerate(entries):
                self.visit(copy.deepcopy(sub), child(path, "allOf", i), stack)
        finally:
            probe, self.report = self.report, real
        real.nodes = probe.nodes
        real.issues.extend(probe.issues)

    def visit(self, node: Any, path: str, stack: tuple[str, ...] = ()) -> Any:
        r = self.report
        if node is True:
            r.fix(
                "anthropic_strict.boolean_schema", path, "boolean schema `true` replaced with `{}`"
            )
            return {}
        if not isinstance(node, dict):
            r.problem("schema.invalid", path, "schema must be a JSON object")
            return node
        if not r.spend(path):
            return node
        node = dict(node)
        dropped: dict[str, Any] = {}

        ref = node.get("$ref")
        if ref is not None:
            if not isinstance(ref, str) or not ref.startswith("#"):
                r.problem(
                    "anthropic_strict.external_ref",
                    child(path, "$ref"),
                    f"external $ref is not supported: {ref!r}",
                )
            else:
                try:
                    resolve_ref(self.original, ref)
                except RefError as exc:
                    r.problem("schema.unresolvable_ref", child(path, "$ref"), str(exc))

        if "oneOf" in node:
            if "anyOf" in node:
                r.problem(
                    "anthropic_strict.oneof",
                    child(path, "oneOf"),
                    "oneOf is not supported and cannot become anyOf because anyOf is already set",
                )
            else:
                node["anyOf"] = node.pop("oneOf")
                r.fix(
                    "anthropic_strict.oneof",
                    child(path, "oneOf"),
                    "oneOf is not supported (use anyOf); replaced by anyOf (no longer exclusive)",
                    lossy=True,
                )

        if "not" in node:
            dropped["not"] = node.pop("not")
            r.fix(
                "anthropic_strict.not",
                child(path, "not"),
                '"not" is not supported; removed and added to the description',
                lossy=True,
            )

        if "allOf" in node and isinstance(node["allOf"], list):
            entries = node.pop("allOf")
            merged: dict[str, Any] | None = node
            for entry in entries:
                # `true` accepts everything: neutral in an allOf.
                resolved = {} if entry is True else self._deref(entry, stack)
                if isinstance(entry, dict) and entry.get("$ref") in self.cyclic:
                    resolved = None  # merging a self-referencing definition never ends
                merged = (
                    merge_all(merged, resolved)
                    if merged is not None and isinstance(resolved, dict)
                    else None
                )
            if merged is None:
                # Report what else is wrong inside, but leave the entries as written: the
                # node is unrepairable, and rewriting its entries would hide the conflict.
                node["allOf"] = entries
                self._probe(entries, path, stack)
                r.problem(
                    "anthropic_strict.allof",
                    child(path, "allOf"),
                    "allOf with conflicting schemas is not supported and they cannot be merged",
                )
            else:
                r.fix(
                    "anthropic_strict.allof", child(path, "allOf"), "allOf merged into one schema"
                )
                return self.visit(merged, path, stack)

        for key in ("enum", "const"):
            if key in node:
                values = node[key] if key == "enum" else [node[key]]
                if not isinstance(values, list) or any(isinstance(v, (dict, list)) for v in values):
                    r.problem(
                        "anthropic_strict.complex_enum",
                        child(path, key),
                        f"{key} values must be strings, numbers, booleans or null",
                    )

        for key in (*NUMERIC_CONSTRAINTS, *STRING_CONSTRAINTS, *ARRAY_CONSTRAINTS):
            if key in node:
                dropped[key] = node.pop(key)
                r.fix(
                    "anthropic_strict.unsupported_constraint",
                    child(path, key),
                    f'"{key}" is not supported; removed and added to the description',
                    lossy=True,
                )
        if "minItems" in node and node["minItems"] not in (0, 1):
            dropped["minItems"] = node.pop("minItems")
            r.fix(
                "anthropic_strict.unsupported_constraint",
                child(path, "minItems"),
                "only minItems 0 or 1 is supported; removed and added to the description",
                lossy=True,
            )
        if "format" in node and node["format"] not in SUPPORTED_FORMATS:
            dropped["format"] = node.pop("format")
            r.fix(
                "anthropic_strict.format",
                child(path, "format"),
                f"string format {dropped['format']!r} is not supported; removed and added to the "
                "description",
                lossy=True,
            )
        if isinstance(node.get("pattern"), str):
            features = unsupported_regex_features(node["pattern"])
            if features:
                dropped["pattern"] = node.pop("pattern")
                r.fix(
                    "anthropic_strict.pattern",
                    child(path, "pattern"),
                    f"regex uses unsupported features ({', '.join(features)}); removed and added "
                    "to the description",
                    lossy=True,
                )

        tuple_items = None
        if isinstance(node.get("items"), list):
            tuple_items = node.pop("items")
        if isinstance(node.get("prefixItems"), list):
            tuple_items = node.pop("prefixItems")
        if tuple_items is not None:
            schemas = [s for s in tuple_items if isinstance(s, dict)]
            if isinstance(node.get("items"), dict) and node["items"]:
                schemas.append(node["items"])
            if schemas:
                node["items"] = schemas[0] if len(schemas) == 1 else {"anyOf": schemas}
            r.fix(
                "anthropic_strict.tuple_items",
                path,
                "tuple arrays are an array constraint that is not supported; items became anyOf "
                "of the positions",
                lossy=True,
            )

        fold_into_description(node, dropped)

        # After unsupported constraints are gone: only real siblings are distributed.
        any_of = node.get("anyOf")
        if isinstance(any_of, list):
            siblings = {k: v for k, v in node.items() if k != "anyOf" and k not in ANNOTATIONS}
            if siblings and path == "":
                r.problem(
                    "anthropic_strict.anyof_siblings",
                    child(path, "anyOf"),
                    'the root must be type: "object" and anyOf cannot sit next to type; move the '
                    "anyOf into a property",
                )
            elif siblings:
                branches = [merge_all(siblings, b) if isinstance(b, dict) else None for b in any_of]
                if any(b is None for b in branches):
                    r.problem(
                        "anthropic_strict.anyof_siblings",
                        child(path, "anyOf"),
                        "anyOf cannot sit next to keywords such as type or properties, and they "
                        "conflict with its branches",
                    )
                else:
                    node = {k: v for k, v in node.items() if k not in siblings}
                    node["anyOf"] = branches
                    r.fix(
                        "anthropic_strict.anyof_siblings",
                        child(path, "anyOf"),
                        f"anyOf cannot sit next to {sorted(siblings)}; moved them into each branch",
                    )

        type_value = node.get("type")
        is_object = (
            type_value == "object"
            or (isinstance(type_value, list) and "object" in type_value)
            or (type_value is None and isinstance(node.get("properties"), dict))
        )
        if is_object:
            if "additionalProperties" not in node:
                node["additionalProperties"] = False
                r.fix(
                    "anthropic_strict.additional_properties",
                    path,
                    "objects must set additionalProperties: false; added",
                )
            elif node["additionalProperties"] is not False:
                r.problem(
                    "anthropic_strict.additional_properties",
                    child(path, "additionalProperties"),
                    "additionalProperties must be false; for arbitrary keys use an array of "
                    "{key, value} objects",
                )
            if not node.get("properties"):
                r.problem(
                    "anthropic_strict.empty_object",
                    path,
                    "an object without properties can only ever be {} under strict mode",
                    severity="warning",
                )

        props = node.get("properties")
        if isinstance(props, dict):
            node["properties"] = {
                name: self.visit(sub, child(path, "properties", name), stack)
                for name, sub in props.items()
            }
        if isinstance(node.get("items"), (dict, bool)):
            node["items"] = self.visit(node["items"], child(path, "items"), stack)
        for key in ("anyOf",):
            if isinstance(node.get(key), list):
                node[key] = [
                    self.visit(sub, child(path, key, i), stack) for i, sub in enumerate(node[key])
                ]
        for key in ("$defs", "definitions"):
            if isinstance(node.get(key), dict):
                node[key] = {
                    name: self.visit(sub, child(path, key, name), stack)
                    for name, sub in node[key].items()
                }
        return node


def complexity(schema: Any) -> tuple[int, int]:
    """(optional parameters, union-typed parameters) in one strict schema, as documented."""
    optional = 0
    unions = 0

    def walk(node: Any) -> None:
        nonlocal optional, unions
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return
        props = node.get("properties")
        if isinstance(props, dict):
            raw_required = node.get("required")
            required: list[Any] = raw_required if isinstance(raw_required, list) else []
            for name, sub in props.items():
                if name not in required:
                    optional += 1
                if isinstance(sub, dict) and (
                    isinstance(sub.get("anyOf"), list) or isinstance(sub.get("type"), list)
                ):
                    unions += 1
        for value in node.values():
            walk(value)

    walk(schema)
    return optional, unions
