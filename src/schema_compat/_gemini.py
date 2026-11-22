"""Gemini ``functionDeclarations[].parameters`` (OpenAPI 3.0 Schema subset).

Sources:
- google-genai SDK ``types.Schema`` (the accepted fields; the model uses
  ``extra='forbid'``) and ``_transformers.process_schema`` / ``handle_null_fields``
  (inlined ``$defs``, ``nullable`` instead of ``{"type": "null"}``, ``const`` to
  ``enum``, integer enums sent as strings).
- Real API errors reported on GitHub: ``Invalid JSON payload received. Unknown
  name "..."``, ``Proto field is not repeating`` (type arrays), ``should be
  non-empty for OBJECT type``, ``only allowed for STRING type`` (enums),
  ``items: missing field``.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from ._util import (
    RefError,
    Report,
    child,
    cyclic_refs,
    fold_into_description,
    is_null_schema,
    json_text,
    merge_all,
    resolve_schema,
    true_to_empty,
)

# Fields of google-genai ``types.Schema`` in their JSON (camelCase) form. ``$ref``
# and ``$defs`` are handled separately (inlined), as the SDK does, and
# ``additionalProperties`` is rejected by the Gemini Developer API.
ALLOWED = frozenset(
    {
        "type",
        "format",
        "title",
        "description",
        "nullable",
        "enum",
        "maxItems",
        "minItems",
        "properties",
        "required",
        "minProperties",
        "maxProperties",
        "minLength",
        "maxLength",
        "pattern",
        "example",
        "anyOf",
        "propertyOrdering",
        "default",
        "items",
        "minimum",
        "maximum",
    }
)

# Dropping these changes which values are valid, so the repair is lossy and the
# constraint is appended to the description for the model to follow.
VALIDATION_KEYWORDS = frozenset(
    {
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "uniqueItems",
        "contains",
        "minContains",
        "maxContains",
        "propertyNames",
        "patternProperties",
        "dependentRequired",
        "dependentSchemas",
        "dependencies",
        "if",
        "then",
        "else",
        "not",
        "unevaluatedProperties",
        "unevaluatedItems",
        "additionalItems",
        "$dynamicRef",
        "$recursiveRef",
    }
)

VALID_TYPES = frozenset({"string", "number", "integer", "boolean", "array", "object", "null"})

# Keywords that only apply to one instance type, used when a type array is split
# into ``anyOf`` branches.
_TYPE_KEYWORDS = {
    "object": (
        "properties",
        "required",
        "additionalProperties",
        "minProperties",
        "maxProperties",
        "propertyOrdering",
        "patternProperties",
        "propertyNames",
        "dependentRequired",
        "dependentSchemas",
        "unevaluatedProperties",
    ),
    "array": (
        "items",
        "prefixItems",
        "minItems",
        "maxItems",
        "uniqueItems",
        "contains",
        "minContains",
        "maxContains",
        "unevaluatedItems",
        "additionalItems",
    ),
    "string": ("minLength", "maxLength", "pattern"),
    "number": ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"),
    "integer": ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"),
}
_NUMERIC_FORMATS = frozenset({"float", "double", "int32", "int64"})

# Gemini: parameter names "must start with a letter or an underscore and must only
# contain chars a-z, A-Z, 0-9, or underscores with a maximum length of 64".
PARAMETER_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


def _ref_in(ref: Any, refs: frozenset[str]) -> bool:
    return isinstance(ref, str) and ref in refs


def _type_name(value: Any) -> str | None:
    return value.lower() if isinstance(value, str) else None


def _enum_matches(value: Any, type_name: str) -> bool:
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "integer":
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
            or (isinstance(value, float) and value.is_integer())
        )
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "object":
        return isinstance(value, dict)
    return False


class GeminiTransformer:
    def __init__(self, schema: Any, report: Report) -> None:
        self.report = report
        self.original = true_to_empty(copy.deepcopy(schema), "", report, "gemini.boolean_schema")
        self.cyclic = cyclic_refs(self.original)

    def run(self) -> Any:
        out = self.visit(copy.deepcopy(self.original), "", ())
        # Root definitions are only needed while a $ref remains, which happens only
        # for (unrepairable) recursive references; otherwise Gemini rejects them.
        if isinstance(out, dict) and not any(
            i.code == "gemini.recursive_ref" for i in self.report.issues
        ):
            for key in ("$defs", "definitions"):
                if key in out:
                    del out[key]
                    self.report.fix(
                        "gemini.unsupported_keyword",
                        child("", key),
                        f'"{key}" is not supported (definitions are inlined); removed',
                    )
        return out

    # -- helpers ---------------------------------------------------------

    def _deref(self, schema: Any, stack: tuple[str, ...]) -> Any:
        """Resolve a top-level ``$ref`` (merging its siblings); None if impossible."""
        if not isinstance(schema, dict) or "$ref" not in schema:
            return schema
        ref = schema["$ref"]
        if not isinstance(ref, str) or ref in stack:
            return None
        try:
            target = resolve_schema(self.original, ref)
        except RefError:
            return None
        if not isinstance(target, dict):
            return None
        siblings = {k: v for k, v in schema.items() if k != "$ref"}
        merged = merge_all(siblings, copy.deepcopy(target))
        if merged is None:
            return None
        # The target may itself be a reference.
        return self._deref(merged, (*stack, ref))

    def _split_types(self, node: dict[str, Any], types: list[str]) -> dict[str, Any]:
        branches: list[dict[str, Any]] = []
        enum = node.pop("enum", None)
        fmt = node.pop("format", None)
        moved: set[str] = set()
        for type_name in types:
            branch: dict[str, Any] = {"type": type_name}
            for key in _TYPE_KEYWORDS.get(type_name, ()):
                if key in node:
                    branch[key] = copy.deepcopy(node[key])
                    moved.add(key)
            if fmt is not None:
                numeric = type_name in ("number", "integer")
                if (isinstance(fmt, str) and fmt in _NUMERIC_FORMATS) == numeric and type_name in (
                    "string",
                    "number",
                    "integer",
                ):
                    branch["format"] = fmt
            if isinstance(enum, list):
                values = [v for v in enum if _enum_matches(v, type_name)]
                if not values:
                    continue  # no value of this type is allowed: drop the branch
                branch["enum"] = values
            branches.append(branch)
        for key in moved:
            node.pop(key, None)
        node.pop("type", None)
        node["anyOf"] = branches
        return node

    # -- main visitor ----------------------------------------------------

    def visit(self, node: Any, path: str, stack: tuple[str, ...]) -> Any:
        r = self.report
        if node is True:
            r.fix("gemini.boolean_schema", path, "boolean schema `true` replaced with `{}`")
            return {}
        if not isinstance(node, dict):
            r.problem("schema.invalid", path, "schema must be a JSON object")
            return node
        if not r.spend(path):
            return node
        node = dict(node)
        keep: set[str] = set()

        # 1. References: Gemini needs definitions inlined (as the google-genai SDK does).
        if "$ref" in node:
            ref = node["$ref"]
            if not isinstance(ref, str):
                r.problem("schema.invalid", child(path, "$ref"), "$ref must be a string")
                return node
            if ref in stack:
                r.problem(
                    "gemini.recursive_ref",
                    path,
                    f"recursive {ref} cannot be inlined, and Gemini's parameters schema has no "
                    "$ref; use parametersJsonSchema or flatten the recursion",
                )
                return node
            try:
                target = resolve_schema(self.original, ref)
            except RefError as exc:
                r.problem("schema.unresolvable_ref", child(path, "$ref"), str(exc))
                return node
            siblings = {k: v for k, v in node.items() if k != "$ref"}
            inlined = (
                merge_all(siblings, copy.deepcopy(target)) if isinstance(target, dict) else None
            )
            if inlined is None:
                r.problem(
                    "gemini.ref_conflict",
                    path,
                    f"cannot inline {ref}: its keywords conflict with the sibling keywords",
                )
                return node
            r.fix("gemini.ref_inlined", path, f"inlined {ref} (Gemini needs definitions inlined)")
            return self.visit(inlined, path, (*stack, ref))

        # 2. allOf: merge into one schema.
        if "allOf" in node:
            entries = node.pop("allOf")
            merged: dict[str, Any] | None = node
            if isinstance(entries, list):
                for entry in entries:
                    # `true` accepts everything: neutral in an allOf.
                    resolved = {} if entry is True else self._deref(entry, stack)
                    if isinstance(entry, dict) and _ref_in(entry.get("$ref"), self.cyclic):
                        resolved = None  # merging a self-referencing definition never ends
                    if merged is None or not isinstance(resolved, dict):
                        merged = None
                        break
                    merged = merge_all(merged, resolved)
            else:
                merged = None
            if merged is None:
                node["allOf"] = entries
                r.problem(
                    "gemini.allof_unmergeable",
                    child(path, "allOf"),
                    "allOf is not supported and its schemas conflict, so they cannot be merged",
                )
                return node  # leave the conflicting node as written
            else:
                r.fix("gemini.allof_merged", child(path, "allOf"), "allOf merged into one schema")
                return self.visit(merged, path, stack)

        # 3. oneOf -> anyOf.
        if "oneOf" in node:
            if "anyOf" in node:
                keep.add("oneOf")
                r.problem(
                    "gemini.oneof_with_anyof",
                    child(path, "oneOf"),
                    "oneOf is not supported and cannot become anyOf because anyOf is already set",
                )
            else:
                node["anyOf"] = node.pop("oneOf")
                r.fix(
                    "gemini.oneof_to_anyof",
                    child(path, "oneOf"),
                    "oneOf is not supported; replaced by anyOf (no longer exclusive)",
                    lossy=True,
                )

        # 4. const -> enum (the google-genai SDK does the same).
        if "const" in node:
            const = node.pop("const")
            enum = node.get("enum")
            node["enum"] = [v for v in enum if v == const] if isinstance(enum, list) else [const]
            r.fix(
                "gemini.const_to_enum", child(path, "const"), "const is not supported; became enum"
            )

        # 5. Types: no type arrays and no "null" type; Gemini uses `nullable`.
        type_value = node.get("type")
        if isinstance(type_value, list):
            if not all(isinstance(t, str) for t in type_value):
                r.problem("schema.invalid", child(path, "type"), "type array must contain strings")
            else:
                names = list(dict.fromkeys(type_value))
                non_null = [t for t in names if t.lower() != "null"]
                if len(non_null) > 1 and "anyOf" in node:
                    r.problem(
                        "gemini.type_array",
                        child(path, "type"),
                        "type arrays are not supported and cannot become anyOf because anyOf "
                        "is already set",
                    )
                else:
                    if len(non_null) < len(names):
                        node["nullable"] = True
                    if not non_null:
                        node.pop("type")
                    elif len(non_null) == 1:
                        node["type"] = non_null[0]
                    else:
                        node = self._split_types(node, non_null)
                    r.fix(
                        "gemini.type_array",
                        child(path, "type"),
                        'type arrays are not supported ("Proto field is not repeating"); '
                        "rewritten with nullable/anyOf",
                        # A null-only type becomes "untyped but nullable": looser.
                        lossy=not non_null,
                    )
        elif _type_name(type_value) == "null":
            node.pop("type")
            node["nullable"] = True
            r.fix(
                "gemini.type_null",
                child(path, "type"),
                'type "null" became nullable: true (Gemini has no null-only type)',
                lossy=True,
            )
        elif type_value is not None and _type_name(type_value) not in VALID_TYPES:
            r.problem("schema.unknown_type", child(path, "type"), f"unknown type {type_value!r}")

        # 6. anyOf: a {"type": "null"} branch becomes nullable (as the SDK does).
        any_of = node.get("anyOf")
        if isinstance(any_of, list) and any(is_null_schema(b) for b in any_of):
            rest = [b for b in any_of if not is_null_schema(b)]
            node["nullable"] = True
            r.fix(
                "gemini.null_branch",
                child(path, "anyOf"),
                'anyOf {"type": "null"} became nullable',
                lossy=not rest,  # anyOf of only null: now untyped but nullable
            )
            if not rest:
                node.pop("anyOf")
            elif len(rest) == 1:
                base = {k: v for k, v in node.items() if k != "anyOf"}
                resolved = self._deref(rest[0], stack)
                if isinstance(rest[0], dict) and _ref_in(rest[0].get("$ref"), self.cyclic):
                    resolved = None
                collapsed = merge_all(base, resolved) if isinstance(resolved, dict) else None
                if collapsed is not None:
                    return self.visit(collapsed, path, stack)
                node["anyOf"] = rest
            else:
                node["anyOf"] = rest

        # 7. enum: non-empty, string values only.
        if "enum" in node:
            enum = node["enum"]
            if isinstance(enum, list) and None in enum:
                node["enum"] = enum = [v for v in enum if v is not None]
                node["nullable"] = True
                r.fix("gemini.enum_null", child(path, "enum"), "null in enum became nullable: true")
            if not isinstance(enum, list) or not enum:
                r.problem("gemini.empty_enum", child(path, "enum"), "Gemini rejects empty enums")
            elif any(not isinstance(v, str) for v in enum):
                node["enum"] = list(
                    dict.fromkeys(v if isinstance(v, str) else json_text(v) for v in enum)
                )
                r.fix(
                    "gemini.enum_stringified",
                    child(path, "enum"),
                    'enum values must be strings ("only allowed for STRING type"); non-string '
                    "values were converted, restore_arguments() converts them back",
                    lossy=True,
                )

        # 8. examples -> example.
        if "examples" in node and "example" not in node:
            examples = node["examples"]
            if isinstance(examples, list) and examples:
                node["example"] = examples[0]

        # 9. additionalProperties is rejected by the Gemini Developer API.
        if "additionalProperties" in node:
            extra = node.pop("additionalProperties")
            lossy = extra is not True and extra != {}
            if isinstance(extra, dict) and extra:
                fold_into_description(node, {"additionalProperties": extra})
            r.fix(
                "gemini.additional_properties",
                child(path, "additionalProperties"),
                "additionalProperties is not supported by the Gemini API; removed",
                lossy=lossy,
            )

        # 10. Tuple arrays (prefixItems / items as a list) -> items: anyOf.
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
                "gemini.tuple_items",
                path,
                "tuple arrays (prefixItems) are not supported; items became anyOf of the positions",
                lossy=True,
            )

        # 11. Remove every remaining keyword Gemini does not know.
        dropped: dict[str, Any] = {}
        for key in list(node):
            if key in ALLOWED or key in keep:
                continue
            if key in ("$defs", "definitions") and path == "":
                continue  # decided in run(), once all references are known
            value = node.pop(key)
            if key in VALIDATION_KEYWORDS:
                dropped[key] = value
                r.fix(
                    "gemini.unsupported_keyword",
                    child(path, key),
                    f'"{key}" is not supported (Invalid JSON payload: Unknown name); removed and '
                    "added to the description",
                    lossy=True,
                )
            else:
                r.fix(
                    "gemini.unsupported_keyword",
                    child(path, key),
                    f'"{key}" is not supported (Invalid JSON payload: Unknown name); removed',
                )
        fold_into_description(node, dropped)

        # 12. Recurse.
        props = node.get("properties")
        if isinstance(props, dict):
            node["properties"] = {
                name: self.visit(sub, child(path, "properties", name), stack)
                for name, sub in props.items()
            }
        elif props is not None:
            r.problem("schema.invalid", child(path, "properties"), "properties must be an object")
        items = node.get("items")
        if isinstance(items, (dict, bool)):
            node["items"] = self.visit(items, child(path, "items"), stack)
        any_of = node.get("anyOf")
        if isinstance(any_of, list):
            node["anyOf"] = [
                self.visit(sub, child(path, "anyOf", i), stack) for i, sub in enumerate(any_of)
            ]

        # 13. Structural requirements.
        type_name = _type_name(node.get("type"))
        if type_name == "object" and not node.get("properties"):
            if path == "":
                r.problem(
                    "gemini.empty_object",
                    path,
                    'OBJECT schemas need at least one property ("should be non-empty for OBJECT '
                    'type"); for a tool without parameters, omit the parameters field '
                    "(transform_tools does this)",
                )
            else:
                r.problem(
                    "gemini.empty_object",
                    path,
                    'OBJECT schemas need at least one property ("should be non-empty for OBJECT '
                    'type"); Gemini cannot express free-form objects or maps',
                )
        if type_name == "array" and "items" not in node:
            r.problem(
                "gemini.array_without_items",
                path,
                'ARRAY schemas need "items" ("items: missing field")',
            )
        if path == "" and isinstance(node.get("properties"), dict):
            for name in node["properties"]:
                if not PARAMETER_NAME.match(name):
                    r.problem(
                        "gemini.parameter_name",
                        child(path, "properties", name),
                        f"parameter name {name!r} must match {PARAMETER_NAME.pattern}",
                    )
        return node
