"""OpenAI function calling (``parameters``) and strict mode (``strict: true``).

Sources:
- openai-python ``FunctionDefinition``: name "must be a-z, A-Z, 0-9, or contain
  underscores and dashes, with a maximum length of 64".
- openai-python ``lib/_pydantic.to_strict_json_schema`` and openai-agents
  ``strict_schema.ensure_strict_json_schema`` (the strict-mode rules OpenAI's own
  SDKs apply): ``additionalProperties: false`` on every object, every property
  required, ``$ref`` with sibling keywords inlined, ``oneOf`` -> ``anyOf``,
  ``default: null`` removed, the root must be a non-nullable object and must not
  be ``anyOf``.
- API errors reported on GitHub: "schema must be a JSON Schema of 'type:
  \"object\"'", "'additionalProperties' is required to be supplied and to be
  false".
"""

from __future__ import annotations

import copy
from typing import Any

from ._shared import allows_null, fix_root_object, make_nullable
from ._util import (
    RefError,
    Report,
    child,
    cyclic_refs,
    merge_all,
    resolve_ref,
    resolve_schema,
    true_to_empty,
)


class OpenAITransformer:
    """Non-strict function calling: only the root must be an object."""

    def __init__(self, schema: Any, report: Report) -> None:
        self.schema = copy.deepcopy(schema)
        self.report = report

    def run(self) -> Any:
        return fix_root_object(self.schema, self.report, "openai")


class OpenAIStrictTransformer:
    def __init__(self, schema: Any, report: Report) -> None:
        self.report = report
        self.original = true_to_empty(
            copy.deepcopy(schema), "", report, "openai_strict.boolean_schema"
        )
        self.cyclic = cyclic_refs(self.original)

    def run(self) -> Any:
        r = self.report
        root = copy.deepcopy(self.original)
        if isinstance(root, dict) and any(
            isinstance(root.get(k), list) for k in ("anyOf", "oneOf")
        ):
            # openai-agents: "The root of a strict JSON schema must not use `anyOf`" (oneOf
            # is converted to anyOf first).
            r.problem(
                "openai_strict.root_anyof",
                "",
                "the root of a strict schema must be a plain object and must not use anyOf/oneOf",
            )
            return root
        root = fix_root_object(root, r, "openai_strict")
        if not isinstance(root, dict):
            return root
        out = self.visit(root, "", ())
        # Inlining a root $ref or merging a root allOf can produce a root anyOf.
        if isinstance(out, dict) and isinstance(out.get("anyOf"), list):
            r.problem(
                "openai_strict.root_anyof",
                "",
                "the root of a strict schema must be a plain object and must not use anyOf/oneOf",
            )
        return out

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

    def _check_ref(self, ref: Any, path: str) -> bool:
        if not isinstance(ref, str):
            self.report.problem("schema.invalid", child(path, "$ref"), "$ref must be a string")
            return False
        try:
            resolve_ref(self.original, ref)
        except RefError as exc:
            self.report.problem("schema.unresolvable_ref", child(path, "$ref"), str(exc))
            return False
        return True

    def visit(self, node: Any, path: str, stack: tuple[str, ...]) -> Any:
        r = self.report
        if node is True:
            r.fix("openai_strict.boolean_schema", path, "boolean schema `true` replaced with `{}`")
            return {}
        if not isinstance(node, dict):
            r.problem("schema.invalid", path, "schema must be a JSON object")
            return node
        if not r.spend(path):
            return node
        node = dict(node)

        if "$ref" in node:
            ref = node["$ref"]
            if not self._check_ref(ref, path):
                return node
            if len(node) == 1:
                return node  # a plain reference; its target is processed under $defs
            if ref in stack:
                r.problem(
                    "openai_strict.ref_siblings",
                    path,
                    f"$ref cannot have sibling keywords in strict mode, and recursive {ref} "
                    "cannot be inlined",
                )
                return node
            siblings = {k: v for k, v in node.items() if k != "$ref"}
            target = resolve_schema(self.original, ref)
            merged = (
                merge_all(siblings, copy.deepcopy(target)) if isinstance(target, dict) else None
            )
            if merged is None:
                r.problem(
                    "openai_strict.ref_siblings",
                    path,
                    f"$ref cannot have sibling keywords in strict mode, and {ref} conflicts with "
                    "them",
                )
                return node
            r.fix(
                "openai_strict.ref_siblings",
                path,
                f"$ref cannot have sibling keywords in strict mode; inlined {ref}",
            )
            return self.visit(merged, path, (*stack, ref))

        if "oneOf" in node:
            if "anyOf" in node:
                r.problem(
                    "openai_strict.oneof",
                    child(path, "oneOf"),
                    "oneOf is not supported and cannot become anyOf because anyOf is already set",
                )
            else:
                node["anyOf"] = node.pop("oneOf")
                r.fix(
                    "openai_strict.oneof",
                    child(path, "oneOf"),
                    "oneOf is not supported in strict mode; replaced by anyOf (no longer "
                    "exclusive)",
                    lossy=True,
                )

        if "allOf" in node and isinstance(node["allOf"], list):
            entries = node.pop("allOf")
            merged_all: dict[str, Any] | None = node
            for entry in entries:
                # `true` accepts everything: neutral in an allOf.
                resolved = {} if entry is True else entry
                if isinstance(entry, dict) and "$ref" in entry and isinstance(entry["$ref"], str):
                    try:
                        target = resolve_schema(self.original, entry["$ref"])
                    except RefError:
                        target = None
                    sib = {k: v for k, v in entry.items() if k != "$ref"}
                    resolved = (
                        merge_all(sib, copy.deepcopy(target))
                        if isinstance(target, dict)
                        and entry["$ref"] not in stack
                        and entry["$ref"] not in self.cyclic
                        else None
                    )
                merged_all = (
                    merge_all(merged_all, resolved)
                    if merged_all is not None and isinstance(resolved, dict)
                    else None
                )
            if merged_all is None:
                # Report what else is wrong inside, but leave the entries as written: the
                # node is unrepairable, and rewriting its entries would hide the conflict.
                node["allOf"] = entries
                self._probe(entries, path, stack)
                r.problem(
                    "openai_strict.allof",
                    child(path, "allOf"),
                    "allOf schemas conflict and cannot be merged into one schema; OpenAI's own "
                    "strict conversion rejects this",
                )
            else:
                r.fix("openai_strict.allof", child(path, "allOf"), "allOf merged into one schema")
                return self.visit(merged_all, path, stack)

        if "default" in node and node["default"] is None:
            node.pop("default")
            r.fix("openai_strict.null_default", child(path, "default"), "default: null removed")

        type_value = node.get("type")
        if type_value is None and isinstance(node.get("properties"), dict):
            node["type"] = type_value = "object"
            r.fix("openai_strict.object_type", child(path, "type"), 'added type: "object"')
        is_object = type_value == "object" or (
            isinstance(type_value, list) and "object" in type_value
        )

        props = node.get("properties")
        if isinstance(props, dict):
            node["properties"] = props = {
                name: self.visit(sub, child(path, "properties", name), stack)
                for name, sub in props.items()
            }
        if isinstance(node.get("items"), (dict, bool)):
            node["items"] = self.visit(node["items"], child(path, "items"), stack)
        for key in ("anyOf", "prefixItems"):
            if isinstance(node.get(key), list):
                node[key] = [
                    self.visit(sub, child(path, key, i), stack) for i, sub in enumerate(node[key])
                ]
        for key in ("$defs", "definitions"):
            defs = node.get(key)
            if isinstance(defs, dict):
                node[key] = {
                    name: self.visit(sub, child(path, key, name), stack)
                    for name, sub in defs.items()
                }

        if is_object:
            if "additionalProperties" not in node:
                node["additionalProperties"] = False
                r.fix(
                    "openai_strict.additional_properties",
                    path,
                    "strict mode requires additionalProperties: false on every object; added",
                )
            elif node["additionalProperties"] is not False:
                r.problem(
                    "openai_strict.additional_properties",
                    child(path, "additionalProperties"),
                    "strict mode requires additionalProperties: false; objects with arbitrary "
                    "keys (maps) cannot be expressed",
                )
            if isinstance(props, dict):
                required = node.get("required")
                required = list(required) if isinstance(required, list) else []
                optional = [name for name in props if name not in required]
                for name in optional:
                    already_nullable = allows_null(props[name])
                    props[name] = make_nullable(props[name])
                    r.fix(
                        "openai_strict.optional_property",
                        child(path, "properties", name),
                        "strict mode requires every property in required; made it required"
                        + (
                            " (it already accepts null)"
                            if already_nullable
                            else " and nullable (restore_arguments() drops the nulls)"
                        ),
                        lossy=not already_nullable,
                    )
                if optional:
                    # Property order, as OpenAI's own SDKs emit it.
                    node["required"] = list(props) + [
                        n for n in required if not (isinstance(n, str) and n in props)
                    ]
                elif not isinstance(node.get("required"), list):
                    node["required"] = []
                    r.fix(
                        "openai_strict.required",
                        child(path, "required"),
                        "strict mode requires a required array; added []",
                    )
        return node
