"""Tool-level handling: tool shapes, names, request-level limits."""

from __future__ import annotations

import copy
import hashlib
import re
from typing import Any

from ._anthropic import (
    MAX_OPTIONAL_PARAMETERS,
    MAX_STRICT_TOOLS,
    MAX_UNION_PARAMETERS,
    complexity,
)
from ._model import TARGETS, Issue, ToolsResult
from ._util import Report

# Tool-name rules:
# - Anthropic: ^[a-zA-Z0-9_-]{1,128}$ (tool-use docs).
# - OpenAI: "a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64"
#   (openai-python FunctionDefinition).
# - Gemini: "Must start with a letter or an underscore. Must be a-z, A-Z, 0-9, or contain
#   underscores, dots, colons and dashes, with a maximum length of 128" (google-genai
#   FunctionDeclaration).
_NAME_RULES: dict[str, tuple[re.Pattern[str], str, int, bool]] = {
    "anthropic": (re.compile(r"^[a-zA-Z0-9_-]{1,128}$"), r"[^a-zA-Z0-9_-]", 128, False),
    "openai": (re.compile(r"^[a-zA-Z0-9_-]{1,64}$"), r"[^a-zA-Z0-9_-]", 64, False),
    "gemini": (re.compile(r"^[a-zA-Z_][a-zA-Z0-9_.:-]{0,127}$"), r"[^a-zA-Z0-9_.:-]", 128, True),
}
_SCHEMA_KEYS = ("inputSchema", "input_schema", "parameters", "parametersJsonSchema")


def _family(target: str) -> str:
    return target.split("-")[0]


def _code(target: str, rule: str) -> str:
    return target.replace("-", "_") + "." + rule


def name_is_valid(name: str, target: str) -> bool:
    pattern, _, _, _ = _NAME_RULES[_family(target)]
    return bool(pattern.match(name))


def sanitize_name(name: str, target: str) -> str:
    """Rewrite ``name`` so ``target`` accepts it (deterministic)."""
    pattern, bad_chars, max_len, letter_first = _NAME_RULES[_family(target)]
    if pattern.match(name):
        return name
    new = re.sub(bad_chars, "_", name) or "tool"
    if letter_first and not re.match(r"[A-Za-z_]", new):
        new = "_" + new
    if len(new) > max_len:
        digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
        new = new[: max_len - 9] + "_" + digest
    return new


def _unique(name: str, used: set[str], target: str) -> str:
    if name not in used:
        return name
    _, _, max_len, _ = _NAME_RULES[_family(target)]
    n = 2
    while True:
        suffix = f"_{n}"
        candidate = name[: max_len - len(suffix)] + suffix
        if candidate not in used:
            return candidate
        n += 1


def _container(tool: Any) -> dict[str, Any] | None:
    """The dict holding ``name`` and the schema (OpenAI Chat nests it under ``function``)."""
    if not isinstance(tool, dict):
        return None
    if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        inner: dict[str, Any] = tool["function"]
        return inner
    return tool


def transform_tools(tools: list[Any], target: str) -> ToolsResult:
    """Check and repair a list of tool definitions for ``target``.

    Accepts MCP (``inputSchema``), Anthropic (``input_schema``), OpenAI Chat
    (``{"type": "function", "function": {...}}``), OpenAI Responses and Gemini
    (``parameters``) tool shapes, and keeps each tool's shape. Invalid tool names
    are rewritten; ``name_map`` maps new names back to the originals.
    """
    from ._api import transform_schema

    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; expected one of {', '.join(TARGETS)}")
    result = ToolsResult(tools=[])
    seen_original: set[str] = set()
    used: set[str] = set()
    total_optional = 0
    total_unions = 0

    for index, tool in enumerate(tools):
        container = _container(tool)
        if container is None:
            result.issues.append(
                Issue("tool.invalid", "", "tool definition must be an object", tool=f"#{index}")
            )
            result.tools.append(tool)
            continue
        new_tool = copy.deepcopy(tool)
        new_container = _container(new_tool)
        assert new_container is not None
        name = container.get("name")
        label = name if isinstance(name, str) and name else f"#{index}"
        report = Report(tool=label)

        if not isinstance(name, str) or not name:
            report.problem("tool.missing_name", "", "tool has no name")
        elif name in seen_original:
            report.problem("tool.duplicate_name", "", f"tool name {name!r} is used more than once")
            used.add(name)
        else:
            seen_original.add(name)
            new_name = _unique(sanitize_name(name, target), used, target)
            if new_name != name:
                new_container["name"] = new_name
                result.name_map[new_name] = name
                pattern = _NAME_RULES[_family(target)][0].pattern
                report.fix(
                    _code(target, "tool_name"),
                    "",
                    f"tool name {name!r} must match {pattern}; renamed to {new_name!r}",
                )
            used.add(new_name)

        # A null schema field means "no schema"; providers reject null there.
        for key in _SCHEMA_KEYS:
            if key in container and container[key] is None:
                del new_container[key]
                report.fix(_code(target, "null_schema"), "", f"{key} is null; removed")
        schema_key = next(
            (k for k in _SCHEMA_KEYS if k in container and container[k] is not None), None
        )
        if schema_key == "parametersJsonSchema" and target == "gemini":
            report.problem(
                "gemini.parameters_json_schema",
                "",
                "parametersJsonSchema uses Gemini's JSON Schema mode, which this version does not "
                "check; use parameters to have it checked",
                severity="warning",
            )
        elif schema_key is not None:
            schema_result = transform_schema(container[schema_key], target, tool=label)
            report.changes.extend(schema_result.changes)
            remaining = schema_result.issues
            if target == "gemini" and any(
                i.code == "gemini.empty_object" and i.path == "" for i in remaining
            ):
                del new_container[schema_key]
                remaining = [
                    i for i in remaining if not (i.code == "gemini.empty_object" and i.path == "")
                ]
                report.fix(
                    "gemini.empty_parameters",
                    "",
                    "Gemini rejects an OBJECT without properties; omitted parameters (the tool "
                    "takes no arguments)",
                )
            else:
                new_container[schema_key] = schema_result.schema
            report.issues.extend(remaining)
            if target == "anthropic-strict" and schema_key in new_container:
                optional, unions = complexity(new_container[schema_key])
                total_optional += optional
                total_unions += unions
        elif _family(target) == "anthropic":
            new_container["input_schema"] = {"type": "object", "properties": {}}
            report.fix(
                _code(target, "missing_input_schema"),
                "",
                "Anthropic requires input_schema; added an empty object schema",
            )

        result.tools.append(new_tool)
        result.changes.extend(report.changes)
        result.issues.extend(report.issues)

    if target == "anthropic-strict":
        limits = (
            (len(tools), MAX_STRICT_TOOLS, "too_many_strict_tools", "strict tools"),
            (total_optional, MAX_OPTIONAL_PARAMETERS, "too_many_optional", "optional parameters"),
            (total_unions, MAX_UNION_PARAMETERS, "too_many_unions", "union-typed parameters"),
        )
        for count, limit, rule, label in limits:
            if count > limit:
                result.issues.append(
                    Issue(
                        f"anthropic_strict.{rule}",
                        "",
                        f"{count} {label} across the request; the limit is {limit}",
                    )
                )
    return result
