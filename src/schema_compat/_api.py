"""Public entry points."""

from __future__ import annotations

from typing import Any

from ._anthropic import AnthropicStrictTransformer, AnthropicTransformer
from ._gemini import GeminiTransformer
from ._model import TARGETS, Issue, SchemaResult, ToolsResult
from ._openai import OpenAIStrictTransformer, OpenAITransformer
from ._util import Report

_TRANSFORMERS: dict[str, Any] = {
    "anthropic": AnthropicTransformer,
    "anthropic-strict": AnthropicStrictTransformer,
    "openai": OpenAITransformer,
    "openai-strict": OpenAIStrictTransformer,
    "gemini": GeminiTransformer,
}


def _validate_target(target: str) -> None:
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; expected one of {', '.join(TARGETS)}")


def transform_schema(schema: Any, target: str, *, tool: str | None = None) -> SchemaResult:
    """Repair a tool's parameter schema so ``target`` accepts it.

    The input is never mutated. ``result.schema`` is the repaired schema,
    ``result.changes`` what was changed, ``result.issues`` what could not be
    repaired (``result.ok`` is False if any of them is an error).
    """
    _validate_target(target)
    report = Report(tool=tool)
    schema_out = _TRANSFORMERS[target](schema, report).run()
    return SchemaResult(schema=schema_out, changes=report.changes, issues=report.issues)


def check_schema(schema: Any, target: str) -> list[Issue]:
    """Every problem ``target`` has with ``schema``; ``fixable`` ones are repaired by transform."""
    result = transform_schema(schema, target)
    return [*result.changes, *result.issues]


def check_tools(tools: list[Any], target: str) -> list[Issue]:
    """Every problem ``target`` has with these tool definitions (names, schemas, limits)."""
    from ._tools import transform_tools

    result: ToolsResult = transform_tools(tools, target)
    return [*result.changes, *result.issues]
