"""Result types shared by every target."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Target = Literal["anthropic", "anthropic-strict", "openai", "openai-strict", "gemini"]
TARGETS: tuple[Target, ...] = (
    "anthropic",
    "anthropic-strict",
    "openai",
    "openai-strict",
    "gemini",
)

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class Issue:
    """A problem found in a schema or tool for one target.

    ``path`` is a JSON Pointer into the schema (``""`` is the root). For
    tool-level problems such as an invalid name, ``path`` is ``""`` and
    ``tool`` names the tool.

    ``fixable`` means :func:`transform_schema` / :func:`transform_tools` repairs
    it. ``lossy`` means the repaired schema accepts some value the original
    rejects, so the model may produce arguments your code did not expect;
    :func:`restore_arguments` undoes the reversible cases (stringified enum
    values, optional properties made required-and-nullable).
    """

    code: str
    path: str
    message: str
    severity: Severity = "error"
    fixable: bool = False
    lossy: bool = False
    tool: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "path": self.path,
            "message": self.message,
            "severity": self.severity,
            "fixable": self.fixable,
            "lossy": self.lossy,
            "tool": self.tool,
        }


@dataclass
class SchemaResult:
    """Output of :func:`transform_schema`.

    ``schema`` is the repaired schema. ``changes`` lists every repair that was
    applied (as fixable issues). ``issues`` lists what could not be repaired,
    plus warnings; if it contains no ``error`` the schema is expected to be
    accepted by the target.
    """

    schema: Any
    changes: list[Issue] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)


@dataclass
class ToolsResult:
    """Output of :func:`transform_tools`.

    ``tools`` keeps each tool's original shape (MCP, Anthropic, OpenAI or
    Gemini). ``name_map`` maps every renamed tool's new name to its original
    name, so tool calls coming back from the model can be routed.
    """

    tools: list[Any]
    changes: list[Issue] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    name_map: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)
