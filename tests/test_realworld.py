"""Real production tool schemas: GitHub's official MCP server (MIT licensed snapshot).

Pins how many tools each target rejects before and after repair, and runs the
same cross-checks as the corpus. A change in these numbers is a behaviour change
that needs a deliberate update.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from schema_compat import transform_schema, transform_tools

TOOLS: list[dict[str, Any]] = json.loads(
    (Path(__file__).parent / "realworld" / "github_mcp_server.json").read_text(encoding="utf-8")
)["tools"]

# target -> (tools needing a change or failing as-is, tools still failing after repair)
EXPECTED = {
    "anthropic": (0, 0),
    "anthropic-strict": (143, 0),
    "openai": (0, 0),
    "openai-strict": (143, 0),
    "gemini": (70, 6),
}


@pytest.mark.parametrize("target", sorted(EXPECTED))
def test_counts(target: str) -> None:
    before = copy.deepcopy(TOOLS)
    result = transform_tools(TOOLS, target)
    assert before == TOOLS
    per_tool_errors = {i.tool for i in result.issues if i.severity == "error" and i.tool}
    affected = {c.tool for c in result.changes if c.severity == "error"} | per_tool_errors
    assert (len(affected), len(per_tool_errors)) == EXPECTED[target]


def _sorted_required(node: Any) -> Any:
    if isinstance(node, dict):
        out = {k: _sorted_required(v) for k, v in node.items()}
        if isinstance(out.get("required"), list):
            out["required"] = sorted(out["required"])
        return out
    if isinstance(node, list):
        return [_sorted_required(v) for v in node]
    return node


@pytest.mark.parametrize("tool", TOOLS, ids=[t["name"] for t in TOOLS])
def test_crosschecks(tool: dict[str, Any]) -> None:
    for target in ("anthropic-strict", "openai-strict", "gemini"):
        first = transform_schema(tool["inputSchema"], target)
        if not first.ok:
            continue
        second = transform_schema(first.schema, target)
        assert second.changes == [] and second.schema == first.schema
    gemini = transform_schema(tool["inputSchema"], "gemini")
    if gemini.ok:
        types = pytest.importorskip("google.genai.types")
        types.Schema.model_validate(gemini.schema)
    strict = transform_schema(tool["inputSchema"], "openai-strict")
    if strict.ok:
        strict_schema = pytest.importorskip("agents.strict_schema")
        converted = strict_schema.ensure_strict_json_schema(copy.deepcopy(strict.schema))
        assert _sorted_required(converted) == _sorted_required(strict.schema)


def test_anthropic_strict_request_limit() -> None:
    # 149 strict tools in one request exceed the documented limit of 20.
    result = transform_tools(TOOLS, "anthropic-strict")
    request_level = {i.code for i in result.issues if i.tool is None}
    assert "anthropic_strict.too_many_strict_tools" in request_level
