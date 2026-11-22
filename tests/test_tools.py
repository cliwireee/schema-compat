from __future__ import annotations

import copy

import pytest

from schema_compat import (
    TARGETS,
    check_tools,
    name_is_valid,
    sanitize_name,
    transform_tools,
)


def codes(issues: object) -> set[str]:
    return {f"{i.code}@{i.path}" for i in issues}  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("name", "target", "expected"),
    [
        ("github.search", "openai", "github_search"),
        ("github.search", "gemini", "github.search"),
        ("github/search", "anthropic", "github_search"),
        ("1st-tool", "gemini", "_1st-tool"),
        ("1st-tool", "openai", "1st-tool"),
        ("", "openai", "tool"),
    ],
)
def test_sanitize_name(name: str, target: str, expected: str) -> None:
    assert sanitize_name(name, target) == expected
    assert name_is_valid(sanitize_name(name, target), target)


def test_long_names_are_truncated_with_a_stable_hash() -> None:
    name = "mcp__server__" + "x" * 100
    new = sanitize_name(name, "openai")
    assert len(new) == 64 and name_is_valid(new, "openai")
    assert new == sanitize_name(name, "openai")
    assert new != sanitize_name(name + "y", "openai")
    assert sanitize_name(name, "anthropic") == name  # 113 chars is fine for Anthropic


def test_shapes_are_preserved_and_names_mapped() -> None:
    tools = [
        {"name": "fs.read", "description": "Read", "inputSchema": {"properties": {}}},
        {
            "type": "function",
            "function": {"name": "fs/read", "parameters": {"type": "object", "properties": {}}},
        },
        {
            "name": "ok_tool",
            "input_schema": {"type": "object", "properties": {"a": {"type": "string"}}},
        },
    ]
    before = copy.deepcopy(tools)
    result = transform_tools(tools, "openai")
    assert tools == before
    assert result.tools[0] == {
        "name": "fs_read",
        "description": "Read",
        "inputSchema": {"type": "object", "properties": {}},
    }
    # "fs/read" also sanitizes to fs_read: made unique.
    assert result.tools[1]["function"]["name"] == "fs_read_2"
    assert result.tools[1]["type"] == "function"
    assert result.tools[2] == tools[2]
    assert result.name_map == {"fs_read": "fs.read", "fs_read_2": "fs/read"}
    assert result.ok


def test_gemini_omits_parameters_for_tools_without_arguments() -> None:
    tools = [
        {"name": "ping", "parameters": {"type": "object", "properties": {}}},
        {"name": "echo", "parameters": {"type": "object", "properties": {"t": {"type": "string"}}}},
    ]
    result = transform_tools(tools, "gemini")
    assert result.tools[0] == {"name": "ping"}
    assert result.tools[1] == tools[1]
    assert "gemini.empty_parameters@" in codes(result.changes)
    assert result.ok


def test_anthropic_requires_input_schema() -> None:
    result = transform_tools([{"name": "ping"}], "anthropic")
    assert result.tools[0]["input_schema"] == {"type": "object", "properties": {}}
    assert result.ok
    assert transform_tools([{"name": "ping"}], "openai").tools == [{"name": "ping"}]


def test_duplicate_and_missing_names() -> None:
    issues = check_tools(
        [{"name": "a", "parameters": {"type": "object"}}, {"name": "a"}, {"description": "x"}, 5],
        "openai",
    )
    assert {i.code for i in issues} >= {"tool.duplicate_name", "tool.missing_name", "tool.invalid"}


def test_schema_issues_carry_the_tool_name() -> None:
    issues = check_tools(
        [{"name": "t", "parameters": {"type": "object", "properties": {"x": {"type": "array"}}}}],
        "gemini",
    )
    assert [(i.tool, i.code, i.path) for i in issues] == [
        ("t", "gemini.array_without_items", "/properties/x")
    ]


def test_anthropic_strict_request_limits() -> None:
    def tool(i: int, optional: int, unions: int = 0) -> dict[str, object]:
        props: dict[str, object] = {f"o{j}": {"type": "string"} for j in range(optional)}
        props.update({f"u{j}": {"type": ["string", "integer"]} for j in range(unions)})
        required = [f"u{j}" for j in range(unions)]
        return {
            "name": f"t{i}",
            "input_schema": {"type": "object", "properties": props, "required": required},
        }

    ok = transform_tools([tool(i, 1) for i in range(20)], "anthropic-strict")
    assert ok.ok
    # 21 tools with 1 optional parameter each: over the 20-tool limit, under 24 optional.
    many = transform_tools([tool(i, 1) for i in range(21)], "anthropic-strict")
    assert {i.code for i in many.issues} == {"anthropic_strict.too_many_strict_tools"}
    optional = transform_tools([tool(0, 25)], "anthropic-strict")
    assert {i.code for i in optional.issues} == {"anthropic_strict.too_many_optional"}
    unions = transform_tools([tool(0, 0, unions=17)], "anthropic-strict")
    assert {i.code for i in unions.issues} == {"anthropic_strict.too_many_unions"}


def test_gemini_json_schema_mode_is_not_checked() -> None:
    issues = check_tools([{"name": "t", "parametersJsonSchema": {"type": "object"}}], "gemini")
    assert [(i.code, i.severity) for i in issues] == [("gemini.parameters_json_schema", "warning")]


@pytest.mark.parametrize("target", TARGETS)
def test_unknown_target_rejected(target: str) -> None:
    with pytest.raises(ValueError):
        transform_tools([], target + "x")


def test_null_schema_is_treated_as_missing() -> None:
    tool = {"name": "t", "inputSchema": None}
    anthropic = transform_tools([tool], "anthropic")
    assert anthropic.tools == [{"name": "t", "input_schema": {"type": "object", "properties": {}}}]
    assert anthropic.ok
    gemini = transform_tools([tool], "gemini")
    assert gemini.tools == [{"name": "t"}] and gemini.ok
    assert "gemini.null_schema@" in codes(gemini.changes)


def test_trailing_newline_is_not_a_valid_name() -> None:
    assert not name_is_valid("search\n", "openai")
    assert sanitize_name("search\n", "openai") == "search_"
