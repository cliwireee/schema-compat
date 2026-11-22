from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

from schema_compat.cli import main

TOOLS = {
    "tools": [
        {
            "name": "search.issues",
            "description": "Search",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "q": {"type": "string"},
                    "limit": {"type": "integer", "exclusiveMinimum": 0},
                },
                "required": ["q"],
                "additionalProperties": False,
                "$schema": "http://json-schema.org/draft-07/schema#",
            },
        },
        {"name": "ping", "inputSchema": {"type": "object", "properties": {}}},
    ]
}


def run(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, out=out, err=err)
    return code, out.getvalue(), err.getvalue()


def write(tmp_path: Path, data: object, name: str = "tools.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_check_text_reports_and_fails(tmp_path: Path) -> None:
    path = write(tmp_path, TOOLS)
    code, out, err = run(["check", str(path), "-t", "gemini"])
    assert code == 1
    assert "[gemini] error: tool 'search.issues' /properties/limit/exclusiveMinimum" in out
    assert "gemini.empty_parameters" in out  # ping: parameters would be omitted
    assert "fixable with `schema-compat fix`" in err


def test_check_anthropic_only_flags_the_dotted_name(tmp_path: Path) -> None:
    path = write(tmp_path, TOOLS)
    code, out, _ = run(["check", str(path), "-t", "anthropic"])
    assert code == 1
    assert out.count("\n") == 1 and "anthropic.tool_name" in out  # "." is not allowed

    clean = write(tmp_path, {"name": "ok", "input_schema": {"type": "object"}}, "ok.json")
    code, out, _ = run(["check", str(clean), "-t", "anthropic"])
    assert (code, out) == (0, "")


def test_check_json_and_github_formats(tmp_path: Path) -> None:
    path = write(tmp_path, TOOLS)
    code, out, _ = run(["check", str(path), "-t", "openai", "--format", "json"])
    records = json.loads(out)
    assert code == 1
    assert {r["code"] for r in records} == {"openai.tool_name"}
    assert records[0]["file"] == str(path) and records[0]["target"] == "openai"

    code, out, _ = run(["check", str(path), "-t", "openai", "--format", "github"])
    assert out.startswith("::error file=")
    assert "title=schema-compat openai.tool_name::" in out
    assert "\n" not in out.rstrip("\n")


def test_fail_on_warning(tmp_path: Path) -> None:
    path = write(tmp_path, {"name": "t", "parametersJsonSchema": {"type": "object"}}, "tool.json")
    assert run(["check", str(path), "-t", "gemini"])[0] == 0
    assert run(["check", str(path), "-t", "gemini", "--fail-on", "warning"])[0] == 1


def test_fix_writes_output_and_name_map(tmp_path: Path) -> None:
    path = write(tmp_path, TOOLS)
    out_path = tmp_path / "fixed.json"
    map_path = tmp_path / "names.json"
    code, out, err = run(
        ["fix", str(path), "-t", "gemini", "-o", str(out_path), "--name-map", str(map_path)]
    )
    assert code == 0, err
    assert out == ""
    fixed = json.loads(out_path.read_text(encoding="utf-8"))
    assert set(fixed) == {"tools"}
    assert fixed["tools"][0]["inputSchema"] == {
        "type": "object",
        "properties": {
            "q": {"type": "string"},
            "limit": {"type": "integer", "description": "{exclusiveMinimum: 0}"},
        },
        "required": ["q"],
    }
    assert fixed["tools"][1] == {"name": "ping"}
    assert json.loads(map_path.read_text(encoding="utf-8")) == {}

    code, _, _ = run(["check", str(out_path), "-t", "gemini"])
    assert code == 0


def test_fix_bare_schema_to_stdout_and_remaining_errors(tmp_path: Path) -> None:
    path = write(tmp_path, {"type": "object", "properties": {"m": {"type": "object"}}}, "s.json")
    code, out, err = run(["fix", str(path), "-t", "gemini"])
    assert code == 1  # the empty nested object cannot be repaired
    assert json.loads(out) == {"type": "object", "properties": {"m": {"type": "object"}}}
    assert "gemini.empty_object" in err


def test_bad_input_exit_2(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{nope", encoding="utf-8")
    code, _, err = run(["check", str(bad)])
    assert code == 2 and "invalid JSON" in err
    code, _, err = run(["check", str(tmp_path / "missing.json")])
    assert code == 2 and "cannot read" in err


def test_console_script_and_stdin(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "schema_compat.cli", "check", "-", "-t", "openai"],
        input=json.dumps({"type": "object", "properties": {}}),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    version = subprocess.run(
        [sys.executable, "-m", "schema_compat.cli", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert version.stdout.startswith("schema-compat ")
