"""Command-line interface: ``schema-compat check`` and ``schema-compat fix``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TextIO

from . import __version__
from ._api import check_schema, transform_schema
from ._model import TARGETS, Issue
from ._tools import transform_tools

EXIT_OK, EXIT_PROBLEMS, EXIT_USAGE = 0, 1, 2


class InputError(Exception):
    pass


def _load(path: str) -> Any:
    try:
        text = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise InputError(f"{path}: cannot read: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise InputError(f"{path}: invalid JSON: {exc.msg} (line {exc.lineno})") from exc


def _kind(data: Any) -> str:
    """'wrapped' ({"tools": [...]}), 'list', 'tool' or 'schema'."""
    if isinstance(data, dict) and isinstance(data.get("tools"), list):
        return "wrapped"
    if isinstance(data, list):
        return "list"
    if isinstance(data, dict) and (
        isinstance(data.get("name"), str)
        or (data.get("type") == "function" and isinstance(data.get("function"), dict))
    ):
        return "tool"
    return "schema"


def _issues_for(data: Any, target: str) -> list[Issue]:
    kind = _kind(data)
    if kind == "schema":
        return check_schema(data, target)
    tools = data["tools"] if kind == "wrapped" else data if kind == "list" else [data]
    result = transform_tools(tools, target)
    return [*result.changes, *result.issues]


def _gh_escape(text: str, *, prop: bool = False) -> str:
    text = text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    if prop:
        text = text.replace(":", "%3A").replace(",", "%2C")
    return text


def _describe(issue: Issue) -> str:
    where = f"tool {issue.tool!r} " if issue.tool else ""
    flags = [f for f, on in (("fixable", issue.fixable), ("lossy", issue.lossy)) if on]
    suffix = f" ({', '.join(flags)})" if flags else ""
    return f"{where}{issue.path or '/'}: {issue.code}{suffix}: {issue.message}"


def _check(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    targets = args.target or list(TARGETS)
    records: list[tuple[str, str, Issue]] = []
    for path in args.files:
        data = _load(path)
        for target in targets:
            records.extend((path, target, issue) for issue in _issues_for(data, target))

    if args.format == "json":
        payload = [{"file": f, "target": t, **i.to_dict()} for f, t, i in records]
        out.write(json.dumps(payload, indent=2) + "\n")
    elif args.format == "github":
        for file, target, issue in records:
            level = "error" if issue.severity == "error" else "warning"
            title = _gh_escape(f"schema-compat {issue.code}", prop=True)
            message = _gh_escape(f"[{target}] {_describe(issue)}")
            out.write(f"::{level} file={_gh_escape(file, prop=True)},title={title}::{message}\n")
    else:
        for file, target, issue in records:
            out.write(f"{file}: [{target}] {issue.severity}: {_describe(issue)}\n")

    errors = sum(1 for _, _, i in records if i.severity == "error")
    warnings = len(records) - errors
    fixable = sum(1 for _, _, i in records if i.severity == "error" and i.fixable)
    err.write(
        f"schema-compat: {errors} error(s) ({fixable} fixable with `schema-compat fix`), "
        f"{warnings} warning(s) for {', '.join(targets)}\n"
    )
    failing = errors + (warnings if args.fail_on == "warning" else 0)
    return EXIT_PROBLEMS if failing else EXIT_OK


def _fix(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    data = _load(args.file)
    kind = _kind(data)
    name_map: dict[str, str] = {}
    if kind == "schema":
        result = transform_schema(data, args.target)
        fixed: Any = result.schema
        changes, remaining = result.changes, result.issues
    else:
        tools = data["tools"] if kind == "wrapped" else data if kind == "list" else [data]
        tools_result = transform_tools(tools, args.target)
        changes, remaining = tools_result.changes, tools_result.issues
        name_map = tools_result.name_map
        if kind == "wrapped":
            fixed = {**data, "tools": tools_result.tools}
        elif kind == "list":
            fixed = tools_result.tools
        else:
            fixed = tools_result.tools[0]

    text = json.dumps(fixed, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        out.write(text)
    if args.name_map:
        Path(args.name_map).write_text(json.dumps(name_map, indent=2) + "\n", encoding="utf-8")

    lossy = sum(1 for c in changes if c.lossy)
    err.write(
        f"schema-compat: applied {len(changes)} change(s) for {args.target} ({lossy} lossy)\n"
    )
    if name_map:
        err.write(f"schema-compat: renamed {len(name_map)} tool(s)\n")
    errors = [i for i in remaining if i.severity == "error"]
    for issue in remaining:
        err.write(f"schema-compat: {issue.severity}: {_describe(issue)}\n")
    return EXIT_PROBLEMS if errors else EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="schema-compat",
        description="Check and repair tool/function JSON schemas for Anthropic, OpenAI and Gemini.",
    )
    parser.add_argument("--version", action="version", version=f"schema-compat {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser(
        "check",
        help="report problems (exit 1 if any error)",
        description='Input: a JSON schema, a tool, a list of tools, or {"tools": [...]} '
        "(MCP tools/list). Use - for stdin.",
    )
    check.add_argument("files", nargs="+", metavar="FILE")
    check.add_argument(
        "-t",
        "--target",
        action="append",
        choices=TARGETS,
        help="target to check (repeatable; default: all)",
    )
    check.add_argument("--format", choices=("text", "json", "github"), default="text")
    check.add_argument(
        "--fail-on",
        choices=("error", "warning"),
        default="error",
        help="lowest severity that makes the exit code 1 (default: error)",
    )

    fix = sub.add_parser("fix", help="write a repaired copy (exit 1 if errors remain)")
    fix.add_argument("file", metavar="FILE")
    fix.add_argument("-t", "--target", required=True, choices=TARGETS)
    fix.add_argument("-o", "--output", help="write here instead of stdout")
    fix.add_argument("--name-map", help="write {new_name: original_name} for renamed tools here")
    return parser


def main(
    argv: list[str] | None = None, out: TextIO | None = None, err: TextIO | None = None
) -> int:
    out = out or sys.stdout
    err = err or sys.stderr
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            return _check(args, out, err)
        return _fix(args, out, err)
    except InputError as exc:
        err.write(f"schema-compat: error: {exc}\n")
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
