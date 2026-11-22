"""Generate the Python/TypeScript parity corpus: tests/parity/corpus.json.

The TypeScript package must reproduce every output in it exactly (js/test/parity.test.ts).
Regenerate after any behaviour change:  python tests/parity/generate.py
"""

from __future__ import annotations

import gzip
import json
import random
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "src"))

from test_fuzz import _mutate, gen_root  # noqa: E402

from schema_compat import (  # noqa: E402
    TARGETS,
    Issue,
    restore_arguments,
    sanitize_name,
    transform_schema,
    transform_tools,
)

COUNT = 300
ARGS: list[Any] = [
    {"x": "1", "p0": None, "a": [1, {"b": None}]},
    {"p0": "2", "p1": None, "x": {"p0": "true", "p1": None}, "p2": ["1", None]},
]
NAMES = [
    "search",
    "fs.read_file",
    "github/search_issues",
    "1st-tool",
    "_private",
    "search\n",
    "",
    "über_tool",
    "tool with spaces",
    "emoji_\U0001f600_tool",
    "mcp__server__" + "x" * 120,
    "a:b.c-d",
    ".hidden",
]


def issues(items: list[Issue]) -> list[dict[str, Any]]:
    return [i.to_dict() for i in items]


def schema_cases() -> list[dict[str, Any]]:
    cases = []
    for seed in range(COUNT):
        schemas = [gen_root(random.Random(seed))]
        rng = random.Random(seed + 10**6)
        schemas.append(_mutate(gen_root(rng), rng))
        for schema in schemas:
            for target in TARGETS:
                result = transform_schema(schema, target)
                cases.append(
                    {
                        "input": schema,
                        "target": target,
                        "schema": result.schema,
                        "changes": issues(result.changes),
                        "issues": issues(result.issues),
                        "ok": result.ok,
                        "restore": [
                            {"args": args, "out": restore_arguments(args, schema, target)}
                            for args in ARGS
                        ],
                    }
                )
    return cases


def tool_cases() -> list[dict[str, Any]]:
    realworld = json.loads((HERE.parent / "realworld" / "github_mcp_server.json").read_text())
    shapes = [
        realworld["tools"],
        [
            {"name": "fs.read", "inputSchema": None},
            {"type": "function", "function": {"name": "fs/read", "parameters": {"type": "object"}}},
            {"name": "ping", "parameters": {"type": "object", "properties": {}}},
            {"name": "ping"},
            {"description": "no name"},
            7,
            {"name": "j", "parametersJsonSchema": {"type": "object"}},
        ],
    ]
    cases = []
    for tools in shapes:
        for target in TARGETS:
            result = transform_tools(tools, target)
            cases.append(
                {
                    "input": tools,
                    "target": target,
                    "tools": result.tools,
                    "changes": issues(result.changes),
                    "issues": issues(result.issues),
                    "name_map": result.name_map,
                    "ok": result.ok,
                }
            )
    return cases


def name_cases() -> list[dict[str, Any]]:
    return [
        {"name": name, "target": target, "out": sanitize_name(name, target)}
        for name in NAMES
        for target in TARGETS
    ]


if __name__ == "__main__":
    corpus = {"schemas": schema_cases(), "tools": tool_cases(), "names": name_cases()}
    out = HERE / "corpus.json.gz"
    data = (json.dumps(corpus, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    out.write_bytes(gzip.compress(data, mtime=0))
    print(
        f"{out}: {len(corpus['schemas'])} schema cases, {len(corpus['tools'])} tool cases, "
        f"{len(corpus['names'])} name cases, {out.stat().st_size // 1024} KiB"
    )
