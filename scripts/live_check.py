"""Send repaired tool schemas to the real provider APIs and report which are rejected.

Not part of the test suite (it needs API keys and costs a fraction of a cent).

    OPENAI_API_KEY=... GEMINI_API_KEY=... python scripts/live_check.py tools.json

``tools.json`` is an MCP ``tools/list`` result (``{"tools": [...]}``). For each
target, tools are sent in batches; a rejected batch is split until the failing
tool is isolated. ``--control`` also sends the ORIGINAL schemas, to show what
the APIs reject without schema-compat.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from schema_compat import transform_tools  # noqa: E402

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
GEMINI_COMPAT_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
GEMINI_NATIVE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)


def post(url: str, body: dict[str, Any], headers: dict[str, str]) -> tuple[int, str]:
    """POST, retrying rate limits and overloads (429, 5xx) with exponential backoff."""
    status, error = _post_once(url, body, headers)
    for delay in (2, 4, 8, 16, 32, 64):
        if status != 429 and status < 500:
            break
        time.sleep(delay)
        status, error = _post_once(url, body, headers)
    return status, error


def _post_once(url: str, body: dict[str, Any], headers: dict[str, str]) -> tuple[int, str]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, ""
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def openai_style_request(model: str, tools: list[dict[str, Any]], strict: bool) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "max_completion_tokens": 16,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": (t.get("description") or "")[:1000],
                    "parameters": t.get("inputSchema", {"type": "object", "properties": {}}),
                    **({"strict": True} if strict else {}),
                },
            }
            for t in tools
        ],
    }


def gemini_native_request(tools: list[dict[str, Any]]) -> dict[str, Any]:
    declarations = []
    for t in tools:
        decl: dict[str, Any] = {
            "name": t["name"],
            "description": (t.get("description") or "")[:1000],
        }
        if "inputSchema" in t:
            decl["parameters"] = t["inputSchema"]
        declarations.append(decl)
    return {
        "contents": [{"role": "user", "parts": [{"text": "Reply with OK."}]}],
        "tools": [{"functionDeclarations": declarations}],
        "generationConfig": {"maxOutputTokens": 16},
    }


def find_rejected(tools: list[dict[str, Any]], send: Any, batch: int = 16) -> list[tuple[str, str]]:
    """Return (tool name, error) for every tool the API rejects."""
    rejected: list[tuple[str, str]] = []

    def check(group: list[dict[str, Any]]) -> None:
        status, error = send(group)
        if status == 200:
            return
        if status not in (400, 422):
            raise RuntimeError(f"unexpected HTTP {status}: {error[:500]}")
        if len(group) == 1:
            rejected.append((group[0]["name"], error))
            return
        mid = len(group) // 2
        check(group[:mid])
        check(group[mid:])

    for start in range(0, len(tools), batch):
        check(tools[start : start + batch])
    return rejected


def summarise(label: str, total: int, rejected: list[tuple[str, str]]) -> None:
    print(f"{label}: {total - len(rejected)}/{total} accepted")
    for name, error in rejected[:8]:
        try:
            message = json.loads(error)
            message = message[0] if isinstance(message, list) else message
            text = message.get("error", {}).get("message", error)
        except (ValueError, AttributeError):
            text = error
        print(f"   REJECTED {name}: {text[:220]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tools_file")
    parser.add_argument("--openai-model", default="gpt-4.1-mini")
    parser.add_argument("--gemini-model", default="gemini-2.5-flash")
    parser.add_argument("--control", action="store_true", help="also send the original schemas")
    parser.add_argument("--only", choices=("openai", "gemini"), action="append")
    args = parser.parse_args()

    data = json.loads(Path(args.tools_file).read_text(encoding="utf-8"))
    tools = data["tools"] if isinstance(data, dict) else data
    only = set(args.only or ("openai", "gemini"))

    if "openai" in only and os.environ.get("OPENAI_API_KEY"):
        headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
        for target, strict in (("openai", False), ("openai-strict", True)):
            fixed = transform_tools(tools, target).tools

            def send(group: list[dict[str, Any]], strict: bool = strict) -> tuple[int, str]:
                return post(
                    OPENAI_URL, openai_style_request(args.openai_model, group, strict), headers
                )

            summarise(f"[{target}] repaired", len(fixed), find_rejected(fixed, send))
            if args.control:
                renamed = transform_tools(tools, "openai").tools  # names only, schemas untouched
                originals = [
                    {**r, "inputSchema": t.get("inputSchema")}
                    for r, t in zip(renamed, tools, strict=True)
                ]
                summarise(
                    f"[{target}] original (control)", len(originals), find_rejected(originals, send)
                )

    if "gemini" in only and os.environ.get("GEMINI_API_KEY"):
        key = os.environ["GEMINI_API_KEY"]
        native_url = GEMINI_NATIVE_URL.format(model=args.gemini_model)
        native_headers = {"x-goog-api-key": key, "Authorization": f"Bearer {key}"}

        def send_native(group: list[dict[str, Any]]) -> tuple[int, str]:
            return post(native_url, gemini_native_request(group), native_headers)

        def send_compat(group: list[dict[str, Any]]) -> tuple[int, str]:
            body = openai_style_request(args.gemini_model, group, strict=False)
            return post(GEMINI_COMPAT_URL, body, {"Authorization": f"Bearer {key}"})

        fixed = transform_tools(tools, "gemini").tools
        # Tools whose parameters were omitted (no arguments) stay without parameters.
        originals = [
            {**f, "inputSchema": t["inputSchema"]} if "inputSchema" in f else f
            for f, t in zip(fixed, tools, strict=True)
        ]
        status, _ = send_native(fixed[:1])
        endpoints = [("native", send_native)] if status not in (401, 403) else []
        if not endpoints:
            print(
                f"[gemini] native endpoint refused the key (HTTP {status}); "
                "using the OpenAI-compatible endpoint"
            )
        endpoints.append(("openai-compatible", send_compat))
        for label, send in endpoints:
            summarise(f"[gemini/{label}] repaired", len(fixed), find_rejected(fixed, send))
            if args.control:
                summarise(
                    f"[gemini/{label}] original (control)",
                    len(originals),
                    find_rejected(originals, send),
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
