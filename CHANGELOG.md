# Changelog

## [0.1.0] – 2026-11-22

First release.

- `check_schema` / `transform_schema` for five targets: `anthropic`, `anthropic-strict`,
  `openai`, `openai-strict`, `gemini`.
- `check_tools` / `transform_tools` for MCP, Anthropic, OpenAI (Chat and Responses) and Gemini
  tool shapes: name rules with a name map back, duplicate names, missing schemas, Anthropic's
  per-request strict limits, Gemini tools without arguments.
- `restore_arguments`: undo stringified Gemini enum/const values and OpenAI-strict nulls.
- `schema-compat check` / `schema-compat fix` CLI (text, JSON and GitHub annotation output) and
  a pre-commit hook.
- Verified against a hand-written case corpus, the providers' own SDK code (`google-genai`
  `types.Schema`, `openai-agents` strict conversion) and randomised schemas.
