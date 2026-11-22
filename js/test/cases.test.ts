/** The shared, hand-written case corpus (tests/cases/*.json) and real-world counts. */

import { readFileSync, readdirSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { type Json, transformSchema, transformTools } from "../src/index.js";

const casesDir = new URL("../../tests/cases/", import.meta.url);
const cases = readdirSync(casesDir)
  .filter((f) => f.endsWith(".json"))
  .flatMap((file) =>
    (JSON.parse(readFileSync(new URL(file, casesDir), "utf8")) as Record<string, Json>[]).map(
      (c) => [`${file}: ${c.name as string}`, c] as const,
    ),
  );

const keys = (items: { code: string; path: string }[]) => new Set(items.map((i) => `${i.code}@${i.path}`));

describe("shared case corpus", () => {
  it.each(cases)("%s", (_name, c) => {
    const before = JSON.stringify(c.input);
    const result = transformSchema(c.input as Json, c.target as string);
    expect(JSON.stringify(c.input)).toBe(before);
    expect(result.schema).toEqual(c.output);
    expect(keys(result.changes)).toEqual(new Set(c.changes as string[]));
    expect(keys(result.changes.filter((x) => x.lossy))).toEqual(new Set(c.lossy as string[]));
    expect(keys(result.issues)).toEqual(new Set(c.issues as string[]));
    expect(result.ok).toBe((c.ok as boolean | undefined) ?? (c.issues as string[]).length === 0);
  });
});

describe("GitHub MCP server (149 real tools)", () => {
  const tools = (
    JSON.parse(
      readFileSync(new URL("../../tests/realworld/github_mcp_server.json", import.meta.url), "utf8"),
    ) as { tools: Json[] }
  ).tools;
  const expected: Record<string, [number, number]> = {
    anthropic: [0, 0],
    "anthropic-strict": [143, 0],
    openai: [0, 0],
    "openai-strict": [143, 0],
    gemini: [70, 6],
  };
  it.each(Object.entries(expected))("%s", (target, [affected, failing]) => {
    const result = transformTools(tools, target);
    const perTool = new Set(result.issues.filter((i) => i.severity === "error" && i.tool).map((i) => i.tool));
    const touched = new Set([
      ...result.changes.filter((c) => c.severity === "error").map((c) => c.tool),
      ...perTool,
    ]);
    expect([touched.size, perTool.size]).toEqual([affected, failing]);
  });
});
