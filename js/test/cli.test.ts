import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { main } from "../src/cli.js";

function run(argv: string[]): { code: number; out: string; err: string } {
  let out = "";
  let err = "";
  const code = main(argv, { out: (t) => void (out += t), err: (t) => void (err += t) });
  return { code, out, err };
}

const dir = mkdtempSync(join(tmpdir(), "schema-compat-"));
const write = (name: string, data: unknown) => {
  const path = join(dir, name);
  writeFileSync(path, JSON.stringify(data));
  return path;
};
const TOOLS = {
  tools: [
    {
      name: "search.issues",
      inputSchema: {
        type: "object",
        properties: { q: { type: "string" }, limit: { type: "integer", exclusiveMinimum: 0 } },
        required: ["q"],
        additionalProperties: false,
        $schema: "http://json-schema.org/draft-07/schema#",
      },
    },
    { name: "ping", inputSchema: { type: "object", properties: {} } },
  ],
};

describe("cli", () => {
  it("check reports problems with exit code 1", () => {
    const { code, out, err } = run(["check", write("t.json", TOOLS), "-t", "gemini"]);
    expect(code).toBe(1);
    expect(out).toContain("[gemini] error: tool 'search.issues' /properties/limit/exclusiveMinimum");
    expect(err).toContain("fixable with `schema-compat fix`");
  });

  it("json and github formats", () => {
    const path = write("t2.json", TOOLS);
    const json = run(["check", path, "-t", "openai", "--format", "json"]);
    const records = JSON.parse(json.out) as { code: string; target: string }[];
    expect(records.map((r) => r.code)).toEqual(["openai.tool_name"]);
    const gh = run(["check", path, "-t", "openai", "--format", "github"]);
    expect(gh.out.startsWith("::error file=")).toBe(true);
    expect(gh.out).toContain("title=schema-compat openai.tool_name::");
  });

  it("fix writes output and name map, and the result passes", () => {
    const out = join(dir, "fixed.json");
    const names = join(dir, "names.json");
    const result = run(["fix", write("t3.json", TOOLS), "-t", "openai-strict", "-o", out, "--name-map", names]);
    expect(result.code).toBe(0);
    expect(JSON.parse(readFileSync(names, "utf8"))).toEqual({ search_issues: "search.issues" });
    expect(run(["check", out, "-t", "openai-strict"]).code).toBe(0);
  });

  it("usage errors and bad input exit 2", () => {
    expect(run(["check"]).code).toBe(2);
    expect(run(["fix", "x.json"]).code).toBe(2);
    expect(run(["check", "x.json", "-t", "nope"]).code).toBe(2);
    const bad = join(dir, "bad.json");
    writeFileSync(bad, "{nope");
    const result = run(["check", bad]);
    expect(result.code).toBe(2);
    expect(result.err).toContain("invalid JSON");
    expect(run(["--version"]).out).toMatch(/^schema-compat \d/);
  });
});
