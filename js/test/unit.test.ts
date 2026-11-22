import { describe, expect, it } from "vitest";
import { jsonText, pyEqual, pyRepr, truthy } from "../src/py.js";
import { sha1Hex } from "../src/sha1.js";
import {
  type Json,
  TARGETS,
  checkSchema,
  checkTools,
  nameIsValid,
  restoreArguments,
  sanitizeName,
  transformSchema,
  transformTools,
} from "../src/index.js";

describe("sha1", () => {
  it("matches the standard test vectors", () => {
    expect(sha1Hex("")).toBe("da39a3ee5e6b4b0d3255bfef95601890afd80709");
    expect(sha1Hex("abc")).toBe("a9993e364706816aba3e25717850c26c9cd0d89d");
    expect(sha1Hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")).toBe(
      "84983e441c3bd26ebaae4aa1f95129e5e54670f1",
    );
    expect(sha1Hex("a".repeat(1000))).toBe("291e9a6c66994949b57ba5e650361e98fc36b1ba");
    // UTF-8 input, value from Python hashlib.sha1("über".encode()).hexdigest()
    expect(sha1Hex("über")).toBe("fcd10c8088744aa99203e948e7c87633b7be5485");
  });
});

describe("Python semantics", () => {
  it("formats JSON like json.dumps(separators=(', ', ': '))", () => {
    expect(jsonText({ a: [1, true, null, "x"], b: {} })).toBe('{"a": [1, true, null, "x"], "b": {}}');
    expect(jsonText(1.5)).toBe("1.5");
    expect(jsonText(0.00001)).toBe("1e-05");
    expect(jsonText(1.5e-7)).toBe("1.5e-07");
    expect(jsonText(1e16)).toBe("1e+16");
    expect(jsonText(123456.789)).toBe("123456.789");
    expect(jsonText("é\n\u0001")).toBe('"é\\n\\u0001"');
  });
  it("compares like Python ==", () => {
    expect(pyEqual(true, 1)).toBe(true);
    expect(pyEqual(false, 0)).toBe(true);
    expect(pyEqual([1, { a: true }], [true, { a: 1 }])).toBe(true);
    expect(pyEqual(null, 0)).toBe(false);
    expect(pyEqual("1", 1)).toBe(false);
  });
  it("has Python truthiness and repr", () => {
    expect([null, false, 0, "", [], {}].map(truthy)).toEqual([false, false, false, false, false, false]);
    expect(pyRepr(["a", 1, null, true, { "it's": 'x"y' }])).toBe(`['a', 1, None, True, {"it's": 'x"y'}]`);
  });
});

describe("safety", () => {
  it("never pollutes Object.prototype from untrusted keys", () => {
    const schema = JSON.parse(
      '{"type":"object","properties":{"__proto__":{"type":"string","polluted":true},"x":{"type":["string","null"]}},"__proto__":{"polluted":true}}',
    ) as Json;
    for (const target of TARGETS) {
      transformSchema(schema, target);
      transformTools([{ name: "__proto__", inputSchema: schema }], target);
      restoreArguments(JSON.parse('{"__proto__":{"polluted":true},"x":null}') as Json, schema, target);
    }
    expect(({} as Record<string, unknown>).polluted).toBeUndefined();
    const result = transformSchema(schema, "gemini");
    expect(Object.keys((result.schema as { properties: object }).properties)).toContain("__proto__");
  });

  it("does not mutate inputs and rejects unknown targets", () => {
    expect(() => transformSchema({}, "claude")).toThrow(/unknown target/);
    expect(() => restoreArguments({}, {}, "x")).toThrow(/unknown target/);
    expect(() => nameIsValid("a", "x")).toThrow(/unknown target/);
  });
});

describe("api", () => {
  it("README examples", () => {
    const schema: Json = {
      type: "object",
      properties: {
        query: { type: "string", minLength: 1 },
        limit: { type: ["integer", "null"], exclusiveMinimum: 0 },
        mode: { enum: [1, 2, 3] },
      },
      required: ["query"],
      additionalProperties: false,
      $schema: "http://json-schema.org/draft-07/schema#",
    };
    const result = transformSchema(schema, "gemini");
    expect(result.ok).toBe(true);
    expect(result.schema).toEqual({
      type: "object",
      properties: {
        query: { type: "string", minLength: 1 },
        limit: { type: "integer", nullable: true, description: "{exclusiveMinimum: 0}" },
        mode: { enum: ["1", "2", "3"] },
      },
      required: ["query"],
    });
    expect(restoreArguments({ query: "x", mode: "2" }, schema, "gemini")).toEqual({ query: "x", mode: 2 });

    const tools: Json[] = [
      { name: "fs.read_file", inputSchema: { type: "object", properties: { path: { type: "string" } } } },
      { name: "ping", inputSchema: { type: "object", properties: {} } },
    ];
    const openai = transformTools(tools, "openai");
    expect(openai.nameMap).toEqual({ fs_read_file: "fs.read_file" });
    expect(transformTools(tools, "gemini").tools[1]).toEqual({ name: "ping" });
    expect(checkTools(tools, "openai").map((i) => i.code)).toEqual(["openai.tool_name"]);
    expect(checkSchema({ type: "string" }, "anthropic").map((i) => i.code)).toEqual(["anthropic.root_not_object"]);
    expect(sanitizeName("search\n", "openai")).toBe("search_");
  });
});

describe("version", () => {
  it("VERSION matches package.json and the Python package", async () => {
    const { readFileSync } = await import("node:fs");
    const { VERSION } = await import("../src/index.js");
    const pkg = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")) as { version: string };
    const py = readFileSync(new URL("../../src/schema_compat/__init__.py", import.meta.url), "utf8");
    expect(VERSION).toBe(pkg.version);
    expect(py).toContain(`__version__ = "${pkg.version}"`);
  });
});
