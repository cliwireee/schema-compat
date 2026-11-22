/** JS-side randomised schemas (independent of the Python generator): nothing may throw. */

import { describe, expect, it } from "vitest";
import { type Json, TARGETS, restoreArguments, transformSchema, transformTools } from "../src/index.js";

function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const KEYS = ["type", "format", "enum", "const", "$ref", "items", "prefixItems", "properties", "required",
  "additionalProperties", "anyOf", "oneOf", "allOf", "not", "minimum", "minItems", "pattern", "nullable",
  "description", "$defs", "default", "examples", "title", "__proto__", "constructor"];
const JUNK: Json[] = [{}, [], [1], { a: 1 }, 1, 1.5, -3, true, false, null, "x", "", "#", "#/$defs/A",
  ["string", 1], { $ref: "#" }, [{}], "a\n", "object", ["object", "null"]];

function gen(r: () => number, depth: number): Json {
  const pick = <T>(xs: readonly T[]): T => xs[Math.floor(r() * xs.length)] as T;
  if (depth > 3 || r() < 0.2) return pick(JUNK);
  const node: Record<string, Json> = {};
  const n = Math.floor(r() * 5);
  for (let i = 0; i < n; i++) {
    const key = pick(KEYS);
    const value = r() < 0.5 ? gen(r, depth + 1) : pick(JUNK);
    Object.defineProperty(node, key, { value, enumerable: true, writable: true, configurable: true });
  }
  if (r() < 0.5) node.properties = { a: gen(r, depth + 1), b: gen(r, depth + 1) };
  if (r() < 0.3) node.$defs = { A: gen(r, depth + 1), B: { $ref: "#/$defs/A" } };
  return node;
}

describe("fuzz", () => {
  it("never throws, never mutates input (2000 schemas x 5 targets)", () => {
    for (let seed = 0; seed < 2000; seed++) {
      const r = rng(seed);
      const schema = gen(r, 0);
      const before = JSON.stringify(schema);
      for (const target of TARGETS) {
        const first = transformSchema(schema, target);
        transformTools([{ name: "t", inputSchema: schema }], target);
        restoreArguments({ a: "1", b: null, x: [1, { y: null }] }, schema, target);
        if (first.ok) {
          const second = transformSchema(first.schema, target);
          expect(second.changes).toEqual([]);
        }
      }
      expect(JSON.stringify(schema)).toBe(before);
    }
    expect(({} as Record<string, unknown>).a).toBeUndefined();
  });
});
