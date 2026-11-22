/** Public entry points. */

import { AnthropicStrictTransformer, AnthropicTransformer } from "./anthropic.js";
import { GeminiTransformer } from "./gemini.js";
import { type Issue, type SchemaResult, assertTarget } from "./model.js";
import { OpenAIStrictTransformer, OpenAITransformer } from "./openai.js";
import type { Json } from "./py.js";
import { transformTools } from "./tools.js";
import { Report } from "./util.js";

const TRANSFORMERS = {
  anthropic: AnthropicTransformer,
  "anthropic-strict": AnthropicStrictTransformer,
  openai: OpenAITransformer,
  "openai-strict": OpenAIStrictTransformer,
  gemini: GeminiTransformer,
} as const;

/**
 * Repair a tool's parameter schema so `target` accepts it. The input is never mutated.
 * `result.schema` is the repaired schema, `result.changes` what changed, `result.issues`
 * what could not be repaired (`result.ok` is false if any of them is an error).
 */
export function transformSchema(
  schema: Json,
  target: string,
  options: { tool?: string | null } = {},
): SchemaResult {
  assertTarget(target);
  const report = new Report(options.tool ?? null);
  const out = new TRANSFORMERS[target](schema, report).run();
  return {
    schema: out,
    changes: report.changes,
    issues: report.issues,
    ok: !report.issues.some((i) => i.severity === "error"),
  };
}

/** Every problem `target` has with `schema`; `fixable` ones are repaired by transformSchema. */
export function checkSchema(schema: Json, target: string): Issue[] {
  const result = transformSchema(schema, target);
  return [...result.changes, ...result.issues];
}

/** Every problem `target` has with these tool definitions (names, schemas, limits). */
export function checkTools(tools: Json[], target: string): Issue[] {
  const result = transformTools(tools, target);
  return [...result.changes, ...result.issues];
}
