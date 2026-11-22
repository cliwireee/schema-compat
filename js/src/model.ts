import type { Json } from "./py.js";

export type Target = "anthropic" | "anthropic-strict" | "openai" | "openai-strict" | "gemini";
export const TARGETS: readonly Target[] = [
  "anthropic",
  "anthropic-strict",
  "openai",
  "openai-strict",
  "gemini",
] as const;

export type Severity = "error" | "warning";

/**
 * A problem found in a schema or tool for one target.
 *
 * `path` is a JSON Pointer into the schema (`""` is the root); tool-level problems
 * use `""` and name the tool in `tool`. `fixable`: transformSchema/transformTools
 * repairs it. `lossy`: the repaired schema accepts some value the original rejects;
 * restoreArguments undoes the reversible cases.
 */
export interface Issue {
  code: string;
  path: string;
  message: string;
  severity: Severity;
  fixable: boolean;
  lossy: boolean;
  tool: string | null;
}

export interface SchemaResult {
  /** The repaired schema (the input is never mutated). */
  schema: Json;
  /** Every repair that was applied. */
  changes: Issue[];
  /** What could not be repaired, plus warnings. */
  issues: Issue[];
  /** True when `issues` contains no error. */
  ok: boolean;
}

export interface ToolsResult {
  /** The tools, each in its original shape (MCP, Anthropic, OpenAI or Gemini). */
  tools: Json[];
  changes: Issue[];
  issues: Issue[];
  /** New tool name -> original name, for every renamed tool. */
  nameMap: Record<string, string>;
  ok: boolean;
}

export function isTarget(value: string): value is Target {
  return (TARGETS as readonly string[]).includes(value);
}

export function assertTarget(value: string): asserts value is Target {
  if (!isTarget(value)) {
    throw new Error(`unknown target '${value}'; expected one of ${TARGETS.join(", ")}`);
  }
}
