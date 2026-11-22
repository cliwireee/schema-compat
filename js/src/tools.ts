/** Tool-level handling: tool shapes, names, request-level limits. */

import {
  MAX_OPTIONAL_PARAMETERS,
  MAX_STRICT_TOOLS,
  MAX_UNION_PARAMETERS,
  complexity,
} from "./anthropic.js";
import { transformSchema } from "./api.js";
import { type Issue, type Target, type ToolsResult, assertTarget } from "./model.js";
import { type Json, type JsonObject, deepCopy, get, has, isObject, isString, pyRepr, setKey } from "./py.js";
import { Report } from "./util.js";
import { sha1Hex } from "./sha1.js";

interface NameRule {
  pattern: RegExp;
  bad: RegExp;
  maxLength: number;
  letterFirst: boolean;
}

// Anthropic: ^[a-zA-Z0-9_-]{1,128}$ (tool-use docs). OpenAI: a-z A-Z 0-9 _ -, <= 64
// (openai-python FunctionDefinition). Gemini: letter or underscore first; letters,
// digits, _ . : -; <= 128 (google-genai FunctionDeclaration).
const NAME_RULES: Record<string, NameRule> = {
  anthropic: {
    pattern: /^[a-zA-Z0-9_-]{1,128}$/,
    bad: /[^a-zA-Z0-9_-]/gu,
    maxLength: 128,
    letterFirst: false,
  },
  openai: { pattern: /^[a-zA-Z0-9_-]{1,64}$/, bad: /[^a-zA-Z0-9_-]/gu, maxLength: 64, letterFirst: false },
  gemini: {
    pattern: /^[a-zA-Z_][a-zA-Z0-9_.:-]{0,127}$/,
    bad: /[^a-zA-Z0-9_.:-]/gu,
    maxLength: 128,
    letterFirst: true,
  },
};
const SCHEMA_KEYS = ["inputSchema", "input_schema", "parameters", "parametersJsonSchema"];

function family(target: string): string {
  return target.split("-")[0] as string;
}

function code(target: string, rule: string): string {
  return `${target.replaceAll("-", "_")}.${rule}`;
}

function rule(target: string): NameRule {
  assertTarget(target);
  return NAME_RULES[family(target)] as NameRule;
}

/** True when `target` accepts `name` as a tool name. */
export function nameIsValid(name: string, target: string): boolean {
  return rule(target).pattern.test(name);
}

/** Rewrite `name` so `target` accepts it (deterministic, identical to the Python package). */
export function sanitizeName(name: string, target: string): string {
  const { pattern, bad, maxLength, letterFirst } = rule(target);
  if (pattern.test(name)) return name;
  let next = name.replace(bad, "_") || "tool";
  if (letterFirst && !/^[A-Za-z_]/.test(next)) next = `_${next}`;
  if (next.length > maxLength) next = `${next.slice(0, maxLength - 9)}_${sha1Hex(name).slice(0, 8)}`;
  return next;
}

function unique(name: string, used: Set<string>, target: string): string {
  if (!used.has(name)) return name;
  const { maxLength } = rule(target);
  for (let n = 2; ; n++) {
    const suffix = `_${n}`;
    const candidate = name.slice(0, maxLength - suffix.length) + suffix;
    if (!used.has(candidate)) return candidate;
  }
}

/** The object holding `name` and the schema (OpenAI Chat nests it under `function`). */
function container(tool: Json): JsonObject | null {
  if (!isObject(tool)) return null;
  const inner = get(tool, "function");
  if (get(tool, "type") === "function" && isObject(inner)) return inner;
  return tool;
}

/**
 * Check and repair a list of tool definitions for `target`, keeping each tool's
 * shape (MCP `inputSchema`, Anthropic `input_schema`, OpenAI Chat/Responses and
 * Gemini `parameters`). `nameMap` maps renamed tools back to their original names.
 */
export function transformTools(tools: Json[], target: string): ToolsResult {
  assertTarget(target);
  const result: ToolsResult = { tools: [], changes: [], issues: [], nameMap: {}, ok: true };
  const seenOriginal = new Set<string>();
  const used = new Set<string>();
  let totalOptional = 0;
  let totalUnions = 0;

  tools.forEach((tool, index) => {
    const holder = container(tool);
    if (holder === null) {
      result.issues.push({
        code: "tool.invalid",
        path: "",
        message: "tool definition must be an object",
        severity: "error",
        fixable: false,
        lossy: false,
        tool: `#${index}`,
      });
      result.tools.push(tool);
      return;
    }
    const newTool = deepCopy(tool);
    const newHolder = container(newTool) as JsonObject;
    const name = get(holder, "name");
    const label = isString(name) && name ? name : `#${index}`;
    const report = new Report(label);

    if (!isString(name) || !name) {
      report.problem("tool.missing_name", "", "tool has no name");
    } else if (seenOriginal.has(name)) {
      report.problem("tool.duplicate_name", "", `tool name ${pyRepr(name)} is used more than once`);
      used.add(name);
    } else {
      seenOriginal.add(name);
      const newName = unique(sanitizeName(name, target), used, target);
      if (newName !== name) {
        setKey(newHolder, "name", newName);
        setKey(result.nameMap as JsonObject, newName, name);
        report.fix(
          code(target, "tool_name"),
          "",
          `tool name ${pyRepr(name)} must match ${rule(target).pattern.source}; renamed to ${pyRepr(newName)}`,
        );
      }
      used.add(newName);
    }

    // A null schema field means "no schema"; providers reject null there.
    for (const key of SCHEMA_KEYS) {
      if (has(holder, key) && holder[key] === null) {
        delete newHolder[key];
        report.fix(code(target, "null_schema"), "", `${key} is null; removed`);
      }
    }
    const schemaKey = SCHEMA_KEYS.find((k) => has(holder, k) && holder[k] !== null) ?? null;
    if (schemaKey === "parametersJsonSchema" && target === "gemini") {
      report.problem(
        "gemini.parameters_json_schema",
        "",
        "parametersJsonSchema uses Gemini's JSON Schema mode, which this version does not check; use parameters to have it checked",
        "warning",
      );
    } else if (schemaKey !== null) {
      const schemaResult = transformSchema(holder[schemaKey] as Json, target, { tool: label });
      report.changes.push(...schemaResult.changes);
      let remaining = schemaResult.issues;
      const isRootEmpty = (i: Issue) => i.code === "gemini.empty_object" && i.path === "";
      if (target === "gemini" && remaining.some(isRootEmpty)) {
        delete newHolder[schemaKey];
        remaining = remaining.filter((i) => !isRootEmpty(i));
        report.fix(
          "gemini.empty_parameters",
          "",
          "Gemini rejects an OBJECT without properties; omitted parameters (the tool takes no arguments)",
        );
      } else {
        setKey(newHolder, schemaKey, schemaResult.schema);
      }
      report.issues.push(...remaining);
      if (target === "anthropic-strict" && has(newHolder, schemaKey)) {
        const [optional, unions] = complexity(newHolder[schemaKey] as Json);
        totalOptional += optional;
        totalUnions += unions;
      }
    } else if (family(target) === "anthropic") {
      setKey(newHolder, "input_schema", { type: "object", properties: {} });
      report.fix(
        code(target, "missing_input_schema"),
        "",
        "Anthropic requires input_schema; added an empty object schema",
      );
    }

    result.tools.push(newTool);
    result.changes.push(...report.changes);
    result.issues.push(...report.issues);
  });

  if ((target as Target) === "anthropic-strict") {
    const limits: [number, number, string, string][] = [
      [tools.length, MAX_STRICT_TOOLS, "too_many_strict_tools", "strict tools"],
      [totalOptional, MAX_OPTIONAL_PARAMETERS, "too_many_optional", "optional parameters"],
      [totalUnions, MAX_UNION_PARAMETERS, "too_many_unions", "union-typed parameters"],
    ];
    for (const [count, limit, name, label] of limits) {
      if (count > limit) {
        result.issues.push({
          code: `anthropic_strict.${name}`,
          path: "",
          message: `${count} ${label} across the request; the limit is ${limit}`,
          severity: "error",
          fixable: false,
          lossy: false,
          tool: null,
        });
      }
    }
  }
  result.ok = !result.issues.some((i) => i.severity === "error");
  return result;
}
