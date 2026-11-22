/** Shared machinery: JSON Pointers, local $ref resolution, merging, reporting. */

import type { Issue, Severity } from "./model.js";
import {
  type Json,
  type JsonObject,
  deepCopy,
  get,
  has,
  isNumber,
  isObject,
  isString,
  jsonText,
  pyEqual,
  pyIn,
  pyRepr,
  setKey,
  shallow,
} from "./py.js";

/** Upper bound on schema nodes processed per schema ($ref fan-out from untrusted input). */
export const MAX_NODES = 100_000;

export const ANNOTATIONS: ReadonlySet<string> = new Set([
  "title",
  "description",
  "$comment",
  "examples",
  "example",
  "default",
  "deprecated",
  "readOnly",
  "writeOnly",
]);

export const SINGLE_SUBSCHEMA_KEYS = [
  "items",
  "additionalProperties",
  "not",
  "contains",
  "propertyNames",
  "if",
  "then",
  "else",
  "unevaluatedProperties",
  "unevaluatedItems",
  "additionalItems",
] as const;
export const LIST_SUBSCHEMA_KEYS = ["anyOf", "oneOf", "allOf", "prefixItems"] as const;
export const MAP_SUBSCHEMA_KEYS = [
  "properties",
  "patternProperties",
  "$defs",
  "definitions",
  "dependentSchemas",
] as const;

export class RefError extends Error {}

export function escape(token: string): string {
  return token.replaceAll("~", "~0").replaceAll("/", "~1");
}

export function child(path: string, ...tokens: (string | number)[]): string {
  return path + tokens.map((t) => `/${escape(String(t))}`).join("");
}

function unquote(text: string): string {
  // Python urllib.parse.unquote: decode valid %XX sequences, leave the rest.
  return text.replace(/(?:%[0-9a-fA-F]{2})+/g, (run) => {
    try {
      return decodeURIComponent(run);
    } catch {
      return run;
    }
  });
}

/** Resolve a local JSON Pointer reference (`#` or `#/...`) against `root`. */
export function resolveRef(root: Json, ref: string): Json {
  if (ref === "#") return root;
  if (!ref.startsWith("#/")) {
    throw new RefError(`only local references ('#/...') are supported, got ${pyRepr(ref)}`);
  }
  let node: Json = root;
  for (const raw of ref.slice(2).split("/")) {
    const token = unquote(raw).replaceAll("~1", "/").replaceAll("~0", "~");
    if (isObject(node) && has(node, token)) {
      node = node[token] as Json;
    } else if (Array.isArray(node) && /^[0-9]+$/.test(token) && Number(token) < node.length) {
      node = node[Number(token)] as Json;
    } else {
      throw new RefError(`reference ${pyRepr(ref)} does not resolve`);
    }
  }
  return node;
}

/** Like resolveRef, but a boolean `true` schema becomes `{}` (same meaning). */
export function resolveSchema(root: Json, ref: string): Json {
  const target = resolveRef(root, ref);
  return target === true ? {} : target;
}

/** Append dropped constraints to the description (Anthropic SDK convention). */
export function foldIntoDescription(node: JsonObject, dropped: JsonObject): void {
  const keys = Object.keys(dropped);
  if (keys.length === 0) return;
  const note = `{${keys.map((k) => `${k}: ${jsonText(dropped[k] as Json)}`).join(", ")}}`;
  const description = get(node, "description");
  setKey(node, "description", isString(description) && description ? `${description}\n\n${note}` : note);
}

const LOWER_BOUNDS = ["minimum", "exclusiveMinimum", "minLength", "minItems", "minProperties"];
const UPPER_BOUNDS = ["maximum", "exclusiveMaximum", "maxLength", "maxItems", "maxProperties"];

function typeSet(value: Json): Set<string> | null {
  if (isString(value)) return new Set([value]);
  if (Array.isArray(value) && value.every(isString)) return new Set(value as string[]);
  return null;
}

function isRealNumber(value: Json): value is number {
  return isNumber(value);
}

function codePointCompare(a: string, b: string): number {
  // Python sorts strings by code point; JS default sort compares UTF-16 units.
  const ca = [...a];
  const cb = [...b];
  for (let i = 0; i < Math.min(ca.length, cb.length); i++) {
    const d = (ca[i]!.codePointAt(0) as number) - (cb[i]!.codePointAt(0) as number);
    if (d !== 0) return d;
  }
  return ca.length - cb.length;
}

export function pySorted(values: Iterable<string>): string[] {
  return [...values].sort(codePointCompare);
}

/**
 * Merge two schemas that must BOTH hold (allOf semantics); null when that cannot be
 * expressed without changing the accepted values. Annotations from `a` win.
 */
export function mergeAll(a: JsonObject, b: JsonObject): JsonObject | null {
  const out = shallow(a);
  for (const key of Object.keys(b)) {
    const bv = b[key] as Json;
    if (!has(out, key)) {
      setKey(out, key, deepCopy(bv));
      continue;
    }
    const av = out[key] as Json;
    if (pyEqual(av, bv) || ANNOTATIONS.has(key)) continue;
    if (key === "required" && Array.isArray(av) && Array.isArray(bv)) {
      setKey(out, key, [...av, ...bv.filter((r) => !pyIn(r, av))]);
    } else if (key === "properties" && isObject(av) && isObject(bv)) {
      const props = shallow(av);
      for (const name of Object.keys(bv)) {
        const schema = bv[name] as Json;
        if (!has(props, name)) {
          setKey(props, name, deepCopy(schema));
        } else if (!pyEqual(props[name] as Json, schema)) {
          const left = props[name] === true ? {} : (props[name] as Json);
          const right = schema === true ? {} : schema;
          if (!isObject(left) || !isObject(right)) return null;
          const merged = mergeAll(left, right);
          if (merged === null) return null;
          setKey(props, name, merged);
        }
      }
      setKey(out, key, props);
    } else if (key === "type") {
      const at = typeSet(av);
      const bt = typeSet(bv);
      if (at === null || bt === null) return null;
      const both = new Set([...at].filter((t) => bt.has(t)));
      if ((at.has("integer") && bt.has("number")) || (at.has("number") && bt.has("integer"))) {
        both.add("integer");
      }
      if (both.size === 0) return null;
      const avList = Array.isArray(av) ? (av as string[]) : [av as string];
      const ordered = avList.filter((t) => both.has(t));
      ordered.push(...pySorted([...both].filter((t) => !ordered.includes(t))));
      setKey(out, key, ordered.length === 1 ? (ordered[0] as string) : ordered);
    } else if (key === "enum" && Array.isArray(av) && Array.isArray(bv)) {
      const bothEnum = av.filter((v) => pyIn(v, bv));
      if (bothEnum.length === 0) return null;
      setKey(out, key, bothEnum);
    } else if (LOWER_BOUNDS.includes(key) && isRealNumber(av) && isRealNumber(bv)) {
      setKey(out, key, Math.max(av, bv) === av ? av : bv);
    } else if (UPPER_BOUNDS.includes(key) && isRealNumber(av) && isRealNumber(bv)) {
      setKey(out, key, Math.min(av, bv) === av ? av : bv);
    } else if (key === "additionalProperties" && (av === false || bv === false)) {
      setKey(out, key, false);
    } else {
      return null;
    }
  }
  return out;
}

/** `{"type": "null"}`, optionally with annotations. */
export function isNullSchema(schema: Json): boolean {
  return (
    isObject(schema) &&
    get(schema, "type") === "null" &&
    Object.keys(schema).every((k) => k === "type" || ANNOTATIONS.has(k))
  );
}

/** `[path, ref]` for every local $ref that closes a reference cycle. */
export function findRecursiveRefs(root: Json): [string, string][] {
  const found = new Map<string, string>();
  const clean = new Set<string>();

  const walk = (node: Json, path: string, stack: string[]): void => {
    if (Array.isArray(node)) {
      node.forEach((item, i) => walk(item, child(path, i), stack));
      return;
    }
    if (!isObject(node)) return;
    const ref = get(node, "$ref");
    if (isString(ref) && ref.startsWith("#")) {
      if (stack.includes(ref)) {
        if (!found.has(ref)) found.set(ref, path);
      } else if (!clean.has(ref)) {
        let target: Json | undefined;
        try {
          target = resolveRef(root, ref);
        } catch (error) {
          if (!(error instanceof RefError)) throw error;
          target = undefined;
        }
        // Python: `if target is not None` (a JSON null target is skipped too).
        if (target !== undefined && target !== null) {
          const before = found.size;
          walk(target, ref.slice(1), [...stack, ref]);
          if (found.size === before) clean.add(ref);
        }
      }
    }
    for (const key of Object.keys(node)) {
      if (key !== "$ref") walk(node[key] as Json, child(path, key), stack);
    }
  };

  walk(root, "", []);
  return [...found.entries()].map(([ref, path]) => [path, ref]);
}

function collectRefs(node: Json, out: Set<string>): void {
  if (isObject(node)) {
    const ref = get(node, "$ref");
    if (isString(ref) && ref.startsWith("#")) out.add(ref);
    for (const key of Object.keys(node)) {
      if (key !== "$ref") collectRefs(node[key] as Json, out);
    }
  } else if (Array.isArray(node)) {
    for (const item of node) collectRefs(item, out);
  }
}

/** Local references that can reach themselves (merging them would never end). */
export function cyclicRefs(root: Json): ReadonlySet<string> {
  const refs = new Set<string>();
  collectRefs(root, refs);
  const edges = new Map<string, Set<string>>();
  for (const ref of refs) {
    const inner = new Set<string>();
    try {
      collectRefs(resolveRef(root, ref), inner);
    } catch (error) {
      if (!(error instanceof RefError)) throw error;
    }
    edges.set(ref, inner);
  }
  const cyclic = new Set<string>();
  for (const ref of refs) {
    const seen = new Set<string>();
    const pending = [...(edges.get(ref) ?? [])];
    while (pending.length > 0) {
      const current = pending.pop() as string;
      if (current === ref) {
        cyclic.add(ref);
        break;
      }
      if (!seen.has(current)) {
        seen.add(current);
        pending.push(...(edges.get(current) ?? []));
      }
    }
  }
  return cyclic;
}

export function refIn(ref: Json, refs: ReadonlySet<string>): boolean {
  return isString(ref) && refs.has(ref);
}

/** Collects repairs and unrepairable problems while a target processes a schema. */
export class Report {
  changes: Issue[] = [];
  issues: Issue[] = [];
  nodes = 0;
  exhausted = false;

  constructor(readonly tool: string | null = null) {}

  fix(
    code: string,
    path: string,
    message: string,
    options: { lossy?: boolean; severity?: Severity } = {},
  ): void {
    this.changes.push({
      code,
      path,
      message,
      severity: options.severity ?? "error",
      fixable: true,
      lossy: options.lossy ?? false,
      tool: this.tool,
    });
  }

  problem(code: string, path: string, message: string, severity: Severity = "error"): void {
    this.issues.push({ code, path, message, severity, fixable: false, lossy: false, tool: this.tool });
  }

  /** Count one processed node; false once the budget is exhausted. */
  spend(path: string): boolean {
    this.nodes += 1;
    if (this.nodes > MAX_NODES) {
      if (!this.exhausted) {
        this.exhausted = true;
        this.problem(
          "schema.too_large",
          path,
          `schema expands to more than ${MAX_NODES} nodes (likely $ref fan-out); stopped processing`,
        );
      }
      return false;
    }
    return true;
  }
}

const BOOLEAN_KEYWORDS = new Set([
  "additionalProperties",
  "unevaluatedProperties",
  "unevaluatedItems",
  "additionalItems",
]);

/** Replace every boolean `true` subschema with `{}` (identical meaning), once, up front. */
export function trueToEmpty(node: Json, path: string, report: Report, code: string): Json {
  if (node === true) {
    report.fix(code, path, "boolean schema `true` replaced with `{}`");
    return {};
  }
  if (!isObject(node)) return node;
  const out = shallow(node);
  for (const key of SINGLE_SUBSCHEMA_KEYS) {
    if (has(out, key) && !BOOLEAN_KEYWORDS.has(key) && !Array.isArray(out[key])) {
      setKey(out, key, trueToEmpty(out[key] as Json, child(path, key), report, code));
    }
  }
  for (const key of [...LIST_SUBSCHEMA_KEYS, "items"]) {
    const value = get(out, key);
    if (Array.isArray(value)) {
      setKey(
        out,
        key,
        value.map((sub, i) => trueToEmpty(sub, child(path, key, i), report, code)),
      );
    }
  }
  for (const key of MAP_SUBSCHEMA_KEYS) {
    const value = get(out, key);
    if (isObject(value)) {
      const mapped: JsonObject = {};
      for (const name of Object.keys(value)) {
        setKey(mapped, name, trueToEmpty(value[name] as Json, child(path, key, name), report, code));
      }
      setKey(out, key, mapped);
    }
  }
  return out;
}
