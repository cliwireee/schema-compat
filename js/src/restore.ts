/** Undo the reversible, lossy repairs on the arguments a model sends back. */

import { assertTarget } from "./model.js";
import { type Json, type JsonObject, filterKeys, get, has, isObject, isString, jsonText, pyIn, setKey } from "./py.js";
import { allowsNull } from "./shared.js";
import { RefError, resolveRef } from "./util.js";

const MAX_DEPTH = 100;

function deref(input: Json, root: Json): Json {
  let schema = input;
  const seen = new Set<string>();
  while (isObject(schema) && isString(get(schema, "$ref"))) {
    const ref = schema.$ref as string;
    if (seen.has(ref)) return {};
    seen.add(ref);
    let target: Json;
    try {
      target = resolveRef(root, ref);
    } catch (error) {
      if (error instanceof RefError) return {};
      throw error;
    }
    const siblings = filterKeys(schema, (k) => k !== "$ref");
    if (isObject(target)) {
      const merged: JsonObject = {};
      for (const key of Object.keys(target)) setKey(merged, key, target[key] as Json);
      for (const key of Object.keys(siblings)) setKey(merged, key, siblings[key] as Json);
      schema = merged;
    } else {
      schema = target;
    }
  }
  return schema;
}

function jsonTypeMatches(value: Json, type: string): boolean {
  switch (type) {
    case "null":
      return value === null;
    case "boolean":
      return typeof value === "boolean";
    case "integer":
      return typeof value === "number" && Number.isInteger(value);
    case "number":
      return typeof value === "number";
    case "string":
      return typeof value === "string";
    case "array":
      return Array.isArray(value);
    case "object":
      return isObject(value);
    default:
      return false;
  }
}

function literalsOf(schema: JsonObject): Json[] {
  const enumValues = get(schema, "enum");
  const literals: Json[] = Array.isArray(enumValues) ? [...enumValues] : [];
  if (has(schema, "const")) literals.push(schema.const as Json);
  return literals;
}

/** Loose shape check used to pick the anyOf/oneOf branch a value belongs to. */
function matches(value: Json, input: Json, root: Json, target: string): boolean {
  const schema = deref(input, root);
  if (!isObject(schema)) return schema !== false;
  const literals = literalsOf(schema);
  if (literals.length > 0) {
    if (pyIn(value, literals)) return true;
    return (
      target === "gemini" &&
      isString(value) &&
      literals.some((v) => !isString(v) && jsonText(v) === value)
    );
  }
  const typeValue = get(schema, "type");
  if (isString(typeValue)) return jsonTypeMatches(value, typeValue);
  if (Array.isArray(typeValue)) {
    return typeValue.some((t) => isString(t) && jsonTypeMatches(value, t));
  }
  if (isObject(get(schema, "properties"))) return isObject(value);
  if (has(schema, "items") || has(schema, "prefixItems")) return Array.isArray(value);
  return true;
}

function walk(input: Json, schemaInput: Json, root: Json, target: string, depth: number): Json {
  let value = input;
  if (depth > MAX_DEPTH) return value;
  const schema = deref(schemaInput, root);
  if (!isObject(schema)) return value;

  const allOf = get(schema, "allOf");
  for (const entry of Array.isArray(allOf) ? allOf : []) {
    value = walk(value, entry, root, target, depth + 1);
  }
  for (const key of ["anyOf", "oneOf"]) {
    const branches = get(schema, key);
    if (Array.isArray(branches)) {
      for (const branch of branches) {
        if (matches(value, branch, root, target)) {
          value = walk(value, branch, root, target, depth + 1);
          break;
        }
      }
    }
  }

  if (target === "gemini" && isString(value)) {
    const literals = literalsOf(schema);
    if (!pyIn(value, literals)) {
      for (const literal of literals) {
        if (!isString(literal) && jsonText(literal) === value) return literal;
      }
    }
  }

  if (isObject(value)) {
    const rawProps = get(schema, "properties");
    const props: JsonObject = isObject(rawProps) ? rawProps : {};
    const rawRequired = get(schema, "required");
    const required: Json[] = Array.isArray(rawRequired) ? rawRequired : [];
    const out: JsonObject = {};
    for (const key of Object.keys(value)) {
      const item = value[key] as Json;
      const sub = get(props, key);
      if (
        target === "openai-strict" &&
        item === null &&
        sub !== null &&
        !pyIn(key, required) &&
        !allowsNull(deref(sub, root))
      ) {
        continue; // null sent only because strict mode made the property required
      }
      setKey(out, key, sub !== null ? walk(item, sub, root, target, depth + 1) : item);
    }
    return out;
  }
  if (Array.isArray(value)) {
    let prefix = get(schema, "prefixItems");
    const rawItems = get(schema, "items");
    if (!Array.isArray(prefix) && Array.isArray(rawItems)) prefix = rawItems;
    const items = isObject(rawItems) ? rawItems : null;
    return value.map((item, i) => {
      const sub = Array.isArray(prefix) && i < prefix.length ? (prefix[i] as Json) : items;
      return sub !== null ? walk(item, sub, root, target, depth + 1) : item;
    });
  }
  return value;
}

/**
 * Map tool-call arguments produced against a transformed schema back to the original.
 * `schema` is the ORIGINAL schema passed to transformSchema.
 *
 * - `gemini`: enum/const values sent as strings ("1", "true") become the original
 *   numbers/booleans again.
 * - `openai-strict`: properties that were optional in the original schema and came
 *   back as null are removed, unless the original itself allowed null.
 *
 * Other targets return the arguments unchanged (as a new object). Never mutates.
 */
export function restoreArguments(argumentsValue: Json, schema: Json, target: string): Json {
  assertTarget(target);
  return walk(argumentsValue, schema, schema, target, 0);
}
