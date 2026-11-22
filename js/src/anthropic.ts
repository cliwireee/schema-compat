/**
 * Anthropic tool `input_schema` and strict tool use (`strict: true`).
 * Sources: Anthropic tool-use docs; structured-outputs "JSON Schema limitations"
 * and "Schema complexity limits"; the SDK's InputSchemaTyped.
 */

import {
  type Json,
  type JsonObject,
  deepCopy,
  filterKeys,
  get,
  has,
  isObject,
  isString,
  pop,
  pyIn,
  pyRepr,
  setKey,
  shallow,
  truthy,
} from "./py.js";
import { fixRootObject } from "./shared.js";
import {
  ANNOTATIONS,
  RefError,
  Report,
  child,
  cyclicRefs,
  findRecursiveRefs,
  foldIntoDescription,
  mergeAll,
  pySorted,
  refIn,
  resolveRef,
  resolveSchema,
  trueToEmpty,
} from "./util.js";

export const SUPPORTED_FORMATS: ReadonlySet<string> = new Set([
  "date-time",
  "time",
  "date",
  "duration",
  "email",
  "hostname",
  "uri",
  "ipv4",
  "ipv6",
  "uuid",
]);
const NUMERIC_CONSTRAINTS = ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"];
const STRING_CONSTRAINTS = ["minLength", "maxLength"];
const ARRAY_CONSTRAINTS = [
  "maxItems",
  "uniqueItems",
  "contains",
  "minContains",
  "maxContains",
  "additionalItems",
  "unevaluatedItems",
];

export const MAX_STRICT_TOOLS = 20;
export const MAX_OPTIONAL_PARAMETERS = 24;
export const MAX_UNION_PARAMETERS = 16;

/** Regex features the docs list as unsupported: backreferences, lookaround, \b/\B. */
export function unsupportedRegexFeatures(pattern: string): string[] {
  const found = new Set<string>();
  const n = pattern.length;
  let i = 0;
  let inClass = false;
  while (i < n) {
    const c = pattern[i] as string;
    if (c === "\\" && i + 1 < n) {
      const next = pattern[i + 1] as string;
      if (!inClass) {
        if ("123456789".includes(next)) found.add("backreference");
        else if (next === "k" && i + 2 < n && "<{'".includes(pattern[i + 2] as string)) {
          found.add("backreference");
        } else if (next === "b" || next === "B") found.add("word boundary");
      }
      i += 2;
      continue;
    }
    if (inClass) {
      if (c === "]") inClass = false;
      i += 1;
      continue;
    }
    if (c === "[") {
      inClass = true;
      i += 1;
      if (i < n && pattern[i] === "^") i += 1;
      if (i < n && pattern[i] === "]") i += 1; // a leading "]" is a literal
      continue;
    }
    if (c === "(" && ["(?=", "(?!", "(?<=", "(?<!"].some((p) => pattern.startsWith(p, i))) {
      found.add("lookaround");
    }
    i += 1;
  }
  return pySorted(found);
}

/** Non-strict tools accept JSON Schema (draft 2020-12); only the root is constrained. */
export class AnthropicTransformer {
  private readonly schema: Json;
  constructor(
    schema: Json,
    private readonly report: Report,
  ) {
    this.schema = deepCopy(schema);
  }

  run(): Json {
    return fixRootObject(this.schema, this.report, "anthropic");
  }
}

export class AnthropicStrictTransformer {
  private report: Report;
  private readonly original: Json;
  private readonly cyclic: ReadonlySet<string>;

  constructor(schema: Json, report: Report) {
    this.report = report;
    this.original = trueToEmpty(deepCopy(schema), "", report, "anthropic_strict.boolean_schema");
    this.cyclic = cyclicRefs(this.original);
  }

  run(): Json {
    const r = this.report;
    let root = deepCopy(this.original);
    if (isObject(root) && !has(root, "type") && Array.isArray(get(root, "anyOf"))) {
      r.problem(
        "anthropic_strict.root_not_object",
        "",
        'the root of input_schema must be type: "object", not anyOf',
      );
      return root;
    }
    root = fixRootObject(root, r, "anthropic_strict");
    if (!isObject(root)) return root;
    for (const [path, ref] of findRecursiveRefs(this.original)) {
      r.problem(
        "anthropic_strict.recursive_schema",
        path,
        `recursive schemas are not supported (${ref} refers back to itself)`,
      );
    }
    return this.visit(root, "", []);
  }

  private deref(schema: Json, stack: string[]): Json {
    if (!isObject(schema) || !has(schema, "$ref")) return schema;
    const ref = schema.$ref as Json;
    if (!isString(ref) || stack.includes(ref)) return null; // a self-referencing allOf cannot be merged
    let target: Json;
    try {
      target = resolveSchema(this.original, ref);
    } catch (error) {
      if (error instanceof RefError) return null;
      throw error;
    }
    if (!isObject(target)) return null;
    return mergeAll(
      filterKeys(schema, (k) => k !== "$ref"),
      deepCopy(target),
    );
  }

  /** Collect unrepairable problems inside `entries` without changing them. */
  private probe(entries: Json[], path: string, stack: string[]): void {
    const real = this.report;
    this.report = new Report(real.tool);
    this.report.nodes = real.nodes;
    let probe: Report;
    try {
      entries.forEach((sub, i) => this.visit(deepCopy(sub), child(path, "allOf", i), stack));
    } finally {
      probe = this.report;
      this.report = real;
    }
    real.nodes = probe.nodes;
    real.issues.push(...probe.issues);
  }

  private visit(input: Json, path: string, stack: string[]): Json {
    const r = this.report;
    if (input === true) {
      r.fix("anthropic_strict.boolean_schema", path, "boolean schema `true` replaced with `{}`");
      return {};
    }
    if (!isObject(input)) {
      r.problem("schema.invalid", path, "schema must be a JSON object");
      return input;
    }
    if (!r.spend(path)) return input;
    let node = shallow(input);
    const dropped: JsonObject = {};

    const ref = get(node, "$ref");
    if (ref !== null) {
      if (!isString(ref) || !ref.startsWith("#")) {
        r.problem(
          "anthropic_strict.external_ref",
          child(path, "$ref"),
          `external $ref is not supported: ${pyRepr(ref)}`,
        );
      } else {
        try {
          resolveRef(this.original, ref);
        } catch (error) {
          if (!(error instanceof RefError)) throw error;
          r.problem("schema.unresolvable_ref", child(path, "$ref"), error.message);
        }
      }
    }

    if (has(node, "oneOf")) {
      if (has(node, "anyOf")) {
        r.problem(
          "anthropic_strict.oneof",
          child(path, "oneOf"),
          "oneOf is not supported and cannot become anyOf because anyOf is already set",
        );
      } else {
        setKey(node, "anyOf", pop(node, "oneOf"));
        r.fix(
          "anthropic_strict.oneof",
          child(path, "oneOf"),
          "oneOf is not supported (use anyOf); replaced by anyOf (no longer exclusive)",
          { lossy: true },
        );
      }
    }

    if (has(node, "not")) {
      setKey(dropped, "not", pop(node, "not"));
      r.fix(
        "anthropic_strict.not",
        child(path, "not"),
        '"not" is not supported; removed and added to the description',
        { lossy: true },
      );
    }

    if (has(node, "allOf") && Array.isArray(node.allOf)) {
      const entries = pop(node, "allOf") as Json[];
      let merged: JsonObject | null = node;
      for (const entry of entries) {
        // `true` accepts everything: neutral in an allOf.
        let resolved: Json = entry === true ? {} : this.deref(entry, stack);
        if (isObject(entry) && refIn(get(entry, "$ref"), this.cyclic)) resolved = null;
        merged = merged !== null && isObject(resolved) ? mergeAll(merged, resolved) : null;
      }
      if (merged === null) {
        setKey(node, "allOf", entries);
        this.probe(entries, path, stack);
        r.problem(
          "anthropic_strict.allof",
          child(path, "allOf"),
          "allOf with conflicting schemas is not supported and they cannot be merged",
        );
      } else {
        r.fix("anthropic_strict.allof", child(path, "allOf"), "allOf merged into one schema");
        return this.visit(merged, path, stack);
      }
    }

    for (const key of ["enum", "const"]) {
      if (has(node, key)) {
        const values = key === "enum" ? (node[key] as Json) : [node[key] as Json];
        if (!Array.isArray(values) || values.some((v) => Array.isArray(v) || isObject(v))) {
          r.problem(
            "anthropic_strict.complex_enum",
            child(path, key),
            `${key} values must be strings, numbers, booleans or null`,
          );
        }
      }
    }

    for (const key of [...NUMERIC_CONSTRAINTS, ...STRING_CONSTRAINTS, ...ARRAY_CONSTRAINTS]) {
      if (has(node, key)) {
        setKey(dropped, key, pop(node, key));
        r.fix(
          "anthropic_strict.unsupported_constraint",
          child(path, key),
          `"${key}" is not supported; removed and added to the description`,
          { lossy: true },
        );
      }
    }
    if (has(node, "minItems") && !pyIn(node.minItems as Json, [0, 1])) {
      setKey(dropped, "minItems", pop(node, "minItems"));
      r.fix(
        "anthropic_strict.unsupported_constraint",
        child(path, "minItems"),
        "only minItems 0 or 1 is supported; removed and added to the description",
        { lossy: true },
      );
    }
    if (has(node, "format")) {
      const format = node.format as Json;
      if (!isString(format) || !SUPPORTED_FORMATS.has(format)) {
        setKey(dropped, "format", pop(node, "format"));
        r.fix(
          "anthropic_strict.format",
          child(path, "format"),
          `string format ${pyRepr(format)} is not supported; removed and added to the description`,
          { lossy: true },
        );
      }
    }
    const pattern = get(node, "pattern");
    if (isString(pattern)) {
      const features = unsupportedRegexFeatures(pattern);
      if (features.length > 0) {
        setKey(dropped, "pattern", pop(node, "pattern"));
        r.fix(
          "anthropic_strict.pattern",
          child(path, "pattern"),
          `regex uses unsupported features (${features.join(", ")}); removed and added to the description`,
          { lossy: true },
        );
      }
    }

    let tupleItems: Json[] | null = null;
    if (Array.isArray(get(node, "items"))) tupleItems = pop(node, "items") as Json[];
    if (Array.isArray(get(node, "prefixItems"))) tupleItems = pop(node, "prefixItems") as Json[];
    if (tupleItems !== null) {
      const schemas: Json[] = tupleItems.filter((s) => isObject(s));
      const items = get(node, "items");
      if (isObject(items) && truthy(items)) schemas.push(items);
      if (schemas.length > 0) {
        setKey(node, "items", schemas.length === 1 ? (schemas[0] as Json) : { anyOf: schemas });
      }
      r.fix(
        "anthropic_strict.tuple_items",
        path,
        "tuple arrays are an array constraint that is not supported; items became anyOf of the positions",
        { lossy: true },
      );
    }

    foldIntoDescription(node, dropped);

    // After unsupported constraints are gone: only real siblings are distributed.
    const anyOf = get(node, "anyOf");
    if (Array.isArray(anyOf)) {
      const siblings = filterKeys(node, (k) => k !== "anyOf" && !ANNOTATIONS.has(k));
      const siblingKeys = Object.keys(siblings);
      if (siblingKeys.length > 0 && path === "") {
        r.problem(
          "anthropic_strict.anyof_siblings",
          child(path, "anyOf"),
          'the root must be type: "object" and anyOf cannot sit next to type; move the anyOf into a property',
        );
      } else if (siblingKeys.length > 0) {
        const branches = anyOf.map((b) => (isObject(b) ? mergeAll(siblings, b) : null));
        if (branches.some((b) => b === null)) {
          r.problem(
            "anthropic_strict.anyof_siblings",
            child(path, "anyOf"),
            "anyOf cannot sit next to keywords such as type or properties, and they conflict with its branches",
          );
        } else {
          node = filterKeys(node, (k) => !siblingKeys.includes(k));
          setKey(node, "anyOf", branches as Json[]);
          r.fix(
            "anthropic_strict.anyof_siblings",
            child(path, "anyOf"),
            `anyOf cannot sit next to ${pyRepr(pySorted(siblingKeys))}; moved them into each branch`,
          );
        }
      }
    }

    const typeValue = get(node, "type");
    const isObjectType =
      typeValue === "object" ||
      (Array.isArray(typeValue) && typeValue.includes("object")) ||
      (typeValue === null && isObject(get(node, "properties")));
    if (isObjectType) {
      if (!has(node, "additionalProperties")) {
        setKey(node, "additionalProperties", false);
        r.fix(
          "anthropic_strict.additional_properties",
          path,
          "objects must set additionalProperties: false; added",
        );
      } else if (node.additionalProperties !== false) {
        r.problem(
          "anthropic_strict.additional_properties",
          child(path, "additionalProperties"),
          "additionalProperties must be false; for arbitrary keys use an array of {key, value} objects",
        );
      }
      if (!truthy(get(node, "properties"))) {
        r.problem(
          "anthropic_strict.empty_object",
          path,
          "an object without properties can only ever be {} under strict mode",
          "warning",
        );
      }
    }

    const props = get(node, "properties");
    if (isObject(props)) {
      const mapped: JsonObject = {};
      for (const name of Object.keys(props)) {
        setKey(mapped, name, this.visit(props[name] as Json, child(path, "properties", name), stack));
      }
      setKey(node, "properties", mapped);
    }
    const items = get(node, "items");
    if (isObject(items) || typeof items === "boolean") {
      setKey(node, "items", this.visit(items, child(path, "items"), stack));
    }
    const branches = get(node, "anyOf");
    if (Array.isArray(branches)) {
      setKey(
        node,
        "anyOf",
        branches.map((sub, i) => this.visit(sub, child(path, "anyOf", i), stack)),
      );
    }
    for (const key of ["$defs", "definitions"]) {
      const defs = get(node, key);
      if (isObject(defs)) {
        const mapped: JsonObject = {};
        for (const name of Object.keys(defs)) {
          setKey(mapped, name, this.visit(defs[name] as Json, child(path, key, name), stack));
        }
        setKey(node, key, mapped);
      }
    }
    return node;
  }
}

/** [optional parameters, union-typed parameters] in one strict schema, as documented. */
export function complexity(schema: Json): [number, number] {
  let optional = 0;
  let unions = 0;
  const walk = (node: Json): void => {
    if (Array.isArray(node)) {
      node.forEach(walk);
      return;
    }
    if (!isObject(node)) return;
    const props = get(node, "properties");
    if (isObject(props)) {
      const rawRequired = get(node, "required");
      const required: Json[] = Array.isArray(rawRequired) ? rawRequired : [];
      for (const name of Object.keys(props)) {
        if (!pyIn(name, required)) optional += 1;
        const sub = props[name] as Json;
        if (isObject(sub) && (Array.isArray(get(sub, "anyOf")) || Array.isArray(get(sub, "type")))) {
          unions += 1;
        }
      }
    }
    for (const key of Object.keys(node)) walk(node[key] as Json);
  };
  walk(schema);
  return [optional, unions];
}
