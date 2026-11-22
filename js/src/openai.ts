/**
 * OpenAI function calling (`parameters`) and strict mode (`strict: true`).
 * Sources: openai-python FunctionDefinition and lib/_pydantic.to_strict_json_schema;
 * openai-agents strict_schema.ensure_strict_json_schema.
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
  setKey,
  shallow,
} from "./py.js";
import { allowsNull, fixRootObject, makeNullable } from "./shared.js";
import {
  RefError,
  Report,
  child,
  cyclicRefs,
  mergeAll,
  resolveRef,
  resolveSchema,
  trueToEmpty,
} from "./util.js";

/** Non-strict function calling: only the root must be an object. */
export class OpenAITransformer {
  private readonly schema: Json;
  constructor(
    schema: Json,
    private readonly report: Report,
  ) {
    this.schema = deepCopy(schema);
  }

  run(): Json {
    return fixRootObject(this.schema, this.report, "openai");
  }
}

const ROOT_ANYOF =
  "the root of a strict schema must be a plain object and must not use anyOf/oneOf";

export class OpenAIStrictTransformer {
  private report: Report;
  private readonly original: Json;
  private readonly cyclic: ReadonlySet<string>;

  constructor(schema: Json, report: Report) {
    this.report = report;
    this.original = trueToEmpty(deepCopy(schema), "", report, "openai_strict.boolean_schema");
    this.cyclic = cyclicRefs(this.original);
  }

  run(): Json {
    const r = this.report;
    let root = deepCopy(this.original);
    if (isObject(root) && ["anyOf", "oneOf"].some((k) => Array.isArray(get(root as JsonObject, k)))) {
      r.problem("openai_strict.root_anyof", "", ROOT_ANYOF);
      return root;
    }
    root = fixRootObject(root, r, "openai_strict");
    if (!isObject(root)) return root;
    const out = this.visit(root, "", []);
    // Inlining a root $ref or merging a root allOf can produce a root anyOf.
    if (isObject(out) && Array.isArray(get(out, "anyOf"))) {
      r.problem("openai_strict.root_anyof", "", ROOT_ANYOF);
    }
    return out;
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

  private checkRef(ref: Json, path: string): boolean {
    if (!isString(ref)) {
      this.report.problem("schema.invalid", child(path, "$ref"), "$ref must be a string");
      return false;
    }
    try {
      resolveRef(this.original, ref);
    } catch (error) {
      if (!(error instanceof RefError)) throw error;
      this.report.problem("schema.unresolvable_ref", child(path, "$ref"), error.message);
      return false;
    }
    return true;
  }

  private visit(input: Json, path: string, stack: string[]): Json {
    const r = this.report;
    if (input === true) {
      r.fix("openai_strict.boolean_schema", path, "boolean schema `true` replaced with `{}`");
      return {};
    }
    if (!isObject(input)) {
      r.problem("schema.invalid", path, "schema must be a JSON object");
      return input;
    }
    if (!r.spend(path)) return input;
    const node = shallow(input);

    if (has(node, "$ref")) {
      const ref = node.$ref as Json;
      if (!this.checkRef(ref, path)) return node;
      const refText = ref as string;
      if (Object.keys(node).length === 1) return node; // processed under $defs
      if (stack.includes(refText)) {
        r.problem(
          "openai_strict.ref_siblings",
          path,
          `$ref cannot have sibling keywords in strict mode, and recursive ${refText} cannot be inlined`,
        );
        return node;
      }
      const siblings = filterKeys(node, (k) => k !== "$ref");
      const target = resolveSchema(this.original, refText);
      const merged = isObject(target) ? mergeAll(siblings, deepCopy(target)) : null;
      if (merged === null) {
        r.problem(
          "openai_strict.ref_siblings",
          path,
          `$ref cannot have sibling keywords in strict mode, and ${refText} conflicts with them`,
        );
        return node;
      }
      r.fix(
        "openai_strict.ref_siblings",
        path,
        `$ref cannot have sibling keywords in strict mode; inlined ${refText}`,
      );
      return this.visit(merged, path, [...stack, refText]);
    }

    if (has(node, "oneOf")) {
      if (has(node, "anyOf")) {
        r.problem(
          "openai_strict.oneof",
          child(path, "oneOf"),
          "oneOf is not supported and cannot become anyOf because anyOf is already set",
        );
      } else {
        setKey(node, "anyOf", pop(node, "oneOf"));
        r.fix(
          "openai_strict.oneof",
          child(path, "oneOf"),
          "oneOf is not supported in strict mode; replaced by anyOf (no longer exclusive)",
          { lossy: true },
        );
      }
    }

    if (has(node, "allOf") && Array.isArray(node.allOf)) {
      const entries = pop(node, "allOf") as Json[];
      let mergedAll: JsonObject | null = node;
      for (const entry of entries) {
        // `true` accepts everything: neutral in an allOf.
        let resolved: Json = entry === true ? {} : entry;
        if (isObject(entry) && has(entry, "$ref") && isString(entry.$ref)) {
          const entryRef = entry.$ref;
          let target: Json | undefined;
          try {
            target = resolveSchema(this.original, entryRef);
          } catch (error) {
            if (!(error instanceof RefError)) throw error;
            target = undefined;
          }
          const sib = filterKeys(entry, (k) => k !== "$ref");
          resolved =
            isObject(target) && !stack.includes(entryRef) && !this.cyclic.has(entryRef)
              ? mergeAll(sib, deepCopy(target))
              : null;
        }
        mergedAll = mergedAll !== null && isObject(resolved) ? mergeAll(mergedAll, resolved) : null;
      }
      if (mergedAll === null) {
        setKey(node, "allOf", entries);
        this.probe(entries, path, stack);
        r.problem(
          "openai_strict.allof",
          child(path, "allOf"),
          "allOf schemas conflict and cannot be merged into one schema; OpenAI's own strict conversion rejects this",
        );
      } else {
        r.fix("openai_strict.allof", child(path, "allOf"), "allOf merged into one schema");
        return this.visit(mergedAll, path, stack);
      }
    }

    if (has(node, "default") && node.default === null) {
      pop(node, "default");
      r.fix("openai_strict.null_default", child(path, "default"), "default: null removed");
    }

    let typeValue = get(node, "type");
    if (typeValue === null && isObject(get(node, "properties"))) {
      setKey(node, "type", "object");
      typeValue = "object";
      r.fix("openai_strict.object_type", child(path, "type"), 'added type: "object"');
    }
    const isObjectType =
      typeValue === "object" || (Array.isArray(typeValue) && typeValue.includes("object"));

    let props = get(node, "properties");
    if (isObject(props)) {
      const mapped: JsonObject = {};
      for (const name of Object.keys(props)) {
        setKey(mapped, name, this.visit(props[name] as Json, child(path, "properties", name), stack));
      }
      setKey(node, "properties", mapped);
      props = mapped;
    }
    const items = get(node, "items");
    if (isObject(items) || typeof items === "boolean") {
      setKey(node, "items", this.visit(items, child(path, "items"), stack));
    }
    for (const key of ["anyOf", "prefixItems"]) {
      const list = get(node, key);
      if (Array.isArray(list)) {
        setKey(
          node,
          key,
          list.map((sub, i) => this.visit(sub, child(path, key, i), stack)),
        );
      }
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

    if (isObjectType) {
      if (!has(node, "additionalProperties")) {
        setKey(node, "additionalProperties", false);
        r.fix(
          "openai_strict.additional_properties",
          path,
          "strict mode requires additionalProperties: false on every object; added",
        );
      } else if (node.additionalProperties !== false) {
        r.problem(
          "openai_strict.additional_properties",
          child(path, "additionalProperties"),
          "strict mode requires additionalProperties: false; objects with arbitrary keys (maps) cannot be expressed",
        );
      }
      if (isObject(props)) {
        const rawRequired = get(node, "required");
        const required: Json[] = Array.isArray(rawRequired) ? [...rawRequired] : [];
        const names = Object.keys(props);
        const optional = names.filter((name) => !pyIn(name, required));
        for (const name of optional) {
          const alreadyNullable = allowsNull(props[name] as Json);
          setKey(props, name, makeNullable(props[name] as Json));
          r.fix(
            "openai_strict.optional_property",
            child(path, "properties", name),
            "strict mode requires every property in required; made it required" +
              (alreadyNullable
                ? " (it already accepts null)"
                : " and nullable (restore_arguments() drops the nulls)"),
            { lossy: !alreadyNullable },
          );
        }
        if (optional.length > 0) {
          // Property order, as OpenAI's own SDKs emit it.
          setKey(node, "required", [
            ...names,
            ...required.filter((n) => !(isString(n) && has(props as JsonObject, n))),
          ]);
        } else if (!Array.isArray(get(node, "required"))) {
          setKey(node, "required", []);
          r.fix(
            "openai_strict.required",
            child(path, "required"),
            "strict mode requires a required array; added []",
          );
        }
      }
    }
    return node;
  }
}
