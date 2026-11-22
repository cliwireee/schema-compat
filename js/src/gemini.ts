/**
 * Gemini `functionDeclarations[].parameters` (OpenAPI 3.0 Schema subset).
 * Sources: google-genai `types.Schema` (extra='forbid'), `_transformers.process_schema`
 * and `handle_null_fields`; API errors reported on GitHub.
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
  jsonText,
  pop,
  pyEqual,
  pyRepr,
  setKey,
  shallow,
  truthy,
} from "./py.js";
import {
  RefError,
  Report,
  child,
  cyclicRefs,
  foldIntoDescription,
  isNullSchema,
  mergeAll,
  refIn,
  resolveSchema,
  trueToEmpty,
} from "./util.js";

/** Fields of google-genai `types.Schema` (JSON names). */
export const ALLOWED: ReadonlySet<string> = new Set([
  "type",
  "format",
  "title",
  "description",
  "nullable",
  "enum",
  "maxItems",
  "minItems",
  "properties",
  "required",
  "minProperties",
  "maxProperties",
  "minLength",
  "maxLength",
  "pattern",
  "example",
  "anyOf",
  "propertyOrdering",
  "default",
  "items",
  "minimum",
  "maximum",
]);

/** Dropping these changes which values are valid: lossy, folded into the description. */
export const VALIDATION_KEYWORDS: ReadonlySet<string> = new Set([
  "exclusiveMinimum",
  "exclusiveMaximum",
  "multipleOf",
  "uniqueItems",
  "contains",
  "minContains",
  "maxContains",
  "propertyNames",
  "patternProperties",
  "dependentRequired",
  "dependentSchemas",
  "dependencies",
  "if",
  "then",
  "else",
  "not",
  "unevaluatedProperties",
  "unevaluatedItems",
  "additionalItems",
  "$dynamicRef",
  "$recursiveRef",
]);

const VALID_TYPES: ReadonlySet<string> = new Set([
  "string",
  "number",
  "integer",
  "boolean",
  "array",
  "object",
  "null",
]);

const TYPE_KEYWORDS: Record<string, readonly string[]> = {
  object: [
    "properties",
    "required",
    "additionalProperties",
    "minProperties",
    "maxProperties",
    "propertyOrdering",
    "patternProperties",
    "propertyNames",
    "dependentRequired",
    "dependentSchemas",
    "unevaluatedProperties",
  ],
  array: [
    "items",
    "prefixItems",
    "minItems",
    "maxItems",
    "uniqueItems",
    "contains",
    "minContains",
    "maxContains",
    "unevaluatedItems",
    "additionalItems",
  ],
  string: ["minLength", "maxLength", "pattern"],
  number: ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"],
  integer: ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"],
};
const NUMERIC_FORMATS: ReadonlySet<string> = new Set(["float", "double", "int32", "int64"]);

/** Gemini parameter names: letter or underscore first, then letters, digits, underscores; <= 64. */
export const PARAMETER_NAME = /^[A-Za-z_][A-Za-z0-9_]{0,63}$/;

function typeName(value: Json): string | null {
  return isString(value) ? value.toLowerCase() : null;
}

function enumMatches(value: Json, type: string): boolean {
  switch (type) {
    case "string":
      return typeof value === "string";
    case "boolean":
      return typeof value === "boolean";
    case "integer":
      return typeof value === "number" && Number.isInteger(value);
    case "number":
      return typeof value === "number";
    case "array":
      return Array.isArray(value);
    case "object":
      return isObject(value);
    default:
      return false;
  }
}

export class GeminiTransformer {
  private readonly report: Report;
  private readonly original: Json;
  private readonly cyclic: ReadonlySet<string>;

  constructor(schema: Json, report: Report) {
    this.report = report;
    this.original = trueToEmpty(deepCopy(schema), "", report, "gemini.boolean_schema");
    this.cyclic = cyclicRefs(this.original);
  }

  run(): Json {
    const out = this.visit(deepCopy(this.original), "", []);
    // Root definitions are only needed while a $ref remains (unrepairable recursion).
    if (isObject(out) && !this.report.issues.some((i) => i.code === "gemini.recursive_ref")) {
      for (const key of ["$defs", "definitions"]) {
        if (has(out, key)) {
          delete out[key];
          this.report.fix(
            "gemini.unsupported_keyword",
            child("", key),
            `"${key}" is not supported (definitions are inlined); removed`,
          );
        }
      }
    }
    return out;
  }

  /** Resolve a top-level `$ref` (merging its siblings); null if impossible. */
  private deref(schema: Json, stack: string[]): Json {
    if (!isObject(schema) || !has(schema, "$ref")) return schema;
    const ref = schema.$ref as Json;
    if (!isString(ref) || stack.includes(ref)) return null;
    let target: Json;
    try {
      target = resolveSchema(this.original, ref);
    } catch (error) {
      if (error instanceof RefError) return null;
      throw error;
    }
    if (!isObject(target)) return null;
    const merged = mergeAll(
      filterKeys(schema, (k) => k !== "$ref"),
      deepCopy(target),
    );
    if (merged === null) return null;
    return this.deref(merged, [...stack, ref]); // the target may itself be a reference
  }

  private splitTypes(node: JsonObject, types: string[]): JsonObject {
    const branches: JsonObject[] = [];
    const enumValues = pop(node, "enum");
    const fmt = pop(node, "format");
    const moved = new Set<string>();
    for (const type of types) {
      const branch: JsonObject = { type };
      for (const key of Object.prototype.hasOwnProperty.call(TYPE_KEYWORDS, type)
        ? (TYPE_KEYWORDS[type] as readonly string[])
        : []) {
        if (has(node, key)) {
          setKey(branch, key, deepCopy(node[key] as Json));
          moved.add(key);
        }
      }
      if (fmt !== null) {
        const numeric = type === "number" || type === "integer";
        if (
          (isString(fmt) && NUMERIC_FORMATS.has(fmt)) === numeric &&
          (type === "string" || type === "number" || type === "integer")
        ) {
          setKey(branch, "format", fmt);
        }
      }
      if (Array.isArray(enumValues)) {
        const values = enumValues.filter((v) => enumMatches(v, type));
        if (values.length === 0) continue; // no value of this type is allowed: drop the branch
        setKey(branch, "enum", values);
      }
      branches.push(branch);
    }
    for (const key of moved) delete node[key];
    delete node.type;
    setKey(node, "anyOf", branches);
    return node;
  }

  private visit(input: Json, path: string, stack: string[]): Json {
    const r = this.report;
    if (input === true) {
      r.fix("gemini.boolean_schema", path, "boolean schema `true` replaced with `{}`");
      return {};
    }
    if (!isObject(input)) {
      r.problem("schema.invalid", path, "schema must be a JSON object");
      return input;
    }
    if (!r.spend(path)) return input;
    let node = shallow(input);
    const keep = new Set<string>();

    // 1. References: Gemini needs definitions inlined (as the google-genai SDK does).
    if (has(node, "$ref")) {
      const ref = node.$ref as Json;
      if (!isString(ref)) {
        r.problem("schema.invalid", child(path, "$ref"), "$ref must be a string");
        return node;
      }
      if (stack.includes(ref)) {
        r.problem(
          "gemini.recursive_ref",
          path,
          `recursive ${ref} cannot be inlined, and Gemini's parameters schema has no $ref; use parametersJsonSchema or flatten the recursion`,
        );
        return node;
      }
      let target: Json;
      try {
        target = resolveSchema(this.original, ref);
      } catch (error) {
        if (!(error instanceof RefError)) throw error;
        r.problem("schema.unresolvable_ref", child(path, "$ref"), error.message);
        return node;
      }
      const siblings = filterKeys(node, (k) => k !== "$ref");
      const inlined = isObject(target) ? mergeAll(siblings, deepCopy(target)) : null;
      if (inlined === null) {
        r.problem(
          "gemini.ref_conflict",
          path,
          `cannot inline ${ref}: its keywords conflict with the sibling keywords`,
        );
        return node;
      }
      r.fix("gemini.ref_inlined", path, `inlined ${ref} (Gemini needs definitions inlined)`);
      return this.visit(inlined, path, [...stack, ref]);
    }

    // 2. allOf: merge into one schema.
    if (has(node, "allOf")) {
      const entries = pop(node, "allOf");
      let merged: JsonObject | null = node;
      if (Array.isArray(entries)) {
        for (const entry of entries) {
          // `true` accepts everything: neutral in an allOf.
          let resolved: Json = entry === true ? {} : this.deref(entry, stack);
          if (isObject(entry) && refIn(get(entry, "$ref"), this.cyclic)) resolved = null;
          if (merged === null || !isObject(resolved)) {
            merged = null;
            break;
          }
          merged = mergeAll(merged, resolved);
        }
      } else {
        merged = null;
      }
      if (merged === null) {
        setKey(node, "allOf", entries);
        r.problem(
          "gemini.allof_unmergeable",
          child(path, "allOf"),
          "allOf is not supported and its schemas conflict, so they cannot be merged",
        );
        return node; // leave the conflicting node as written
      }
      r.fix("gemini.allof_merged", child(path, "allOf"), "allOf merged into one schema");
      return this.visit(merged, path, stack);
    }

    // 3. oneOf -> anyOf.
    if (has(node, "oneOf")) {
      if (has(node, "anyOf")) {
        keep.add("oneOf");
        r.problem(
          "gemini.oneof_with_anyof",
          child(path, "oneOf"),
          "oneOf is not supported and cannot become anyOf because anyOf is already set",
        );
      } else {
        setKey(node, "anyOf", pop(node, "oneOf"));
        r.fix(
          "gemini.oneof_to_anyof",
          child(path, "oneOf"),
          "oneOf is not supported; replaced by anyOf (no longer exclusive)",
          { lossy: true },
        );
      }
    }

    // 4. const -> enum (the google-genai SDK does the same).
    if (has(node, "const")) {
      const constValue = pop(node, "const");
      const enumValues = get(node, "enum");
      setKey(
        node,
        "enum",
        Array.isArray(enumValues) ? enumValues.filter((v) => pyEqual(v, constValue)) : [constValue],
      );
      r.fix("gemini.const_to_enum", child(path, "const"), "const is not supported; became enum");
    }

    // 5. Types: no type arrays and no "null" type; Gemini uses `nullable`.
    const typeValue = get(node, "type");
    if (Array.isArray(typeValue)) {
      if (!typeValue.every(isString)) {
        r.problem("schema.invalid", child(path, "type"), "type array must contain strings");
      } else {
        const names = [...new Set(typeValue as string[])];
        const nonNull = names.filter((t) => t.toLowerCase() !== "null");
        if (nonNull.length > 1 && has(node, "anyOf")) {
          r.problem(
            "gemini.type_array",
            child(path, "type"),
            "type arrays are not supported and cannot become anyOf because anyOf is already set",
          );
        } else {
          if (nonNull.length < names.length) setKey(node, "nullable", true);
          if (nonNull.length === 0) delete node.type;
          else if (nonNull.length === 1) setKey(node, "type", nonNull[0] as string);
          else node = this.splitTypes(node, nonNull);
          r.fix(
            "gemini.type_array",
            child(path, "type"),
            'type arrays are not supported ("Proto field is not repeating"); rewritten with nullable/anyOf',
            // A null-only type becomes "untyped but nullable": looser.
            { lossy: nonNull.length === 0 },
          );
        }
      }
    } else if (typeName(typeValue) === "null") {
      delete node.type;
      setKey(node, "nullable", true);
      r.fix(
        "gemini.type_null",
        child(path, "type"),
        'type "null" became nullable: true (Gemini has no null-only type)',
        { lossy: true },
      );
    } else if (typeValue !== null && !VALID_TYPES.has(typeName(typeValue) ?? "")) {
      r.problem("schema.unknown_type", child(path, "type"), `unknown type ${pyRepr(typeValue)}`);
    }

    // 6. anyOf: a {"type": "null"} branch becomes nullable (as the SDK does).
    const anyOf = get(node, "anyOf");
    if (Array.isArray(anyOf) && anyOf.some(isNullSchema)) {
      const rest = anyOf.filter((b) => !isNullSchema(b));
      setKey(node, "nullable", true);
      r.fix("gemini.null_branch", child(path, "anyOf"), 'anyOf {"type": "null"} became nullable', {
        lossy: rest.length === 0, // anyOf of only null: now untyped but nullable
      });
      if (rest.length === 0) {
        delete node.anyOf;
      } else if (rest.length === 1) {
        const base = filterKeys(node, (k) => k !== "anyOf");
        const only = rest[0] as Json;
        let resolved: Json = this.deref(only, stack);
        if (isObject(only) && refIn(get(only, "$ref"), this.cyclic)) resolved = null;
        const collapsed = isObject(resolved) ? mergeAll(base, resolved) : null;
        if (collapsed !== null) return this.visit(collapsed, path, stack);
        setKey(node, "anyOf", rest);
      } else {
        setKey(node, "anyOf", rest);
      }
    }

    // 7. enum: non-empty, string values only.
    if (has(node, "enum")) {
      let enumValues = node.enum as Json;
      if (Array.isArray(enumValues) && enumValues.some((v) => v === null)) {
        enumValues = enumValues.filter((v) => v !== null);
        setKey(node, "enum", enumValues);
        setKey(node, "nullable", true);
        r.fix("gemini.enum_null", child(path, "enum"), "null in enum became nullable: true");
      }
      if (!Array.isArray(enumValues) || enumValues.length === 0) {
        r.problem("gemini.empty_enum", child(path, "enum"), "Gemini rejects empty enums");
      } else if (enumValues.some((v) => !isString(v))) {
        setKey(node, "enum", [
          ...new Set(enumValues.map((v) => (isString(v) ? v : jsonText(v)))),
        ]);
        r.fix(
          "gemini.enum_stringified",
          child(path, "enum"),
          'enum values must be strings ("only allowed for STRING type"); non-string values were converted, restore_arguments() converts them back',
          { lossy: true },
        );
      }
    }

    // 8. examples -> example.
    if (has(node, "examples") && !has(node, "example")) {
      const examples = node.examples as Json;
      if (Array.isArray(examples) && examples.length > 0) setKey(node, "example", examples[0] as Json);
    }

    // 9. additionalProperties is rejected by the Gemini Developer API.
    if (has(node, "additionalProperties")) {
      const extra = pop(node, "additionalProperties");
      const lossy = extra !== true && !pyEqual(extra, {});
      if (isObject(extra) && truthy(extra)) {
        foldIntoDescription(node, { additionalProperties: extra });
      }
      r.fix(
        "gemini.additional_properties",
        child(path, "additionalProperties"),
        "additionalProperties is not supported by the Gemini API; removed",
        { lossy },
      );
    }

    // 10. Tuple arrays (prefixItems / items as a list) -> items: anyOf.
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
        "gemini.tuple_items",
        path,
        "tuple arrays (prefixItems) are not supported; items became anyOf of the positions",
        { lossy: true },
      );
    }

    // 11. Remove every remaining keyword Gemini does not know.
    const dropped: JsonObject = {};
    for (const key of Object.keys(node)) {
      if (ALLOWED.has(key) || keep.has(key)) continue;
      if ((key === "$defs" || key === "definitions") && path === "") continue; // decided in run()
      const value = pop(node, key);
      if (VALIDATION_KEYWORDS.has(key)) {
        setKey(dropped, key, value);
        r.fix(
          "gemini.unsupported_keyword",
          child(path, key),
          `"${key}" is not supported (Invalid JSON payload: Unknown name); removed and added to the description`,
          { lossy: true },
        );
      } else {
        r.fix(
          "gemini.unsupported_keyword",
          child(path, key),
          `"${key}" is not supported (Invalid JSON payload: Unknown name); removed`,
        );
      }
    }
    foldIntoDescription(node, dropped);

    // 12. Recurse.
    const props = get(node, "properties");
    if (isObject(props)) {
      const mapped: JsonObject = {};
      for (const name of Object.keys(props)) {
        setKey(mapped, name, this.visit(props[name] as Json, child(path, "properties", name), stack));
      }
      setKey(node, "properties", mapped);
    } else if (props !== null) {
      r.problem("schema.invalid", child(path, "properties"), "properties must be an object");
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

    // 13. Structural requirements.
    const finalType = typeName(get(node, "type"));
    if (finalType === "object" && !truthy(get(node, "properties"))) {
      r.problem(
        "gemini.empty_object",
        path,
        path === ""
          ? 'OBJECT schemas need at least one property ("should be non-empty for OBJECT type"); for a tool without parameters, omit the parameters field (transform_tools does this)'
          : 'OBJECT schemas need at least one property ("should be non-empty for OBJECT type"); Gemini cannot express free-form objects or maps',
      );
    }
    if (finalType === "array" && !has(node, "items")) {
      r.problem("gemini.array_without_items", path, 'ARRAY schemas need "items" ("items: missing field")');
    }
    const finalProps = get(node, "properties");
    if (path === "" && isObject(finalProps)) {
      for (const name of Object.keys(finalProps)) {
        if (!PARAMETER_NAME.test(name)) {
          r.problem(
            "gemini.parameter_name",
            child(path, "properties", name),
            `parameter name ${pyRepr(name)} must match ${PARAMETER_NAME.source}`,
          );
        }
      }
    }
    return node;
  }
}
