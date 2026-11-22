/** Rules shared by several targets. */

import { type Json, type JsonObject, get, has, isObject, pyIn, pyRepr, setKey, shallow } from "./py.js";
import { type Report, isNullSchema } from "./util.js";

/** The root of a tool's parameters must be `{"type": "object", ...}`. */
export function fixRootObject(node: Json, report: Report, family: string): Json {
  if (!isObject(node)) {
    report.problem(`${family}.root_not_object`, "", "the parameters schema must be an object");
    return node;
  }
  const typeValue = get(node, "type");
  if (typeValue === null) {
    report.fix(`${family}.root_type`, "", 'the root schema needs type: "object"; added');
    const out: JsonObject = { type: "object" };
    for (const key of Object.keys(node)) if (key !== "type") setKey(out, key, node[key] as Json);
    return out;
  }
  if (typeValue === "object") return node;
  if (Array.isArray(typeValue) && typeValue.includes("object")) {
    report.fix(
      `${family}.root_type`,
      "/type",
      `the root schema must be type: "object", not ${pyRepr(typeValue)}; narrowed`,
    );
    const out = shallow(node);
    setKey(out, "type", "object");
    return out;
  }
  report.problem(
    `${family}.root_not_object`,
    "/type",
    `the root schema must be type: "object", got ${pyRepr(typeValue)}`,
  );
  return node;
}

const CONSTRAINING = ["type", "enum", "const", "anyOf", "oneOf", "allOf", "$ref", "not"];

/** True when `schema` visibly accepts null (conservative). */
export function allowsNull(schema: Json): boolean {
  if (!isObject(schema)) return schema === true;
  const typeValue = get(schema, "type");
  if (typeValue === "null" || (Array.isArray(typeValue) && typeValue.includes("null"))) return true;
  if (get(schema, "nullable") === true) return true;
  const anyOf = get(schema, "anyOf");
  if (Array.isArray(anyOf) && anyOf.some(isNullSchema)) return true;
  const enumValues = get(schema, "enum");
  if (Array.isArray(enumValues) && enumValues.some((v) => v === null)) return true;
  if (has(schema, "const") && schema.const === null) return true;
  return !CONSTRAINING.some((k) => has(schema, k));
}

const WRAP_KEYS = ["$ref", "const", "allOf", "oneOf", "not"];

/** `schema` widened to also accept null (OpenAI's pattern for optional fields). */
export function makeNullable(schema: Json): Json {
  if (!isObject(schema) || allowsNull(schema)) return schema;
  if (WRAP_KEYS.some((k) => has(schema, k))) return { anyOf: [schema, { type: "null" }] };
  const out = shallow(schema);
  const typeValue = get(out, "type");
  if (typeof typeValue === "string") setKey(out, "type", [typeValue, "null"]);
  else if (Array.isArray(typeValue)) setKey(out, "type", [...typeValue, "null"]);
  const anyOf = get(out, "anyOf");
  if (Array.isArray(anyOf)) setKey(out, "anyOf", [...anyOf, { type: "null" }]);
  const enumValues = get(out, "enum");
  if (Array.isArray(enumValues)) setKey(out, "enum", [...enumValues, null]);
  return out;
}

export { pyIn };
