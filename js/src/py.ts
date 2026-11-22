/**
 * Python semantics the port must reproduce exactly, so that both packages give
 * identical results (checked by the parity tests): dict lookups, equality
 * (`True == 1`, `1 == 1.0`), truthiness, `json.dumps` text and `repr`.
 */

export type Json = null | boolean | number | string | Json[] | JsonObject;
export interface JsonObject {
  [key: string]: Json;
}

export function isObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function isString(value: unknown): value is string {
  return typeof value === "string";
}

export function isNumber(value: unknown): value is number {
  return typeof value === "number";
}

export function has(obj: JsonObject, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(obj, key);
}

/** `dict.get(key)`: a missing key and a JSON null both read as null. */
export function get(obj: JsonObject, key: string): Json {
  return has(obj, key) ? (obj[key] as Json) : null;
}

/** Assign without ever touching a prototype (keys come from untrusted schemas). */
export function setKey(obj: JsonObject, key: string, value: Json): void {
  if (key === "__proto__") {
    Object.defineProperty(obj, key, { value, enumerable: true, writable: true, configurable: true });
  } else {
    obj[key] = value;
  }
}

/** `dict.pop(key, None)`. */
export function pop(obj: JsonObject, key: string): Json {
  if (!has(obj, key)) return null;
  const value = obj[key] as Json;
  delete obj[key];
  return value;
}

/** `dict(obj)`: a shallow copy. */
export function shallow(obj: JsonObject): JsonObject {
  const out: JsonObject = {};
  for (const key of Object.keys(obj)) setKey(out, key, obj[key] as Json);
  return out;
}

/** `{k: v for k, v in obj.items() if keep(k)}`. */
export function filterKeys(obj: JsonObject, keep: (key: string) => boolean): JsonObject {
  const out: JsonObject = {};
  for (const key of Object.keys(obj)) if (keep(key)) setKey(out, key, obj[key] as Json);
  return out;
}

export function deepCopy<T extends Json>(value: T): T {
  if (Array.isArray(value)) return value.map((v) => deepCopy(v)) as T;
  if (isObject(value)) {
    const out: JsonObject = {};
    for (const key of Object.keys(value)) setKey(out, key, deepCopy(value[key] as Json));
    return out as T;
  }
  return value;
}

/** Python `==` on JSON values: booleans compare as 0/1 with numbers. */
export function pyEqual(a: Json, b: Json): boolean {
  const numeric = (v: Json) => typeof v === "number" || typeof v === "boolean";
  if (numeric(a) && numeric(b)) return Number(a) === Number(b);
  if (a === null || b === null) return a === b;
  if (typeof a === "string" || typeof b === "string") return a === b;
  if (Array.isArray(a) || Array.isArray(b)) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
    return a.every((v, i) => pyEqual(v, b[i] as Json));
  }
  if (isObject(a) && isObject(b)) {
    const keys = Object.keys(a);
    if (keys.length !== Object.keys(b).length) return false;
    return keys.every((k) => has(b, k) && pyEqual(a[k] as Json, b[k] as Json));
  }
  return false;
}

/** Python `x in list`. */
export function pyIn(value: Json, list: readonly Json[]): boolean {
  return list.some((item) => pyEqual(item, value));
}

/** Python truthiness of a JSON value. */
export function truthy(value: Json): boolean {
  if (value === null || value === false || value === 0 || value === "") return false;
  if (Array.isArray(value)) return value.length > 0;
  if (isObject(value)) return Object.keys(value).length > 0;
  return true;
}

function pyFloat(value: number): string {
  // Python repr: shortest round-trip digits (as JS), scientific outside [1e-4, 1e16).
  const [mantissa, exponentText] = value.toExponential().split("e") as [string, string];
  const exponent = Number(exponentText);
  if (exponent >= -4 && exponent < 16) {
    const text = String(value);
    return /[.e]/.test(text) ? text : `${text}.0`;
  }
  const sign = exponent < 0 ? "-" : "+";
  return `${mantissa}e${sign}${String(Math.abs(exponent)).padStart(2, "0")}`;
}

function pyNumber(value: number): string {
  if (Number.isInteger(value) && Math.abs(value) < 1e16) return String(value);
  if (!Number.isFinite(value)) return value > 0 ? "Infinity" : value < 0 ? "-Infinity" : "NaN";
  return pyFloat(value);
}

/** `json.dumps(value, ensure_ascii=False, separators=(", ", ": "))`. */
export function jsonText(value: Json): string {
  if (value === null) return "null";
  if (value === true) return "true";
  if (value === false) return "false";
  if (typeof value === "number") return pyNumber(value);
  if (typeof value === "string") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map((v) => jsonText(v)).join(", ")}]`;
  return `{${Object.keys(value)
    .map((k) => `${JSON.stringify(k)}: ${jsonText(value[k] as Json)}`)
    .join(", ")}}`;
}

function reprString(text: string): string {
  const quote = text.includes("'") && !text.includes('"') ? '"' : "'";
  let out = "";
  for (const ch of text) {
    const code = ch.codePointAt(0) as number;
    if (ch === "\\") out += "\\\\";
    else if (ch === quote) out += `\\${quote}`;
    else if (ch === "\n") out += "\\n";
    else if (ch === "\r") out += "\\r";
    else if (ch === "\t") out += "\\t";
    else if (code < 0x20 || code === 0x7f) out += `\\x${code.toString(16).padStart(2, "0")}`;
    else out += ch;
  }
  return quote + out + quote;
}

/** Python `repr()` for JSON values (used in messages only). */
export function pyRepr(value: Json): string {
  if (value === null) return "None";
  if (value === true) return "True";
  if (value === false) return "False";
  if (typeof value === "number") return pyNumber(value);
  if (typeof value === "string") return reprString(value);
  if (Array.isArray(value)) return `[${value.map((v) => pyRepr(v)).join(", ")}]`;
  return `{${Object.keys(value)
    .map((k) => `${reprString(k)}: ${pyRepr(value[k] as Json)}`)
    .join(", ")}}`;
}
