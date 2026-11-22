/**
 * schema-compat: make one tool/function JSON schema work with Anthropic, OpenAI
 * (including strict mode) and Gemini. A port of the Python package with identical
 * behaviour, verified by shared test cases and a parity corpus.
 */

export { checkSchema, checkTools, transformSchema } from "./api.js";
export {
  type Issue,
  type SchemaResult,
  type Severity,
  type Target,
  type ToolsResult,
  TARGETS,
} from "./model.js";
export type { Json, JsonObject } from "./py.js";
export { restoreArguments } from "./restore.js";
export { nameIsValid, sanitizeName, transformTools } from "./tools.js";

export const VERSION = "0.1.0";
