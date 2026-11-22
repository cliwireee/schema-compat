# schema-compat

**One tool schema, every provider.** Check and repair JSON Schemas for tool/function calling so
Anthropic, OpenAI (including strict mode) and Gemini all accept them — then map the model's
arguments back to what your code expects.

This is the TypeScript/JavaScript package. It has the same behaviour as the
[Python package](https://pypi.org/project/schema-compat/), verified by a shared parity corpus
of thousands of schemas (repaired schemas, every change and message, restored arguments).

- Zero runtime dependencies; ESM and CommonJS; types included
- Runs anywhere JavaScript runs (no Node-only APIs in the library)
- Untrusted input safe: never mutates input, no prototype pollution from `__proto__` keys

```bash
npm install schema-compat
```

## Usage

```ts
import { transformSchema, restoreArguments, transformTools } from "schema-compat";

const schema = {
  type: "object",
  properties: {
    query: { type: "string", minLength: 1 },
    limit: { type: ["integer", "null"], exclusiveMinimum: 0 },
    mode: { enum: [1, 2, 3] },
  },
  required: ["query"],
  additionalProperties: false,
};

const result = transformSchema(schema, "gemini");
// result.ok === true
// result.schema:
// { type: "object", required: ["query"], properties: {
//     query: { type: "string", minLength: 1 },
//     limit: { type: "integer", nullable: true, description: "{exclusiveMinimum: 0}" },
//     mode: { enum: ["1", "2", "3"] } } }
// result.changes: every repair, with a JSON Pointer, `lossy` and a message

// Gemini sends the enum value back as a string; restore the original integer:
restoreArguments({ query: "x", mode: "2" }, schema, "gemini"); // { query: "x", mode: 2 }

// Whole MCP tools/list results, OpenAI or Anthropic tool lists keep their shape:
const { tools, nameMap } = transformTools(mcpTools, "openai");
// nameMap: { fs_read_file: "fs.read_file" } — route tool calls back to the original names
```

Targets: `anthropic`, `anthropic-strict`, `openai`, `openai-strict`, `gemini`.
API: `checkSchema`, `transformSchema`, `checkTools`, `transformTools`, `restoreArguments`,
`nameIsValid`, `sanitizeName`, `TARGETS`.

## CLI

```bash
npx schema-compat check tools.json -t gemini -t openai-strict --format github
npx schema-compat fix tools.json -t gemini -o tools.gemini.json --name-map names.json
```

Rules, their sources (provider docs, the providers' own SDK code, real API errors), the meaning
of "lossy" and how everything is verified: see the
[main README](https://github.com/cliwireee/schema-compat#readme).

Known difference from the Python package: JavaScript cannot tell `1.0` from `1` in parsed JSON,
so a whole-number float in a dropped constraint is described as `1`, not `1.0`.

## License

MIT
