/** Command-line interface: `schema-compat check` and `schema-compat fix`. */

import { readFileSync, writeFileSync } from "node:fs";
import { checkSchema, transformSchema } from "./api.js";
import { type Issue, TARGETS, isTarget } from "./model.js";
import { type Json, type JsonObject, get, isObject, isString, pyRepr } from "./py.js";
import { transformTools } from "./tools.js";
import { VERSION } from "./index.js";

const EXIT_OK = 0;
const EXIT_PROBLEMS = 1;
const EXIT_USAGE = 2;

class InputError extends Error {}
class UsageError extends Error {}

function load(path: string): Json {
  let text: string;
  try {
    text = path === "-" ? readFileSync(0, "utf8") : readFileSync(path, "utf8");
  } catch (error) {
    throw new InputError(`${path}: cannot read: ${(error as Error).message}`);
  }
  try {
    return JSON.parse(text) as Json;
  } catch (error) {
    throw new InputError(`${path}: invalid JSON: ${(error as Error).message}`);
  }
}

type Kind = "wrapped" | "list" | "tool" | "schema";

function kindOf(data: Json): Kind {
  if (isObject(data) && Array.isArray(get(data, "tools"))) return "wrapped";
  if (Array.isArray(data)) return "list";
  if (
    isObject(data) &&
    (isString(get(data, "name")) || (get(data, "type") === "function" && isObject(get(data, "function"))))
  ) {
    return "tool";
  }
  return "schema";
}

function toolsOf(data: Json, kind: Kind): Json[] {
  if (kind === "wrapped") return (data as JsonObject).tools as Json[];
  if (kind === "list") return data as Json[];
  return [data];
}

function issuesFor(data: Json, target: string): Issue[] {
  const kind = kindOf(data);
  if (kind === "schema") return checkSchema(data, target);
  const result = transformTools(toolsOf(data, kind), target);
  return [...result.changes, ...result.issues];
}

function ghEscape(text: string, prop = false): string {
  let out = text.replaceAll("%", "%25").replaceAll("\r", "%0D").replaceAll("\n", "%0A");
  if (prop) out = out.replaceAll(":", "%3A").replaceAll(",", "%2C");
  return out;
}

function describe(issue: Issue): string {
  const where = issue.tool ? `tool ${pyRepr(issue.tool)} ` : "";
  const flags = [issue.fixable ? "fixable" : "", issue.lossy ? "lossy" : ""].filter(Boolean);
  const suffix = flags.length > 0 ? ` (${flags.join(", ")})` : "";
  return `${where}${issue.path || "/"}: ${issue.code}${suffix}: ${issue.message}`;
}

interface Io {
  out: (text: string) => void;
  err: (text: string) => void;
}

function check(args: { files: string[]; targets: string[]; format: string; failOn: string }, io: Io): number {
  const targets = args.targets.length > 0 ? args.targets : [...TARGETS];
  const records: [string, string, Issue][] = [];
  for (const path of args.files) {
    const data = load(path);
    for (const target of targets) {
      for (const issue of issuesFor(data, target)) records.push([path, target, issue]);
    }
  }
  if (args.format === "json") {
    io.out(`${JSON.stringify(records.map(([file, target, issue]) => ({ file, target, ...issue })), null, 2)}\n`);
  } else if (args.format === "github") {
    for (const [file, target, issue] of records) {
      const level = issue.severity === "error" ? "error" : "warning";
      const title = ghEscape(`schema-compat ${issue.code}`, true);
      io.out(`::${level} file=${ghEscape(file, true)},title=${title}::${ghEscape(`[${target}] ${describe(issue)}`)}\n`);
    }
  } else {
    for (const [file, target, issue] of records) {
      io.out(`${file}: [${target}] ${issue.severity}: ${describe(issue)}\n`);
    }
  }
  const errors = records.filter(([, , i]) => i.severity === "error").length;
  const warnings = records.length - errors;
  const fixable = records.filter(([, , i]) => i.severity === "error" && i.fixable).length;
  io.err(
    `schema-compat: ${errors} error(s) (${fixable} fixable with \`schema-compat fix\`), ${warnings} warning(s) for ${targets.join(", ")}\n`,
  );
  const failing = errors + (args.failOn === "warning" ? warnings : 0);
  return failing > 0 ? EXIT_PROBLEMS : EXIT_OK;
}

function fix(
  args: { file: string; target: string; output: string | null; nameMap: string | null },
  io: Io,
): number {
  const data = load(args.file);
  const kind = kindOf(data);
  let fixed: Json;
  let changes: Issue[];
  let remaining: Issue[];
  let nameMap: Record<string, string> = {};
  if (kind === "schema") {
    const result = transformSchema(data, args.target);
    fixed = result.schema;
    changes = result.changes;
    remaining = result.issues;
  } else {
    const result = transformTools(toolsOf(data, kind), args.target);
    changes = result.changes;
    remaining = result.issues;
    nameMap = result.nameMap;
    if (kind === "wrapped") fixed = { ...(data as JsonObject), tools: result.tools };
    else if (kind === "list") fixed = result.tools;
    else fixed = result.tools[0] as Json;
  }
  const text = `${JSON.stringify(fixed, null, 2)}\n`;
  if (args.output) writeFileSync(args.output, text, "utf8");
  else io.out(text);
  if (args.nameMap) writeFileSync(args.nameMap, `${JSON.stringify(nameMap, null, 2)}\n`, "utf8");

  const lossy = changes.filter((c) => c.lossy).length;
  io.err(`schema-compat: applied ${changes.length} change(s) for ${args.target} (${lossy} lossy)\n`);
  const renamed = Object.keys(nameMap).length;
  if (renamed > 0) io.err(`schema-compat: renamed ${renamed} tool(s)\n`);
  for (const issue of remaining) io.err(`schema-compat: ${issue.severity}: ${describe(issue)}\n`);
  return remaining.some((i) => i.severity === "error") ? EXIT_PROBLEMS : EXIT_OK;
}

const USAGE = `usage: schema-compat [--version] {check,fix} ...

  schema-compat check FILE... [-t TARGET]... [--format text|json|github] [--fail-on error|warning]
  schema-compat fix FILE -t TARGET [-o OUTPUT] [--name-map FILE]

Targets: ${TARGETS.join(", ")}. FILE may be - for stdin.
`;

function takeValue(argv: string[], i: number, flag: string): string {
  const value = argv[i + 1];
  if (value === undefined) throw new UsageError(`${flag} needs a value`);
  return value;
}

function checkChoice(value: string, choices: readonly string[], flag: string): string {
  if (!choices.includes(value)) {
    throw new UsageError(`${flag}: invalid choice '${value}' (choose from ${choices.join(", ")})`);
  }
  return value;
}

export function main(argv: string[], io: Io = defaultIo()): number {
  try {
    if (argv.includes("--version")) {
      io.out(`schema-compat ${VERSION}\n`);
      return EXIT_OK;
    }
    if (argv.length === 0 || argv[0] === "-h" || argv[0] === "--help") {
      io.out(USAGE);
      return argv.length === 0 ? EXIT_USAGE : EXIT_OK;
    }
    const [command, ...rest] = argv;
    const positional: string[] = [];
    const targets: string[] = [];
    let format = "text";
    let failOn = "error";
    let output: string | null = null;
    let nameMap: string | null = null;
    for (let i = 0; i < rest.length; i++) {
      const arg = rest[i] as string;
      if (arg === "-t" || arg === "--target") {
        targets.push(checkChoice(takeValue(rest, i, arg), TARGETS, arg));
        i++;
      } else if (arg === "--format" && command === "check") {
        format = checkChoice(takeValue(rest, i, arg), ["text", "json", "github"], arg);
        i++;
      } else if (arg === "--fail-on" && command === "check") {
        failOn = checkChoice(takeValue(rest, i, arg), ["error", "warning"], arg);
        i++;
      } else if ((arg === "-o" || arg === "--output") && command === "fix") {
        output = takeValue(rest, i, arg);
        i++;
      } else if (arg === "--name-map" && command === "fix") {
        nameMap = takeValue(rest, i, arg);
        i++;
      } else if (arg.startsWith("-") && arg !== "-") {
        throw new UsageError(`unrecognized argument: ${arg}`);
      } else {
        positional.push(arg);
      }
    }
    if (command === "check") {
      if (positional.length === 0) throw new UsageError("check needs at least one FILE");
      return check({ files: positional, targets, format, failOn }, io);
    }
    if (command === "fix") {
      if (positional.length !== 1) throw new UsageError("fix needs exactly one FILE");
      if (targets.length !== 1 || !isTarget(targets[0] as string)) {
        throw new UsageError("fix needs exactly one -t TARGET");
      }
      return fix({ file: positional[0] as string, target: targets[0] as string, output, nameMap }, io);
    }
    throw new UsageError(`unknown command '${command}' (choose from check, fix)`);
  } catch (error) {
    if (error instanceof UsageError) {
      io.err(`${USAGE}\nschema-compat: error: ${error.message}\n`);
      return EXIT_USAGE;
    }
    if (error instanceof InputError) {
      io.err(`schema-compat: error: ${error.message}\n`);
      return EXIT_USAGE;
    }
    throw error;
  }
}

export function defaultIo(): Io {
  return { out: (t) => process.stdout.write(t), err: (t) => process.stderr.write(t) };
}
