// TypeScript 7 has no compiler API for bundlers, so declarations come from tsc
// itself. The same .d.ts files are copied to a folder marked CommonJS so that
// `require` consumers get CommonJS-flavoured types (checked with arethetypeswrong).
import { execFileSync } from "node:child_process";
import { cpSync, rmSync, writeFileSync } from "node:fs";

rmSync("dist/types", { recursive: true, force: true });
rmSync("dist/types-cjs", { recursive: true, force: true });
execFileSync("npx", ["tsc", "-p", "tsconfig.build.json"], { stdio: "inherit" });
cpSync("dist/types", "dist/types-cjs", { recursive: true });
writeFileSync("dist/types-cjs/package.json", JSON.stringify({ type: "commonjs" }) + "\n");
