import { defineConfig } from "tsup";

export default defineConfig([
  {
    entry: { index: "src/index.ts" },
    format: ["esm", "cjs"],
    dts: false,
    clean: true,
    target: "es2022",
    sourcemap: false,
  },
  {
    entry: { bin: "src/bin.ts" },
    format: ["esm"],
    target: "node18",
    banner: { js: "#!/usr/bin/env node" },
  },
]);
