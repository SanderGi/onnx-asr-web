import { execFileSync } from "node:child_process";
import { mkdir, rm } from "node:fs/promises";
import { build } from "esbuild";

await rm("dist", { recursive: true, force: true });
await mkdir("dist", { recursive: true });

const shared = {
  bundle: true,
  format: "esm",
  external: ["onnxruntime-web"],
  sourcemap: true,
  target: ["es2020"],
};

await build({
  ...shared,
  platform: "neutral",
  entryPoints: ["src/index.ts"],
  outfile: "dist/index.js",
});

await build({
  ...shared,
  platform: "node",
  entryPoints: ["src/node.ts"],
  outfile: "dist/node.js",
});

await build({
  ...shared,
  platform: "browser",
  entryPoints: ["src/browser.ts"],
  outfile: "dist/browser.js",
});

execFileSync(process.execPath, ["./node_modules/typescript/bin/tsc", "-p", "tsconfig.json"], {
  stdio: "inherit",
});

execFileSync(process.execPath, ["scripts/generate-api-docs.mjs"], {
  stdio: "inherit",
});
