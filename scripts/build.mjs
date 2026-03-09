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
  entryPoints: ["src/index.js"],
  outfile: "dist/index.js",
});

await build({
  ...shared,
  platform: "node",
  entryPoints: ["src/node.js"],
  outfile: "dist/node.js",
});

await build({
  ...shared,
  platform: "browser",
  entryPoints: ["src/browser.js"],
  outfile: "dist/browser.js",
});
