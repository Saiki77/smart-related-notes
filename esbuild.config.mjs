import esbuild from "esbuild";
import { builtinModules, createRequire } from "module";
import { genOrt } from "./gen-ort.mjs";

const require = createRequire(import.meta.url);
const prod = process.argv.includes("production");

// Native / Node-only modules transformers.js + onnxruntime reference. They have no
// browser build, so they MUST be external: esbuild then emits a require() that is
// simply never hit at runtime, because embeddings.ts forces the web/WASM backend.
// The transformers JS itself is NOT external — it is bundled into main.js.
const nodeExternals = [
  "onnxruntime-node", // native .node binding; the renderer uses onnxruntime-web
  "sharp", // native libvips image lib; unused for text embedding
  ...builtinModules,
  ...builtinModules.map((m) => `node:${m}`),
];

// Regenerate src/ort-version.ts and copy the matching onnxruntime-web .wasm/.mjs
// into ./ort. transformers bundles its own copy of the ORT JS glue into main.js;
// the runtime fetches the .wasm from wasmPaths and ORT requires the two to be the
// EXACT SAME build. Copying from the resolved node_modules install guarantees that
// and lets the plugin run offline. See gen-ort.mjs for detail.
const { ortVersion } = genOrt();
console.log(`onnxruntime-web@${ortVersion}: ort-version.ts written, ort/ populated.`);

const ctx = await esbuild.context({
  entryPoints: ["src/main.ts"],
  bundle: true,
  // platform:'browser' makes esbuild prefer the web/browser export conditions of
  // @huggingface/transformers, selecting onnxruntime-web over onnxruntime-node.
  // This is the single most important option for the Electron renderer.
  platform: "browser",
  external: [
    "obsidian",
    "electron",
    "@codemirror/*",
    "@lezer/*",
    ...nodeExternals,
  ],
  format: "cjs",
  target: "es2022",
  outfile: "main.js",
  sourcemap: prod ? false : "inline",
  treeShaking: true,
  minify: prod,
  logLevel: "info",
});

// `node esbuild.config.mjs` (dev) watches with inline sourcemaps;
// `node esbuild.config.mjs production` does a one-shot minified build.
if (prod) {
  await ctx.rebuild();
  await ctx.dispose();
} else {
  await ctx.watch();
  console.log("Watching for changes...");
}

// `require` is referenced so the createRequire import is not flagged unused by
// linters; node builtins above are resolved through esbuild, not this require.
void require;
