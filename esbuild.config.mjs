import esbuild from "esbuild";
import { builtinModules, createRequire } from "module";
import { genOrt } from "./gen-ort.mjs";

const require = createRequire(import.meta.url);
const prod = process.argv.includes("production");

// Native / Node-only modules transformers.js + onnxruntime reference. They have no
// browser build, so they MUST be external: esbuild then emits a require() that is
// simply never hit at runtime, because the embed worker forces the web/WASM
// backend. The transformers JS itself is NOT external — it is bundled into the
// WORKER bundle (and only there; the renderer bundle no longer evaluates it).
const nodeExternals = [
  "onnxruntime-node", // native .node binding; the worker uses onnxruntime-web
  "sharp", // native libvips image lib; unused for text embedding
  ...builtinModules,
  ...builtinModules.map((m) => `node:${m}`),
];

// Regenerate src/ort-version.ts and copy the matching onnxruntime-web .wasm/.mjs
// into ./ort. transformers bundles its own copy of the ORT JS glue into the
// worker bundle; the runtime imports the glue and .wasm the renderer fetches from
// ort/ (or the pinned CDN), and ORT requires the two to be the EXACT SAME build.
// Copying from the resolved node_modules install guarantees that and lets the
// plugin run offline. See gen-ort.mjs for detail.
const { ortVersion } = genOrt();
console.log(`onnxruntime-web@${ortVersion}: ort-version.ts written, ort/ populated.`);

// Bundle src/worker/embed-worker.ts (transformers.js + ort glue + our RPC shell)
// into a single ESM source string. Runs once per (re)build of the main bundle.
//
// format:"esm" is REQUIRED — the engine spawns it as a MODULE worker because ort
// dynamic-import()s its wasm glue, which classic workers disallow.
//
// Renderer-only modules (obsidian/electron/@codemirror) are deliberately NOT
// external here: if a refactor ever drags one into the worker's import graph,
// this nested build fails loudly instead of shipping a worker that dies at
// runtime with a bare unresolved require().
async function bundleEmbedWorker() {
  const result = await esbuild.build({
    entryPoints: ["src/worker/embed-worker.ts"],
    bundle: true,
    platform: "browser",
    format: "esm",
    target: "es2022",
    external: nodeExternals,
    write: false,
    minify: prod,
    metafile: true,
    logLevel: "silent",
  });
  return {
    source: result.outputFiles[0].text,
    // The worker's source files, so watch mode rebuilds main.js when they change.
    watchFiles: Object.keys(result.metafile.inputs),
  };
}

// Resolves the "virtual:embed-worker" import in src/embeddings.ts to the bundled
// worker source text (a default-exported string). tsc sees the ambient
// declaration in src/virtual-worker.d.ts instead.
//
// Watch-mode subtlety: esbuild replaces its watch set with each rebuild's
// inputs, and the worker graph's files only enter it via onLoad's watchFiles.
// If a FAILED nested build simply threw, no watchFiles would be registered and
// the watcher would go dead for worker files — the fix to a worker compile
// error would never trigger a rebuild. So failures are returned as `errors`
// (still failing the outer build loudly) WITH the last good watch set.
let lastWorkerWatchFiles = null;
const inlineWorkerPlugin = {
  name: "inline-embed-worker",
  setup(build) {
    build.onResolve({ filter: /^virtual:embed-worker$/ }, (args) => ({
      path: args.path,
      namespace: "embed-worker",
    }));
    build.onLoad({ filter: /.*/, namespace: "embed-worker" }, async () => {
      try {
        const { source, watchFiles } = await bundleEmbedWorker();
        lastWorkerWatchFiles = watchFiles;
        return {
          contents: `export default ${JSON.stringify(source)};`,
          loader: "js",
          watchFiles,
        };
      } catch (e) {
        return {
          errors: e.errors ?? [{ text: String(e) }],
          watchFiles: lastWorkerWatchFiles ?? ["src/worker/embed-worker.ts"],
        };
      }
    });
  },
};

const ctx = await esbuild.context({
  entryPoints: ["src/main.ts"],
  bundle: true,
  // platform:'browser' keeps web export conditions preferred throughout.
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
  plugins: [inlineWorkerPlugin],
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
