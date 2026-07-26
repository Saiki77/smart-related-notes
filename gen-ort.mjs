// Generates src/ort-version.ts and copies the onnxruntime-web .wasm/.mjs assets
// out of the resolved node_modules install into ./ort, next to main.js.
//
// Run BEFORE tsc/eslint/esbuild so the generated import exists on a clean
// checkout (build/lint/dev all depend on src/ort-version.ts, which is gitignored
// because it is a build artifact pinned to the installed dependency version).
import { copyFileSync, existsSync, mkdirSync, readFileSync, rmSync, statSync, writeFileSync } from "fs";
import { dirname, join } from "path";
import { createRequire } from "module";
import { fileURLToPath, pathToFileURL } from "url";

const require = createRequire(import.meta.url);
const here = dirname(fileURLToPath(import.meta.url));

// Resolve the onnxruntime-web package ROOT. Its package.json "exports" map blocks
// require.resolve("onnxruntime-web/package.json"), so resolve a file that IS
// exported (the wasm entry) and walk up to the dir that holds package.json.
// Falls back to the conventional node_modules location if resolution moves.
function resolveOrtDir() {
  try {
    let dir = dirname(require.resolve("onnxruntime-web"));
    for (let i = 0; i < 6; i++) {
      if (existsSync(join(dir, "package.json")) && existsSync(join(dir, "dist"))) {
        return dir;
      }
      dir = dirname(dir);
    }
  } catch {
    // fall through to the conventional path
  }
  const fallback = join(here, "node_modules", "onnxruntime-web");
  if (existsSync(join(fallback, "package.json"))) return fallback;
  throw new Error("Could not locate the onnxruntime-web package directory.");
}

const ortDir = resolveOrtDir();
// This version tracks the onnxruntime-web that @huggingface/transformers actually
// bundles — which is frequently a -dev snapshot (e.g. 1.22.0-dev.2025...). Do NOT
// pin it to a stable release independently: the self-hosted app:// primary .wasm
// and the JS glue transformers bundled must be the exact same build or ORT init
// fails ("create of undefined" / wasm-glue mismatch). The CDN fallback URL below
// is derived from this same version, so it can't drift either.
const ortVersion = JSON.parse(
  readFileSync(join(ortDir, "package.json"), "utf8"),
).version;

// TWO runtime pairs, because they do NOT carry the same kernel set:
//   plain    — CPU only, but the FULL CPU kernel set, including the com.microsoft
//              GatherBlockQuantized kernel that a block-quantized embedding TABLE
//              needs. jina-v5-nano quantizes its 128256x768 embed_tokens table, so
//              every one of its quantized exports (model_quantized = dtype q8,
//              model_q4, model_q4f16) carries a GatherBlockQuantized node. On the
//              asyncify build those die at session creation with "Could not find an
//              implementation for GatherBlockQuantized(1) node with name
//              '/model/embed_tokens/Gather_Quant'". MiniLM/mpnet/e5 keep a plain
//              Gather in their q8 exports and load on either build.
//   asyncify — the only build carrying the WebGPU EP, but its CPU EP is missing
//              GatherBlockQuantized. Shipped for the explicit WebGPU device pin.
// main.ts picks the pair per spawn from the device (see ORT_RUNTIMES below).
// Neither glue needs the old v3 patchGlueForWeb step: both guard their
// Node/worker_threads branch on "real Node, not Electron".
const ORT_RUNTIME_BUILDS = {
  wasm: {
    glue: "ort-wasm-simd-threaded.mjs",
    wasm: "ort-wasm-simd-threaded.wasm",
  },
  webgpu: {
    glue: "ort-wasm-simd-threaded.asyncify.mjs",
    wasm: "ort-wasm-simd-threaded.asyncify.wasm",
  },
};
const ORT_WASM_FILES = Object.values(ORT_RUNTIME_BUILDS).flatMap((b) => [
  b.wasm,
  b.glue,
]);

export function genOrt() {
  // 1) Copy the matching wasm assets next to main.js. Clean first so a previous
  // build's (possibly differently-named) assets never linger in the folder.
  const outDir = join(here, "ort");
  rmSync(outDir, { recursive: true, force: true });
  mkdirSync(outDir, { recursive: true });
  for (const f of ORT_WASM_FILES) {
    const srcFile = join(ortDir, "dist", f);
    if (!existsSync(srcFile)) {
      throw new Error(
        `gen-ort: expected ORT asset missing: ${f} — the onnxruntime-web build ` +
          "layout changed; update ORT_WASM_FILES in gen-ort.mjs.",
      );
    }
    copyFileSync(srcFile, join(outDir, f));
  }

  // 2) Write the pinned version + CDN URL +, for EACH runtime pair, its file names
  // and EXACT BYTE SIZES. The plugin validates any local or freshly downloaded pair
  // against these sizes before serving it to the worker — a stateless integrity
  // check that catches torn cache writes, stale files from older installs after an
  // onnxruntime-web bump, and CDN/proxy garbage, all of which would otherwise fail
  // ORT init with cryptic mixed-build errors.
  const cdn = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ortVersion}/dist/`;
  const runtimes = Object.fromEntries(
    Object.entries(ORT_RUNTIME_BUILDS).map(([build, files]) => [
      build,
      {
        ...files,
        glueBytes: statSync(join(outDir, files.glue)).size,
        wasmBytes: statSync(join(outDir, files.wasm)).size,
      },
    ]),
  );
  const contents =
    "// AUTO-GENERATED by gen-ort.mjs — do not edit by hand.\n" +
    "// Pinned to the onnxruntime-web version @huggingface/transformers actually\n" +
    "// bundles, so the runtime download can never drift from the bundled glue.\n" +
    "// One entry per ORT build: 'wasm' is the CPU-only pair (full CPU kernel set,\n" +
    "// including GatherBlockQuantized); 'webgpu' is the asyncify pair, the only one\n" +
    "// carrying the WebGPU EP. See gen-ort.mjs for why both must ship.\n" +
    `export const ORT_WEB_VERSION = ${JSON.stringify(ortVersion)};\n` +
    `export const ORT_WEB_CDN = ${JSON.stringify(cdn)};\n` +
    "export interface OrtRuntimeFiles {\n" +
    "  glue: string;\n" +
    "  wasm: string;\n" +
    "  glueBytes: number;\n" +
    "  wasmBytes: number;\n" +
    "}\n" +
    `export const ORT_RUNTIMES: Record<"wasm" | "webgpu", OrtRuntimeFiles> = ${JSON.stringify(runtimes, null, 2)};\n`;
  const target = join(here, "src", "ort-version.ts");
  let current = "";
  try {
    current = readFileSync(target, "utf8");
  } catch {
    current = "";
  }
  if (current !== contents) writeFileSync(target, contents);

  return { ortVersion, cdn };
}

// Allow running directly: `node gen-ort.mjs`. Compare via pathToFileURL so the
// check survives paths containing spaces or other URL-encoded characters
// (import.meta.url is percent-encoded; a raw `file://${argv[1]}` template is not).
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const { ortVersion: v } = genOrt();
  console.log(`Generated ort-version.ts and copied onnxruntime-web@${v} -> ort/`);
}
