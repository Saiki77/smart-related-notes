// Release gate: every ORT runtime pair gen-ort.mjs pinned must exist in ort/ at
// exactly the pinned size and SHA-256 digest. Run after `npm run build` (which
// runs gen-ort.mjs).
//
// Both pairs matter, for different reasons:
//   wasm   — the CPU-only build, and the ONLY one carrying a com.microsoft
//            GatherBlockQuantized kernel. jina-v5-nano block-quantizes its
//            embed_tokens table, so without this pair it cannot load at all.
//   webgpu — the asyncify build, the only one carrying the WebGPU EP.
// A missing or wrong-sized file means onnxruntime-web changed its dist layout,
// or gen-ort.mjs drifted from it. Either way the plugin's size validation would
// reject the download at runtime, so fail the release here instead.
import { createHash } from "node:crypto";
import { existsSync, readFileSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(dirname(fileURLToPath(import.meta.url)));
const src = readFileSync(join(here, "src", "ort-version.ts"), "utf8");
const match = src.match(/export const ORT_RUNTIMES[^=]*=\s*(\{[\s\S]*?\n\});/);
if (!match) {
  console.error("ORT_RUNTIMES missing from src/ort-version.ts — run gen-ort.mjs");
  process.exit(1);
}

let bad = 0;
for (const [build, files] of Object.entries(JSON.parse(match[1]))) {
  for (const [name, pinned, pinnedSha] of [
    [files.glue, files.glueBytes, files.glueSha256],
    [files.wasm, files.wasmBytes, files.wasmSha256],
  ]) {
    const path = join("ort", name);
    const full = join(here, path);
    const size = existsSync(full) ? statSync(full).size : -1;
    const sha = size >= 0 ? createHash("sha256").update(readFileSync(full)).digest("hex") : "";
    if (size === pinned && sha === pinnedSha) {
      console.log(`${build}: ${path} ${size} B, sha256 ${sha.slice(0, 12)} ok`);
    } else {
      console.error(`${build}: ${path} is ${size} B sha256 ${sha.slice(0, 12)}, pinned ${pinned} B sha256 ${String(pinnedSha).slice(0, 12)}`);
      bad++;
    }
  }
}

const version = src.match(/ORT_WEB_VERSION = "([^"]+)"/)?.[1] ?? "unknown";
console.log(`Runtime pinned to onnxruntime-web@${version}.`);
process.exit(bad ? 1 : 0);
