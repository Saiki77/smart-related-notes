// Regression test for the ORT runtime pairs. Run: node bench/ort-runtime-check.mjs
//
// The two onnxruntime-web builds the plugin ships do NOT carry the same kernel
// set, and nothing in the type system or the build says so:
//   ORT_RUNTIMES.wasm   (ort-wasm-simd-threaded.*)          — full CPU kernel set,
//                        including com.microsoft GatherBlockQuantized.
//   ORT_RUNTIMES.webgpu (ort-wasm-simd-threaded.asyncify.*) — carries the WebGPU EP,
//                        but its CPU EP has NO GatherBlockQuantized kernel.
// jina-v5-nano block-quantizes its 128256x768 embed_tokens table, so all of its
// quantized exports need that kernel and load ONLY on the "wasm" pair. Pointing the
// CPU path at the asyncify pair (as every build up to 2.1.2 did) killed a jina
// reindex at session creation with:
//   Could not find an implementation for GatherBlockQuantized(1) node with name
//   '/model/embed_tokens/Gather_Quant'
// These checks pin that behaviour down so the pairing can't silently regress.
//
// ORT instantiates its wasm module ONCE per realm and latches it — a later
// wasmPaths change is silently ignored. So every build, and the end-to-end embed,
// runs in its own CHILD PROCESS (this file re-invokes itself); testing them in one
// process would quietly measure the first build four times.
//
// Model files are cached under ORT_TEST_CACHE (default ./.ort-test-cache). The jina
// pair is ~250 MB; the first run downloads, later runs are offline.
// SKIP_CDN=1 skips the ~72 MB CDN integrity fetch.
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(dirname(fileURLToPath(import.meta.url)));
const CACHE = process.env.ORT_TEST_CACHE || join(here, ".ort-test-cache");
const HF = "https://huggingface.co";
const SELF = fileURLToPath(import.meta.url);

// ORT_RUNTIMES lives in the generated src/ort-version.ts (TypeScript). Parse the
// literal out rather than adding a TS loader just for a check script.
function readGenerated(name) {
  const src = readFileSync(join(here, "src", "ort-version.ts"), "utf8");
  const obj = src.match(new RegExp(`export const ${name}[^=]*=\\s*(\\{[\\s\\S]*?\\n\\});`));
  if (obj) return JSON.parse(obj[1]);
  const str = src.match(new RegExp(`export const ${name} = ("[^"]*")`));
  if (str) return JSON.parse(str[1]);
  throw new Error(`could not find ${name} in src/ort-version.ts — run gen-ort.mjs`);
}

const RUNTIMES = readGenerated("ORT_RUNTIMES");

const MODELS = {
  jina: {
    repo: "jinaai/jina-embeddings-v5-text-nano-text-matching",
    file: "model_quantized.onnx",
    external: "model_quantized.onnx_data",
    // Only loadable on the "wasm" pair — this is the whole point of the split.
    builds: { wasm: true, webgpu: false },
  },
  minilm: { repo: "Xenova/paraphrase-multilingual-MiniLM-L12-v2", file: "model_quantized.onnx", builds: { wasm: true, webgpu: true } },
  mpnet: { repo: "Xenova/paraphrase-multilingual-mpnet-base-v2", file: "model_quantized.onnx", builds: { wasm: true, webgpu: true } },
  e5: { repo: "Xenova/multilingual-e5-small", file: "model_quantized.onnx", builds: { wasm: true, webgpu: true } },
};

async function download(url, dest) {
  if (existsSync(dest)) return dest;
  mkdirSync(dirname(dest), { recursive: true });
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} -> HTTP ${res.status}`);
  writeFileSync(dest, Buffer.from(await res.arrayBuffer()));
  return dest;
}
async function fetchModel(key) {
  const m = MODELS[key];
  const dir = join(CACHE, key);
  return {
    graph: await download(`${HF}/${m.repo}/resolve/main/onnx/${m.file}`, join(dir, m.file)),
    ext: m.external ? await download(`${HF}/${m.repo}/resolve/main/onnx/${m.external}`, join(dir, m.external)) : null,
  };
}

// ============================ child: session matrix ==========================
// One build per process, so ORT's latched wasm module is the one under test.
if (process.argv[2] === "--matrix") {
  const build = process.argv[3];
  const f = RUNTIMES[build];
  const ort = await import(pathToFileURL(join(here, "node_modules/onnxruntime-web/dist/ort.all.mjs")).href);
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  ort.env.logLevel = "error";
  ort.env.wasm.wasmPaths = {
    mjs: pathToFileURL(join(here, "ort", f.glue)).href,
    wasm: pathToFileURL(join(here, "ort", f.wasm)).href,
  };
  const out = {};
  for (const [key, m] of Object.entries(MODELS)) {
    const files = await fetchModel(key);
    try {
      const opts = { executionProviders: ["wasm"] };
      if (files.ext) opts.externalData = [{ path: m.external, data: readFileSync(files.ext) }];
      const s = await ort.InferenceSession.create(readFileSync(files.graph), opts);
      out[key] = { loaded: true, outputs: s.outputNames };
      await s.release?.();
    } catch (e) {
      out[key] = { loaded: false, err: String(e.message ?? e).split("\n")[0] };
    }
  }
  console.log("__RESULT__" + JSON.stringify(out));
  process.exit(0);
}

// ============================ child: end-to-end embed ========================
// Mirrors worker/embed-worker.ts: ort-web (not ort-node) via the ort-shim trick,
// wasmPaths.mjs + wasmBinary handed in, device "wasm", dtype "q8", last-token
// pooling. Proves the fixed pairing produces real, sane vectors — not just a
// session that happens to construct.
if (process.argv[2] === "--e2e") {
  const out = {};
  try {
    // ort-shim.ts equivalent: transformers picks node-vs-web exactly once, from
    // process.release.name, at backend-import time. The property is read-only on
    // current Node, so redefine it (the shim does the same).
    if (process.release?.name === "node") {
      try {
        process.release.name = "obsidian-renderer";
      } catch {
        Object.defineProperty(process.release, "name", {
          value: "obsidian-renderer",
          configurable: true,
          writable: true,
        });
      }
    }
    process.type = "renderer";

    const { pipeline, env } = await import("@huggingface/transformers");
    env.allowRemoteModels = true;
    env.allowLocalModels = false;
    const f = RUNTIMES.wasm;
    const onnx = env.backends?.onnx;
    // onnx.versions names the runtime transformers bound to: {common, web} for
    // onnxruntime-web, {common, node} for onnxruntime-node. The plugin MUST be on
    // web — the node build has different kernels and no wasmPaths at all, so a
    // node-backed run would prove nothing about what Obsidian executes.
    out.backend = onnx?.versions ?? null;
    out.webBackend = !!onnx?.versions?.web && !onnx?.versions?.node;
    for (const w of [onnx?.wasm, onnx?.env?.wasm]) {
      if (!w) continue;
      w.wasmPaths = { mjs: pathToFileURL(join(here, "ort", f.glue)).href };
      w.wasmBinary = readFileSync(join(here, "ort", f.wasm)).buffer;
      w.numThreads = 1;
      w.proxy = false;
    }

    const pipe = await pipeline("feature-extraction", MODELS.jina.repo, { device: "wasm", dtype: "q8" });
    // modelSpec("jina-embeddings-v5"): "Document: " prefix, LAST-TOKEN pooling,
    // single input (no batch padding), L2-normalised.
    const embed = async (t) => {
      const o = await pipe("Document: " + t, { pooling: "none" });
      const d = o.dims, data = o.data;
      const seq = d.length === 3 ? d[1] : d[0];
      const dim = d.length === 3 ? d[2] : d[1];
      const v = Array.from(data.subarray((seq - 1) * dim, seq * dim));
      const n = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
      o.dispose?.();
      return v.map((x) => x / n);
    };
    const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
    const de = await embed("Eine schlechtere Lösung wird trotzdem akzeptiert, mit Wahrscheinlichkeit exp(-ΔE/T), wobei T die aktuelle Temperatur ist.");
    const en = await embed("A worse candidate is still accepted with probability exp(-ΔE/T), where T is the current temperature.");
    const off = await embed("Der Feldspieler darf den Ball nicht mit der Hand spielen, sonst gibt es Freistoß.");
    out.dim = de.length;
    out.norm = Math.sqrt(dot(de, de));
    out.finite = de.every(Number.isFinite);
    out.twin = dot(de, en);
    out.unrelated = dot(de, off);
    out.ok = true;
  } catch (e) {
    out.ok = false;
    out.err = String(e.message ?? e).split("\n")[0];
  }
  console.log("__RESULT__" + JSON.stringify(out));
  process.exit(0);
}

// ================================ parent ====================================
let passed = 0;
const failures = [];
function check(name, ok, detail = "") {
  if (ok) {
    passed++;
    console.log(`  ok   ${name}`);
  } else {
    failures.push(`${name}${detail ? ` — ${detail}` : ""}`);
    console.log(`  FAIL ${name}${detail ? ` — ${detail}` : ""}`);
  }
}
function runChild(args) {
  const stdout = execFileSync(process.execPath, ["--max-old-space-size=8192", SELF, ...args], {
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
    stdio: ["ignore", "pipe", "inherit"],
  });
  const line = stdout.split("\n").find((l) => l.startsWith("__RESULT__"));
  if (!line) throw new Error("child produced no result");
  return JSON.parse(line.slice("__RESULT__".length));
}

console.log("\n[1] ort/ assets match the sizes pinned in src/ort-version.ts");
for (const [build, f] of Object.entries(RUNTIMES)) {
  for (const [kind, name, bytes] of [["glue", f.glue, f.glueBytes], ["wasm", f.wasm, f.wasmBytes]]) {
    const p = join(here, "ort", name);
    const actual = existsSync(p) ? statSync(p).size : -1;
    check(`${build}/${kind} ${name}`, actual === bytes, `${actual} != ${bytes}`);
  }
}
check(
  "the two builds are distinct files",
  RUNTIMES.wasm.wasm !== RUNTIMES.webgpu.wasm && RUNTIMES.wasm.glue !== RUNTIMES.webgpu.glue,
);

// A community-directory install has no ort/ folder and downloads the pair it needs;
// loadOrtAssets REJECTS anything whose DECOMPRESSED size differs from the pin. The
// CDNs serve these br/gzip-encoded, so the body must be read, not the header.
console.log("\n[2] pinned CDNs serve byte-identical assets");
if (process.env.SKIP_CDN) {
  console.log("  skip (SKIP_CDN set)");
} else {
  const CDNS = [
    readGenerated("ORT_WEB_CDN"),
    `https://unpkg.com/onnxruntime-web@${readGenerated("ORT_WEB_VERSION")}/dist/`,
  ];
  for (const base of CDNS) {
    for (const [build, f] of Object.entries(RUNTIMES)) {
      for (const [kind, name, bytes] of [["glue", f.glue, f.glueBytes], ["wasm", f.wasm, f.wasmBytes]]) {
        let len = -1;
        let err = "";
        try {
          const res = await fetch(base + name, { redirect: "follow" });
          len = res.ok ? (await res.arrayBuffer()).byteLength : -1;
          if (!res.ok) err = `HTTP ${res.status}`;
        } catch (e) {
          err = String(e.message ?? e);
        }
        check(`${new URL(base).host} ${build}/${kind} ${name}`, len === bytes, err || `${len} != ${bytes}`);
      }
    }
  }
}

console.log("\n[3] session creation matrix (the kernel-set difference)");
for (const build of Object.keys(RUNTIMES)) {
  let res;
  try {
    res = runChild(["--matrix", build]);
  } catch (e) {
    check(`${build} matrix`, false, String(e.message ?? e).split("\n")[0]);
    continue;
  }
  for (const [key, m] of Object.entries(MODELS)) {
    const want = m.builds[build];
    const got = res[key]?.loaded === true;
    check(
      `${build.padEnd(6)} + ${key.padEnd(6)} -> ${want ? "loads" : "rejected"}`,
      got === want,
      got ? "loaded but should not" : (res[key]?.err ?? "no result"),
    );
    // The rejection must be the KNOWN kernel gap, not some unrelated breakage.
    if (!want && !got) {
      check(
        `${build} + ${key} fails on GatherBlockQuantized`,
        (res[key]?.err ?? "").includes("GatherBlockQuantized"),
        res[key]?.err,
      );
    }
  }
}

console.log("\n[4] end-to-end embed via transformers.js on the 'wasm' pair");
let e2e;
try {
  e2e = runChild(["--e2e"]);
} catch (e) {
  e2e = { ok: false, err: String(e.message ?? e).split("\n")[0] };
}
check("transformers selected the WEB backend", e2e.webBackend === true, JSON.stringify(e2e.backend ?? {}));
check("pipeline ran", e2e.ok === true, e2e.err);
if (e2e.ok) {
  check("embedding has the expected width (768)", e2e.dim === 768, String(e2e.dim));
  check("vector is L2-normalised", Math.abs(e2e.norm - 1) < 1e-3, String(e2e.norm));
  check("vector is finite", e2e.finite === true);
  check(
    "cross-language twin outranks the unrelated note",
    e2e.twin > e2e.unrelated + 0.05,
    `twin ${e2e.twin?.toFixed(3)} vs unrelated ${e2e.unrelated?.toFixed(3)}`,
  );
  console.log(`       twin=${e2e.twin.toFixed(3)}  unrelated=${e2e.unrelated.toFixed(3)}`);
}

console.log("\n[5] main.js ships both runtime descriptors");
if (existsSync(join(here, "main.js"))) {
  const bundle = readFileSync(join(here, "main.js"), "utf8");
  for (const [build, f] of Object.entries(RUNTIMES)) {
    check(`${build} glue name in bundle`, bundle.includes(f.glue));
    check(`${build} wasm name in bundle`, bundle.includes(f.wasm));
    check(`${build} sizes in bundle`, bundle.includes(String(f.glueBytes)) && bundle.includes(String(f.wasmBytes)));
  }
} else {
  console.log("  skip main.js checks (not built)");
}

console.log(`\n${failures.length ? "FAILED" : "PASSED"}: ${passed} checks passed, ${failures.length} failed`);
for (const f of failures) console.log(`  - ${f}`);
process.exit(failures.length ? 1 : 0);
