// Regression test for GLUE_PRELUDE. Run: node bench/glue-prelude-check.mjs
//
// The ORT glue decides Node-vs-browser with (both shipped builds, two different
// minified spellings each):
//   globalThis.process?.versions?.node && globalThis.process?.type != "renderer"
// and its Node branch require()s worker_threads, which cannot resolve in a blob
// module worker. Obsidian's worker realms DO expose a Node-like `process`, so in
// the pthread workers the threaded runtime spawns — realms ort-shim.ts never
// touches — that branch won every time and killed threaded bring-up with
//   Uncaught TypeError: Failed to resolve module specifier 'worker_threads'
// worker/embed-worker.ts prepends GLUE_PRELUDE to the glue TEXT so the guard is
// satisfied wherever the glue is evaluated.
//
// This extracts the REAL condition expressions out of the REAL shipped glue files
// and evaluates them against a simulated Obsidian worker realm, with and without
// the prelude. It fails if a build stops being fixed by the prelude, or if the
// upstream condition changes shape so the extractor no longer finds it.
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { runInNewContext } from "node:vm";

const here = dirname(dirname(fileURLToPath(import.meta.url)));

// Both layers are read OUT OF the worker source, so the test can never drift
// from what actually ships.
const WORKER_SRC = readFileSync(join(here, "src", "worker", "embed-worker.ts"), "utf8");
const NODE_CHECK = JSON.parse(WORKER_SRC.match(/const GLUE_NODE_CHECK = ("[^"]*");/)?.[1] ?? '""');
const PRELUDE = WORKER_SRC
  .match(/const GLUE_PRELUDE = \[([\s\S]*?)\]\.join\("\\n"\);/)?.[1]
  ?.split("\n")
  .map((l) => l.trim().replace(/^"/, "").replace(/",?$/, ""))
  .filter((l) => l.length)
  .join("\n");

let passed = 0;
const failures = [];
const check = (name, ok, detail = "") => {
  if (ok) { passed++; console.log(`  ok   ${name}`); }
  else { failures.push(`${name}${detail ? ` — ${detail}` : ""}`); console.log(`  FAIL ${name}${detail ? ` — ${detail}` : ""}`); }
};

check("GLUE_PRELUDE extracted from embed-worker.ts", !!PRELUDE && PRELUDE.includes("renderer"), PRELUDE ?? "not found");
check("GLUE_NODE_CHECK extracted from embed-worker.ts", !!NODE_CHECK, NODE_CHECK || "not found");
if (!PRELUDE || !NODE_CHECK) { console.log("\nFAILED"); process.exit(1); }

// Every way the two builds spell the check. Both must be neutralised: the glue
// evaluates one in the module body and one inside the runtime factory.
const PATTERNS = [
  /globalThis\.process\?\.versions\?\.node&&"renderer"!=globalThis\.process\?\.type/g,
  /globalThis\.process\?\.versions\?\.node&&globalThis\.process\?\.type!="renderer"/g,
];

for (const file of ["ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.asyncify.mjs"]) {
  console.log(`\n[${file}]`);
  const glue = readFileSync(join(here, "ort", file), "utf8");
  const found = PATTERNS.flatMap((p) => glue.match(p) ?? []);
  check("node-vs-browser checks located in the shipped glue", found.length >= 2, `found ${found.length}`);

  // Layer 1: the token transformers.js itself replaces. If onnxruntime-web ever
  // remints it, the replaceAll silently stops matching — fail loudly here instead.
  const hits = glue.split(NODE_CHECK).length - 1;
  check("GLUE_NODE_CHECK token still present in the glue", hits >= 2, `${hits} occurrences`);
  const replaced = glue.replaceAll(NODE_CHECK, "false");
  check(
    "replacing it neutralises every node-vs-browser check",
    PATTERNS.every((p) => (replaced.match(p) ?? []).length === 0),
    "a check survived the replacement",
  );
  for (const [i, expr] of found.entries()) {
    check(
      `check ${i + 1}: evaluates false once the token is replaced`,
      runInNewContext(expr.replaceAll(NODE_CHECK, "false"), { process: { versions: { node: "20.18.0" }, type: undefined } }) === false,
    );
  }

  // Layer 2: the prelude, which holds even if the token above drifts.
  for (const [i, expr] of found.entries()) {
    // A worker realm the way Obsidian exposes it: Node-like process, and NOT
    // process.type === "renderer" (that is the renderer-only value ort-shim sets).
    // (the context object IS the realm's globalThis, so `globalThis.process` resolves here)
    const realm = () => ({ process: { versions: { node: "20.18.0" }, type: undefined } });

    const before = runInNewContext(expr, realm());
    check(`check ${i + 1}: takes the Node branch unpatched (reproduces the bug)`, before === true, String(before));

    const ctx = realm();
    runInNewContext(PRELUDE + "\n;" + expr, ctx);
    const after = runInNewContext(PRELUDE + "\n;" + expr, ctx);
    check(`check ${i + 1}: takes the browser branch after the prelude`, after === false, String(after));
    check(`check ${i + 1}: prelude set process.type`, ctx.process.type === "renderer", String(ctx.process.type));
  }

  // A real browser worker has no `process` at all — the prelude must be inert
  // there, not throw and take the glue down with it.
  const browser = { process: undefined };
  let threw = null;
  try { runInNewContext(PRELUDE, browser); } catch (e) { threw = String(e.message ?? e); }
  check("prelude is a no-op in a realm with no process", threw === null, threw ?? "");

  // A realm that froze process must not break glue evaluation either.
  const frozen = { process: Object.freeze({ versions: { node: "20.18.0" }, type: "browser" }) };
  let threw2 = null;
  try { runInNewContext(PRELUDE, frozen); } catch (e) { threw2 = String(e.message ?? e); }
  check("prelude survives a frozen process", threw2 === null, threw2 ?? "");
}

console.log(`\n${failures.length ? "FAILED" : "PASSED"}: ${passed} checks passed, ${failures.length} failed`);
for (const f of failures) console.log(`  - ${f}`);
process.exit(failures.length ? 1 : 0);
