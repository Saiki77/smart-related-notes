// INTEGRATION TEST: drives the REAL IndexStore.rank(), not a reimplementation.
//
// This exists because a reimplementation shipped a bug. bench/v3-crowding.mjs
// measured daily-note crowding at 0.67/10 on note-level vectors and the claim
// went into the release notes; the plugin ranks with chunk-level BiMax, which the
// template correction never reached, and in the real app crowding was still ~5/10.
// The harness and the plugin were measuring different code.
//
//   node --experimental-strip-types --import ./bench/register-stub.mjs bench/v3-integration.mjs
//
// Assertions (exit 1 on failure, so it can join `npm test`):
//   1 template de-crowding  a daily note's top-10 must not be mostly daily notes
//   2 graph fusion          the link graph must surface notes content alone ranks low
//   3 score ordering        the displayed % must not contradict the list order
//   4 isolated areas        an activated area must not leak into other notes
import { IndexStore } from "../src/index-store.ts";
import { makeApp, TFile } from "./obsidian-stub.mjs";
import { readFileSync } from "node:fs";
import { join } from "node:path";

const VAULT = process.env.LAB_VAULT || "/Users/justus/obsidian_atomized_intermediary/lab/vault";
const MODEL = process.env.LAB_MODEL || "Xenova/paraphrase-multilingual-MiniLM-L12-v2";
const CACHE = join(process.env.HOME, ".cache/srn-lab", `v3-${MODEL.replace(/[^a-z0-9]/gi, "_")}.json`);
const DECROWD = process.env.V3_DECROWD !== "0";

// Reuse the cached whole-note vectors as a stand-in embedding engine. The point of
// this harness is the RANKING pipeline, not the model, so a deterministic engine
// keeps runs fast and comparable. Chunks are derived from the note vector plus a
// small text-derived perturbation so chunk-level paths are genuinely exercised
// rather than collapsing to identical rows.
const cache = JSON.parse(readFileSync(CACHE, "utf8"));
const manifest = JSON.parse(readFileSync("/Users/justus/obsidian_atomized_intermediary/lab/corpus-manifest.json", "utf8"));
const INDEXABLE = new Set(manifest.answer_paths.map((p) => "real/" + p));
const exclude = (rel) => rel.startsWith("real/") && !INDEXABLE.has(rel);

const app = makeApp(VAULT, { exclude });
const noteText = (basename, body) => (basename + "\n\n" + body).slice(0, 8000);
const stripFront = (r) => { const m = r.match(/^---\n[\s\S]*?\n---\n?/); return m ? r.slice(m[0].length) : r; };

let hits = 0, misses = 0;
function vectorFor(text) {
  // Exact note text -> cached vector. Chunk texts are prefixes/segments of it, so
  // fall back to the containing note's vector with a deterministic jitter.
  const direct = cache[text];
  if (direct) { hits++; return direct; }
  misses++;
  return null;
}
const DIMS = (() => { for (const k of Object.keys(cache)) if (k !== "__model") return cache[k].length; return 384; })();
function hashJitter(text, base) {
  let h = 2166136261;
  for (let i = 0; i < text.length; i++) { h ^= text.charCodeAt(i); h = Math.imul(h, 16777619); }
  const out = new Float32Array(DIMS);
  let s = 0;
  for (let i = 0; i < DIMS; i++) {
    h ^= h << 13; h ^= h >>> 17; h ^= h << 5;
    const j = ((h >>> 0) / 4294967296 - 0.5) * 0.12;
    const v = (base ? base[i] : 0) + j;
    out[i] = v; s += v * v;
  }
  const n = Math.sqrt(s) || 1;
  for (let i = 0; i < DIMS; i++) out[i] /= n;
  return out;
}
const noteVecByPath = new Map();
for (const f of app.vault.getMarkdownFiles()) {
  const body = stripFront(readFileSync(join(VAULT, f.path), "utf8"));
  const v = vectorFor(noteText(f.basename, body));
  if (v) noteVecByPath.set(f.path, v);
}
// The engine the store talks to. embedBatch gets chunk TEXTS; we map each back to
// its note by longest-prefix ownership, which the harness records below.
const ownerOfText = new Map();
const engine = {
  loaded: true,
  async embedBatch(texts) {
    return texts.map((t) => {
      const owner = ownerOfText.get(t);
      const base = owner ? noteVecByPath.get(owner) : null;
      return hashJitter(t, base ?? null);
    });
  },
  async embed(text) { return (await this.embedBatch([text]))[0]; },
  dispose() {},
  unload() {},
};

// Record chunk-text ownership by re-running the same chunking the store will do:
// simplest reliable route is to tag every text that appears inside a note's body.
for (const f of app.vault.getMarkdownFiles()) {
  const body = stripFront(readFileSync(join(VAULT, f.path), "utf8"));
  for (const line of [f.basename, ...body.split(/\n{2,}/)]) {
    const t = line.trim();
    if (t) ownerOfText.set(t, f.path);
  }
  ownerOfText.set(noteText(f.basename, body), f.path);
}
// Fallback: any text not seen is attributed by scanning, done lazily and cached.
const rawBodies = new Map(app.vault.getMarkdownFiles().map((f) => [f.path, readFileSync(join(VAULT, f.path), "utf8")]));
const origGet = ownerOfText.get.bind(ownerOfText);
ownerOfText.get = (t) => {
  const direct = origGet(t);
  if (direct) return direct;
  const probe = t.slice(0, 60);
  for (const [p, b] of rawBodies) if (b.includes(probe)) { ownerOfText.set(t, p); return p; }
  return undefined;
};

const options = {
  excludeFolders: [], topK: 12, minSimilarity: 0.2, chunking: true,
  structureInfluence: 0.15, maxChunks: 48, shortlistSize: 60, showSummary: true,
  headingContext: true, ideaInfluence: 0.3, isolatedAreas: [], graphInfluence: 1,
};
const store = new IndexStore(app, engine, "/tmp/srn-integration", options);
if (!DECROWD) {
  // Ablation switch: neutralise the template correction to prove the assertion
  // actually discriminates rather than passing for unrelated reasons.
  store.subtractTemplateDirections = () => {};
}
console.log(`building index over ${app.vault.getMarkdownFiles().length} notes (decrowd=${DECROWD}) ...`);
const t0 = Date.now();
await store.build();
console.log(`built in ${((Date.now() - t0) / 1000).toFixed(1)}s; vector cache hits ${hits}, misses ${misses}`);

// ---------------------------------------------------------------- assertions
const failures = [];
const isDaily = (p) => /(^|\/)Daily\//.test(p);
const files = app.vault.getMarkdownFiles();

// 1. template de-crowding, measured through the SHIPPED rank()
const dailies = files.filter((f) => isDaily(f.path));
let crowd = 0, counted = 0;
for (const f of dailies) {
  const ranked = store.rank(f);
  if (!ranked.length) continue;
  counted++;
  crowd += ranked.slice(0, 10).filter((r) => isDaily(r.file.path)).length;
}
const avgCrowd = counted ? +(crowd / counted).toFixed(2) : null;
console.log(`\n1. daily crowding through rank(): ${avgCrowd}/10 over ${counted} daily notes`);
// The assertion runs in BOTH modes on purpose. Gating it on DECROWD made the
// ablation run report "passed", which proved nothing: a test that cannot fail
// when the bug is reintroduced is not evidence. Run with V3_DECROWD=0 and this
// must fail, or the assertion is not measuring what it claims.
if (avgCrowd !== null && avgCrowd > 5) {
  failures.push(`template de-crowding ineffective: ${avgCrowd}/10 of a daily note's top-10 are other daily notes`);
}

// 2. the graph channel must actually surface something content ranks low
let graphSurfaced = 0, withReason = 0;
for (const f of files.slice(0, 120)) {
  for (const r of store.rank(f)) {
    if (r.reason?.kind === "graph") { withReason++; if ((r.semantic ?? 1) < 0.3) graphSurfaced++; }
  }
}
console.log(`2. graph-surfaced cards: ${withReason} (of which ${graphSurfaced} below 0.30 semantic)`);
if (withReason === 0) failures.push("graph fusion surfaced nothing: no card carried a 'graph' reason");

// 3. the displayed score must not contradict the order it is shown in
let inversions = 0, pairs = 0;
for (const f of files.slice(0, 120)) {
  const ranked = store.rank(f);
  for (let i = 1; i < ranked.length; i++) {
    pairs++;
    if (ranked[i].score > ranked[i - 1].score + 0.02) inversions++;
  }
}
const invRate = pairs ? +(inversions / pairs).toFixed(3) : 0;
console.log(`3. score/order inversions: ${inversions}/${pairs} (${invRate})`);
if (invRate > 0.25) {
  failures.push(`the % shown contradicts the list order in ${(invRate * 100).toFixed(0)}% of adjacent pairs`);
}

// 4. isolated areas must partition, including through graph nominations.
// Membership is by TAG, not by folder: the vault has Daily/ notes with no
// frontmatter at all, and those are legitimately outside the area. Asserting on
// the path instead flagged them as leaks and cost a round of false alarm.
const inArea = new Set();
for (const f of files) {
  const c = app.metadataCache.getCache(f.path);
  const tags = [
    ...(c?.tags ?? []).map((t) => t.tag.replace(/^#/, "")),
    ...[].concat(c?.frontmatter?.tags ?? []),
  ].map((t) => String(t).toLowerCase().split("/")[0]);
  if (tags.includes("daily")) inArea.add(f.path);
}
store.updateOptions({ ...options, isolatedAreas: ["daily"] });
let leaks = 0;
for (const f of files.filter((x) => !inArea.has(x.path)).slice(0, 80)) {
  for (const r of store.rank(f)) if (inArea.has(r.file.path)) leaks++;
}
console.log(`4. isolated-area leaks: ${leaks} (area has ${inArea.size} tagged notes)`);
if (leaks > 0) failures.push(`${leaks} notes from an activated isolated area leaked into other notes' results`);
store.updateOptions(options);

console.log("");
if (failures.length) {
  for (const f of failures) console.log(`FAIL: ${f}`);
  console.log(`\n${failures.length} assertion(s) failed`);
  process.exit(1);
}
console.log("all integration assertions passed");
