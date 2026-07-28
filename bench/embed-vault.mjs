// Fill the v3 whole-note embedding cache for every note in the lab vault, including
// the 526 added on 2026-07-28. Keys are exactly what the studies look up:
//   basename + "\n\n" + body-without-frontmatter, truncated to 8000 chars.
// Model spec mirrors src/embeddings.ts: "Document: " prefix, last-token pooling,
// one input at a time so no batch padding shifts the last token.
//
//   node bench/embed-vault.mjs
import { pipeline, env } from "@huggingface/transformers";
import { readFileSync, writeFileSync, readdirSync, statSync, mkdirSync } from "node:fs";
import { join } from "node:path";

env.allowLocalModels = false;
const VAULT = process.env.LAB_VAULT || "/Users/justus/obsidian_atomized_intermediary/lab/vault";
const MODEL = process.env.LAB_MODEL || "jinaai/jina-embeddings-v5-text-nano-text-matching";
const DIR = join(process.env.HOME, ".cache/srn-lab");
const CACHE = join(DIR, `v3-${MODEL.replace(/[^a-z0-9]/gi, "_")}.json`);
const PREFIX = MODEL.includes("jina") ? "Document: " : "";

function walk(d, a = []) {
  for (const n of readdirSync(d)) {
    if (n.startsWith(".")) continue;
    const p = join(d, n);
    if (statSync(p).isDirectory()) walk(p, a); else if (n.endsWith(".md")) a.push(p);
  }
  return a;
}
const strip = (r) => { const m = r.match(/^---\n[\s\S]*?\n---\n?/); return m ? r.slice(m[0].length) : r; };
const files = walk(VAULT);
const texts = files.map((f) => {
  const base = f.slice(f.lastIndexOf("/") + 1, -3);
  return (base + "\n\n" + strip(readFileSync(f, "utf8"))).slice(0, 8000);
});

mkdirSync(DIR, { recursive: true });
let cache = {};
try { cache = JSON.parse(readFileSync(CACHE, "utf8")); } catch { /* first run */ }
const missing = [...new Set(texts)].filter((t) => !cache[t]);
console.log(`${files.length} notes | ${missing.length} to embed | model ${MODEL}`);
if (!missing.length) { console.log("nothing to do"); process.exit(0); }

const extractor = await pipeline("feature-extraction", MODEL, { dtype: "fp32" });
let done = 0;
for (const t of missing) {
  const out = await extractor(PREFIX + t, { pooling: "none", normalize: false });
  // last-token pooling, then L2 normalise, matching the plugin
  const [, seq, dim] = out.dims;
  const d = out.data, off = (seq - 1) * dim;
  const v = new Float32Array(dim);
  let s = 0;
  for (let i = 0; i < dim; i++) { v[i] = d[off + i]; s += v[i] * v[i]; }
  const n = Math.sqrt(s) || 1;
  cache[t] = Array.from(v, (x) => +(x / n).toFixed(5));
  if (++done % 25 === 0) { console.log(`  ${done}/${missing.length}`); writeFileSync(CACHE, JSON.stringify(cache)); }
}
cache.__model = MODEL;
writeFileSync(CACHE, JSON.stringify(cache));
console.log(`done: ${done} embedded, cache now ${Object.keys(cache).length} entries -> ${CACHE}`);
