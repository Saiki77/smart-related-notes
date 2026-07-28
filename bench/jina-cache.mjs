// Build the shared jina-v5-nano embedding cache for the lab vault:
// one whole-note vector + paragraph vectors per note, replicating the plugin's
// model spec (Document: prefix, LAST-TOKEN pooling, single-input — no batch
// padding; see related-notes/src/embeddings.ts modelSpec()).
//
//   node bench/jina-cache.mjs
//
// Cache: JINA_CACHE (default scratchpad jina-cache.json), keyed by the exact
// prefixed input text; values are float arrays rounded to 5 decimals.
import { pipeline, env } from "@huggingface/transformers";
import { readFileSync, writeFileSync, readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";

env.allowLocalModels = false;

const VAULT = process.env.LAB_VAULT || "/Users/justus/obsidian_atomized_intermediary/lab/vault";
const MODEL = "jinaai/jina-embeddings-v5-text-nano-text-matching";
const CACHE = process.env.JINA_CACHE || (process.env.HOME + "/.cache/srn-lab/jina-cache.json");
const PREFIX = "Document: ";

export function walk(dir) {
  const out = [];
  for (const name of readdirSync(dir)) {
    if (name.startsWith(".")) continue;
    const p = join(dir, name);
    const s = statSync(p);
    if (s.isDirectory()) out.push(...walk(p));
    else if (name.endsWith(".md")) out.push(p);
  }
  return out;
}
export function stripFront(raw) {
  const m = raw.match(/^---\n[\s\S]*?\n---\n?/);
  return m ? raw.slice(m[0].length) : raw;
}
// Paragraph = blank-line block; code fences and pure-heading/table blocks dropped;
// list runs count as one block (many notes are list-heavy). Wikilinks -> display text.
export function paragraphsOf(body) {
  const noCode = body.replace(/```[\s\S]*?```/g, "\n").replace(/%%[\s\S]*?%%/g, "\n");
  const blocks = noCode.split(/\n\s*\n/);
  const out = [];
  for (const b of blocks) {
    const lines = b.split("\n").filter((l) => !/^\s*#{1,6}\s/.test(l) && !/^\s*\|/.test(l))
      .map((l) => l.replace(/^\s*>+\s*/, "").replace(/\[![A-Za-z]+\][+-]?\s*/g, ""));
    let text = lines.join(" ")
      .replace(/\[\[([^\]|]+)\|([^\]]+)\]\]/g, "$2").replace(/\[\[([^\]]+)\]\]/g, "$1")
      .replace(/^[\s>*+-]+/, "").replace(/[*_~`]/g, "").replace(/\s+/g, " ").trim();
    const words = text.split(/\s+/).filter(Boolean);
    if (words.length >= 25) out.push(text.slice(0, 3500));
  }
  return out;
}
export function noteText(basename, body) {
  return (basename + "\n\n" + stripFront(body)).slice(0, 24000);
}

const isMain = process.argv[1] && process.argv[1].endsWith("jina-cache.mjs");
if (isMain) {
  const files = walk(VAULT);
  const manifest = JSON.parse(readFileSync("/Users/justus/obsidian_atomized_intermediary/lab/corpus-manifest.json", "utf8"));
  const INDEXABLE = new Set(manifest.answer_paths.map((p) => "real/" + p));
  const texts = new Set();
  let notes = 0, paras = 0;
  for (const abs of files) {
    const rel = relative(VAULT, abs);
    if (rel.startsWith("real/") && !INDEXABLE.has(rel)) continue;
    const basename = rel.replace(/\.md$/, "").split("/").pop();
    if (/\bMOC\b/i.test(basename) || /(^|\/)Attachments\//.test(rel)) continue;
    const body = stripFront(readFileSync(abs, "utf8"));
    texts.add(PREFIX + noteText(basename, body)); notes++;
    for (const p of paragraphsOf(body)) { texts.add(PREFIX + p); paras++; }
  }
  console.log(`${notes} notes, ${paras} paragraphs -> ${texts.size} unique inputs`);
  let cache = {};
  try { cache = JSON.parse(readFileSync(CACHE, "utf8")); if (cache.__model !== MODEL) cache = {}; } catch { }
  const missing = [...texts].filter((t) => !cache[t]);
  console.log(`${missing.length} to embed (${texts.size - missing.length} cached)`);
  if (missing.length) {
    const pipe = await pipeline("feature-extraction", MODEL, { dtype: "fp32" });
    const t0 = Date.now();
    for (let i = 0; i < missing.length; i++) {
      const o = await pipe(missing[i], { pooling: "none" });
      const d = o.dims, data = o.data;
      const seq = d.length === 3 ? d[1] : d[0], dim = d.length === 3 ? d[2] : d[1];
      let v = Array.from(data.subarray((seq - 1) * dim, seq * dim));
      const n = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
      cache[missing[i]] = v.map((x) => +(x / n).toFixed(5));
      o.dispose?.();
      if ((i + 1) % 400 === 0) {
        console.log(`  ${i + 1}/${missing.length} (${((Date.now() - t0) / (i + 1)).toFixed(0)} ms/embed)`);
        writeFileSync(CACHE, JSON.stringify({ ...cache, __model: MODEL }));
      }
    }
    cache.__model = MODEL;
    writeFileSync(CACHE, JSON.stringify(cache));
    console.log(`done in ${((Date.now() - t0) / 1000).toFixed(0)}s; cache ${CACHE}`);
  }
}
