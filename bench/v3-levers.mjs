// V3 gating experiments — run on the SHIPPED default model unless LAB_MODEL says
// otherwise, because the decision is what to put in the plugin, not what scores
// best in a lab.
//
//   node bench/v3-levers.mjs                 # MiniLM (shipped default)
//   LAB_MODEL=jinaai/... node bench/v3-levers.mjs
//
// Answers three questions that gate the 3.0 build:
//   Q1  Do the untrained engine levers (hubness + graph fusion) work on MiniLM,
//       or were they a jina-only artefact?  -> decides the shipped model default
//   Q2  Does SAMPLED hubness correction (O(N*S)) retain the gain of EXACT
//       (O(N^2))?  -> decides whether 1.1 is shippable at vault scale
//   Q3  Does boilerplate dedup fix the daily-note crowding bug (measured 10/10)
//       without hurting link recall?  -> validates the Tier 0 fix
//
// Protocol: hold out 20% of real wikilinks (fixed seed), rank every note against
// every other, report recall@10 on held-out edges. Non-obvious slice = held-out
// edges whose target is NOT in the source's content top-10 (the reframe metric).
import { pipeline, env } from "@huggingface/transformers";
import { readFileSync, writeFileSync, readdirSync, statSync, existsSync, mkdirSync } from "node:fs";
import { join, relative, dirname } from "node:path";
import { fisherYates } from "./shuffle.mjs";

env.allowLocalModels = false;

const VAULT = process.env.LAB_VAULT || "/Users/justus/obsidian_atomized_intermediary/lab/vault";
const OUT = process.env.LAB_OUT || "/Users/justus/obsidian_atomized_intermediary/lab/results/v3-levers.json";
const MODEL = process.env.LAB_MODEL || "Xenova/paraphrase-multilingual-MiniLM-L12-v2";
const HOLDOUT = Number(process.env.V3_HOLDOUT ?? 0.2);
const SAMPLE = Number(process.env.V3_SAMPLE ?? 500); // sampled-hubness sample size
const KNN = Number(process.env.V3_KNN ?? 10);        // CSLS neighbourhood
const CACHE_DIR = process.env.HOME + "/.cache/srn-lab";
const CACHE = join(CACHE_DIR, `v3-${MODEL.replace(/[^a-z0-9]/gi, "_")}.json`);

const fold = (s) => s.toLowerCase().normalize("NFC")
  .replace(/ä/g, "ae").replace(/ö/g, "oe").replace(/ü/g, "ue").replace(/ß/g, "ss")
  .replace(/\s+/g, " ").trim();
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const l2 = (v) => { let s = 0; for (const x of v) s += x * x; s = Math.sqrt(s) || 1; return v.map((x) => x / s); };
function mulberry32(a) { return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

function walk(dir) {
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
const stripFront = (raw) => { const m = raw.match(/^---\n[\s\S]*?\n---\n?/); return m ? raw.slice(m[0].length) : raw; };

// ---------- corpus ----------
const manifest = JSON.parse(readFileSync("/Users/justus/obsidian_atomized_intermediary/lab/corpus-manifest.json", "utf8"));
const INDEXABLE = new Set(manifest.answer_paths.map((p) => "real/" + p));
const HUB_RE = /\bMOC\b|Uebersicht|Zettelkasten Index|Vault Insights|^Untitled$/i;

const notes = [];
for (const abs of walk(VAULT)) {
  const rel = relative(VAULT, abs);
  if (rel.startsWith("real/") && !INDEXABLE.has(rel)) continue;
  const basename = rel.replace(/\.md$/, "").split("/").pop();
  if (/(^|\/)Attachments\//.test(rel) || /\.dup$/.test(basename)) continue;
  const body = stripFront(readFileSync(abs, "utf8"));
  notes.push({ rel, basename, body, isHub: HUB_RE.test(basename), isDaily: /(^|\/)Daily\//.test(rel) });
}

// ---------- Q3: boilerplate dedup (gate G1) ----------
// A line that appears verbatim in >= LINE_DF_MAX notes is template furniture, not
// content. Drop it from the embed input.
const LINE_DF_MAX = 3;
const lineDf = new Map();
for (const n of notes) {
  const seen = new Set();
  for (const line of n.body.split("\n")) { const f = fold(line); if (f.length >= 6) seen.add(f); }
  for (const f of seen) lineDf.set(f, (lineDf.get(f) ?? 0) + 1);
}
function noteText(n, dedup) {
  let body = n.body;
  if (dedup) {
    body = body.split("\n").filter((line) => {
      const f = fold(line);
      return f.length < 6 || (lineDf.get(f) ?? 0) < LINE_DF_MAX;
    }).join("\n");
    if (!body.trim()) body = n.body; // never empty a note out
  }
  return (n.basename + "\n\n" + body).slice(0, 8000);
}

// ---------- embed ----------
let cache = {};
try { cache = JSON.parse(readFileSync(CACHE, "utf8")); if (cache.__model !== MODEL) cache = {}; } catch { }
const wanted = [];
for (const n of notes) { wanted.push(noteText(n, false), noteText(n, true)); }
const missing = [...new Set(wanted.filter((t) => !cache[t]))];
console.log(`${notes.length} notes; ${missing.length} texts to embed (${wanted.length - missing.length} cached) with ${MODEL}`);
if (missing.length) {
  const extractor = await pipeline("feature-extraction", MODEL, { dtype: "fp32" });
  const isWholeNote = /jina-embeddings-v5/i.test(MODEL);
  const PREFIX = isWholeNote ? "Document: " : "";
  for (let i = 0; i < missing.length; i += isWholeNote ? 1 : 16) {
    const batch = missing.slice(i, i + (isWholeNote ? 1 : 16));
    if (isWholeNote) {
      const o = await extractor(PREFIX + batch[0], { pooling: "none" });
      const d = o.dims, data = o.data;
      const seq = d.length === 3 ? d[1] : d[0], dim = d.length === 3 ? d[2] : d[1];
      cache[batch[0]] = l2(Array.from(data.subarray((seq - 1) * dim, seq * dim))).map((x) => +x.toFixed(5));
      o.dispose?.();
    } else {
      const t = await extractor(batch, { pooling: "mean", normalize: true });
      t.tolist().forEach((v, j) => { cache[batch[j]] = v.map((x) => +x.toFixed(5)); });
    }
    if ((i + 1) % 200 < (isWholeNote ? 1 : 16)) {
      console.log(`  ${i + 1}/${missing.length}`);
      mkdirSync(CACHE_DIR, { recursive: true });
      writeFileSync(CACHE, JSON.stringify({ ...cache, __model: MODEL }));
    }
  }
  cache.__model = MODEL;
  mkdirSync(CACHE_DIR, { recursive: true });
  writeFileSync(CACHE, JSON.stringify(cache));
}

// ---------- vectors, two variants ----------
function buildSpace(dedup) {
  const V = notes.map((n) => cache[noteText(n, dedup)]).filter(Boolean);
  const D = V[0].length;
  const mean = new Array(D).fill(0);
  for (const v of V) for (let i = 0; i < D; i++) mean[i] += v[i];
  for (let i = 0; i < D; i++) mean[i] /= V.length;
  return V.map((v) => l2(v.map((x, i) => x - mean[i]))); // centered, as the plugin does
}
const Craw = buildSpace(false);
const Cdedup = buildSpace(true);
const N = notes.length;
const idx = new Map(notes.map((n, i) => [n.rel, i]));
const byFold = new Map(notes.map((n) => [fold(n.basename), n]));

// ---------- link graph + holdout ----------
const allEdges = [];
const adjFull = Array.from({ length: N }, () => new Set());
for (const n of notes) {
  const s = idx.get(n.rel);
  for (const m of n.body.matchAll(/\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]/g)) {
    const name = m[1].trim();
    if (!name) continue; // the literal [[]] case
    const t = byFold.get(fold(name));
    if (!t || t.rel === n.rel) continue;
    const ti = idx.get(t.rel);
    adjFull[s].add(ti); adjFull[ti].add(s);
    allEdges.push([s, ti]);
  }
}
const uniqEdges = [...new Set(allEdges.map(([a, b]) => `${Math.min(a, b)}-${Math.max(a, b)}`))]
  .map((k) => k.split("-").map(Number));
const rnd = mulberry32(20260727);
const shuffled = fisherYates(uniqEdges, rnd);
const nHold = Math.floor(shuffled.length * HOLDOUT);
const heldOut = shuffled.slice(0, nHold);
const trainEdges = shuffled.slice(nHold);
const adj = Array.from({ length: N }, () => new Set());
for (const [a, b] of trainEdges) { adj[a].add(b); adj[b].add(a); }
console.log(`graph: ${uniqEdges.length} unique edges, ${trainEdges.length} train / ${heldOut.length} held out`);

// ---------- channels ----------
const contentSim = (C, s, t) => dot(C[s], C[t]);

// exact hubness radius: mean similarity to own k nearest
function radiiExact(C) {
  const r = new Array(N).fill(0);
  for (let s = 0; s < N; s++) {
    const sims = [];
    for (let t = 0; t < N; t++) if (t !== s) sims.push(dot(C[s], C[t]));
    sims.sort((a, b) => b - a);
    let acc = 0;
    for (let i = 0; i < KNN && i < sims.length; i++) acc += sims[i];
    r[s] = acc / Math.min(KNN, sims.length);
  }
  return r;
}
// sampled: mean over a fixed random sample, scaled to the same k-NN quantile
function radiiSampled(C, sampleSize) {
  const rs = mulberry32(7);
  const sample = [];
  const pool = [...Array(N).keys()];
  for (let i = 0; i < Math.min(sampleSize, N); i++) sample.push(pool.splice(Math.floor(rs() * pool.length), 1)[0]);
  const r = new Array(N).fill(0);
  const frac = KNN / N; // take the same top fraction inside the sample
  const kIn = Math.max(1, Math.round(frac * sample.length));
  for (let s = 0; s < N; s++) {
    const sims = [];
    for (const t of sample) if (t !== s) sims.push(dot(C[s], C[t]));
    sims.sort((a, b) => b - a);
    let acc = 0;
    for (let i = 0; i < kIn && i < sims.length; i++) acc += sims[i];
    r[s] = acc / Math.min(kIn, sims.length);
  }
  return r;
}
// Resource Allocation over shared neighbours
const RA = (s, t) => { let a = 0; for (const x of adj[s]) if (adj[t].has(x)) a += 1 / Math.max(1, adj[x].size); return a; };
// rooted PageRank: truncated random walk with restart from s
function pprFrom(s, alpha = 0.15, steps = 3) {
  let dist = new Map([[s, 1]]);
  const acc = new Map();
  for (let step = 0; step < steps; step++) {
    const next = new Map();
    for (const [node, mass] of dist) {
      const nbrs = adj[node];
      if (!nbrs.size) continue;
      const share = (mass * (1 - alpha)) / nbrs.size;
      for (const nb of nbrs) next.set(nb, (next.get(nb) ?? 0) + share);
    }
    for (const [node, mass] of next) acc.set(node, (acc.get(node) ?? 0) + mass);
    dist = next;
  }
  return acc;
}

// ---------- rank + evaluate ----------
const zstats = (arr) => { const m = arr.reduce((a, b) => a + b, 0) / arr.length; const sd = Math.sqrt(arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length) || 1; return [m, sd]; };

function evaluate(label, C, { csls = null, useRA = false, usePPR = false } = {}) {
  // per-source ranking over all candidate targets
  const bySource = new Map();
  for (const [a, b] of heldOut) {
    if (!bySource.has(a)) bySource.set(a, []); bySource.get(a).push(b);
    if (!bySource.has(b)) bySource.set(b, []); bySource.get(b).push(a);
  }
  let hits = 0, total = 0, nonObvHits = 0, nonObvTotal = 0;
  const top1Counts = new Array(N).fill(0);
  for (const [s, targets] of bySource) {
    const cand = [];
    for (let t = 0; t < N; t++) {
      if (t === s || adj[s].has(t)) continue; // training links are known, not predictions
      cand.push(t);
    }
    if (!cand.length) continue;
    const cSims = cand.map((t) => csls ? 2 * contentSim(C, s, t) - csls[s] - csls[t] : contentSim(C, s, t));
    // content-only ranking (for the non-obvious definition)
    const contentRank = cand.map((t, i) => [contentSim(C, s, t), t]).sort((a, b) => b[0] - a[0]);
    const contentTop10 = new Set(contentRank.slice(0, 10).map((x) => x[1]));
    let score = cSims.slice();
    if (useRA || usePPR) {
      const [cm, csd] = zstats(cSims);
      const zc = cSims.map((x) => (x - cm) / csd);
      let struct = new Array(cand.length).fill(0);
      if (useRA) { const ra = cand.map((t) => RA(s, t)); const [m, sd] = zstats(ra); struct = struct.map((x, i) => x + (ra[i] - m) / sd); }
      if (usePPR) { const p = pprFrom(s); const pv = cand.map((t) => p.get(t) ?? 0); const [m, sd] = zstats(pv); struct = struct.map((x, i) => x + (pv[i] - m) / sd); }
      score = zc.map((x, i) => x + struct[i]);
    }
    const ranked = cand.map((t, i) => [score[i], t]).sort((a, b) => b[0] - a[0]);
    const top10 = new Set(ranked.slice(0, 10).map((x) => x[1]));
    if (ranked.length) top1Counts[ranked[0][1]]++;
    for (const t of new Set(targets)) {
      if (adj[s].has(t)) continue;
      total++;
      const isNonObvious = !contentTop10.has(t);
      if (isNonObvious) nonObvTotal++;
      if (top10.has(t)) { hits++; if (isNonObvious) nonObvHits++; }
    }
  }
  // hubness: skewness of the top-1 occurrence distribution
  const [m, sd] = zstats(top1Counts);
  const skew = top1Counts.reduce((a, x) => a + ((x - m) / sd) ** 3, 0) / N;
  return {
    config: label,
    recall_at_10: +(hits / total).toFixed(4),
    nonObvious_recall_at_10: nonObvTotal ? +(nonObvHits / nonObvTotal).toFixed(4) : null,
    nonObvious_share: +(nonObvTotal / total).toFixed(3),
    hubness_skew: +skew.toFixed(3),
    evaluated: total,
  };
}

console.log("computing hubness radii ...");
const rExact = radiiExact(Craw);
const rSampled = radiiSampled(Craw, SAMPLE);
const rSampled200 = radiiSampled(Craw, 200);
const corr = (() => {
  const [ma, sa] = zstats(rExact), [mb, sb] = zstats(rSampled);
  return +(rExact.reduce((acc, x, i) => acc + ((x - ma) / sa) * ((rSampled[i] - mb) / sb), 0) / N).toFixed(4);
})();

const results = [
  evaluate("content (baseline)", Craw),
  evaluate("content + CSLS exact", Craw, { csls: rExact }),
  evaluate(`content + CSLS sampled(${SAMPLE})`, Craw, { csls: rSampled }),
  evaluate("content + CSLS sampled(200)", Craw, { csls: rSampled200 }),
  evaluate("content + RA", Craw, { useRA: true }),
  evaluate("content + RA + PPR", Craw, { useRA: true, usePPR: true }),
  evaluate("content + CSLS(sampled) + RA + PPR  [3.0 candidate]", Craw, { csls: rSampled, useRA: true, usePPR: true }),
  evaluate("dedup content (baseline)", Cdedup),
  evaluate("dedup + CSLS(sampled) + RA + PPR", Cdedup, { csls: rSampled, useRA: true, usePPR: true }),
];

// ---------- Q3: daily-note crowding, both spaces ----------
function dailyCrowding(C) {
  const daily = notes.map((n, i) => [n, i]).filter(([n]) => n.isDaily);
  if (!daily.length) return null;
  let acc = 0;
  for (const [, s] of daily) {
    const ranked = [];
    for (let t = 0; t < N; t++) if (t !== s) ranked.push([dot(C[s], C[t]), t]);
    ranked.sort((a, b) => b[0] - a[0]);
    acc += ranked.slice(0, 10).filter(([, t]) => notes[t].isDaily).length;
  }
  return +(acc / daily.length).toFixed(2);
}
const crowdRaw = dailyCrowding(Craw);
const crowdDedup = dailyCrowding(Cdedup);

const summary = {
  model: MODEL, notes: N, edges: uniqEdges.length, heldOut: heldOut.length,
  sampleSize: SAMPLE, knn: KNN,
  radius_correlation_exact_vs_sampled: corr,
  daily_crowding_top10: { raw: crowdRaw, dedup: crowdDedup, note: "avg # of other daily notes in a daily note's top-10 (10 = the bug)" },
  results,
};
console.log("\n==== V3 LEVERS ====");
console.log(`model ${MODEL} | ${N} notes | ${uniqEdges.length} edges (${heldOut.length} held out)`);
console.log(`radius corr exact vs sampled(${SAMPLE}): ${corr}`);
console.log(`daily crowding: raw ${crowdRaw}/10 -> dedup ${crowdDedup}/10`);
console.log("");
for (const r of results) {
  console.log(`  ${r.config.padEnd(48)} R@10 ${r.recall_at_10.toFixed(4)}  nonObv ${String(r.nonObvious_recall_at_10).padEnd(6)}  skew ${String(r.hubness_skew).padEnd(7)}`);
}
mkdirSync(dirname(OUT), { recursive: true });
writeFileSync(OUT, JSON.stringify(summary, null, 1));
console.log("\nwrote", OUT);
