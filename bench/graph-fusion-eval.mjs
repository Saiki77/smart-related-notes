// Graph x content fusion: does the wikilink graph add link-prediction signal on
// top of pure jina semantics? Held-out-edge study (see lab/RESEARCH-embedding-papers.md).
// Tests the precondition for the whole graph-embedding branch (MUSAE/LINE/DistMult).
//   node bench/graph-fusion-eval.mjs
// Methods on 20% held-out edges: pure-jina cosine; Common Neighbors; Adamic-Adar;
// DistMult (trained on surviving edges); fusion(jina + AA); fusion(jina + DistMult).
import { readFileSync, writeFileSync } from "node:fs";
import { relative } from "node:path";
import { walk, stripFront, noteText } from "./jina-cache.mjs";
import { fisherYates } from "./shuffle.mjs";

const VAULT = "/Users/justus/obsidian_atomized_intermediary/lab/vault";
const CACHE_PATH = process.env.JINA_CACHE || (process.env.HOME + "/.cache/srn-lab/jina-cache.json");
const PREFIX = "Document: ";
const fold = (s) => s.toLowerCase().normalize("NFC").replace(/ä/g, "ae").replace(/ö/g, "oe").replace(/ü/g, "ue").replace(/ß/g, "ss").replace(/\s+/g, " ").trim();
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const l2 = (v) => { let s = 0; for (const x of v) s += x * x; s = Math.sqrt(s) || 1; return v.map((x) => x / s); };
function mulberry32(a) { return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

const manifest = JSON.parse(readFileSync("/Users/justus/obsidian_atomized_intermediary/lab/corpus-manifest.json", "utf8"));
const INDEXABLE = new Set(manifest.answer_paths.map((p) => "real/" + p));
const cache = JSON.parse(readFileSync(CACHE_PATH, "utf8"));
const vec = (t) => cache[PREFIX + t] || null;

const notes = [];
for (const abs of walk(VAULT)) {
  const rel = relative(VAULT, abs);
  if (rel.startsWith("real/") && !INDEXABLE.has(rel)) continue;
  const basename = rel.replace(/\.md$/, "").split("/").pop();
  if (/\bMOC\b/i.test(basename) || /(^|\/)Attachments\//.test(rel) || /\.dup$/.test(basename)) continue;
  const raw = readFileSync(abs, "utf8");
  const body = stripFront(raw);
  notes.push({ rel, basename, body, v: vec(noteText(basename, body)) });
}
const withV = notes.filter((n) => n.v);
const idx = new Map(withV.map((n, i) => [n, i]));
const byFold = new Map(withV.map((n) => [fold(n.basename), n]));
// centered semantic vectors
const D = withV[0].v.length, mean = new Array(D).fill(0);
for (const n of withV) for (let i = 0; i < D; i++) mean[i] += n.v[i];
for (let i = 0; i < D; i++) mean[i] /= withV.length;
for (const n of withV) n.c = l2(n.v.map((x, i) => x - mean[i]));

// undirected wikilink edges among indexed notes (exclude MOC/hub targets implicitly)
const edgeSet = new Set();
const edges = [];
for (const n of withV) {
  for (const m of n.body.matchAll(/\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]/g)) {
    const t = byFold.get(fold(m[1]));
    if (!t || t === n) continue;
    const a = Math.min(idx.get(n), idx.get(t)), b = Math.max(idx.get(n), idx.get(t));
    const key = a + "-" + b;
    if (!edgeSet.has(key)) { edgeSet.add(key); edges.push([a, b]); }
  }
}
const N = withV.length;
console.log(`graph: ${N} notes, ${edges.length} undirected edges, avg degree ${(2 * edges.length / N).toFixed(2)}`);

// 20% held-out split (fixed seed)
const rnd = mulberry32(42);
const shuffled = fisherYates(edges, rnd);
const nTest = Math.floor(edges.length * 0.2);
const test = shuffled.slice(0, nTest), train = shuffled.slice(nTest);
const adj = Array.from({ length: N }, () => new Set());
for (const [a, b] of train) { adj[a].add(b); adj[b].add(a); }
const trainSet = new Set(train.map(([a, b]) => a + "-" + b));
console.log(`held-out edges: ${test.length}; train edges: ${train.length}`);

// candidate ranking for a source node s: all other nodes not already train-linked to s
function rankBy(sIdx, scoreFn) {
  const out = [];
  for (let t = 0; t < N; t++) {
    if (t === sIdx) continue;
    if (adj[sIdx].has(t)) continue; // known train edge — not a prediction
    out.push([scoreFn(sIdx, t), t]);
  }
  out.sort((a, b) => b[0] - a[0]);
  return out;
}
const jinaScore = (s, t) => dot(withV[s].c, withV[t].c);
const commonNeighbors = (s, t) => { let c = 0; for (const x of adj[s]) if (adj[t].has(x)) c++; return c; };
const adamicAdar = (s, t) => { let a = 0; for (const x of adj[s]) if (adj[t].has(x)) a += 1 / Math.log(1 + adj[x].size); return a; };

// DistMult on surviving edges (single relation): score = <e_s, e_t> (diagonal R = 1)
// => symmetric bilinear; trained by margin ranking with negative sampling.
function trainDistMult(dim = 32, epochs = 40, lr = 0.05) {
  const E = Array.from({ length: N }, () => new Array(dim).fill(0).map(() => (rnd() - 0.5) * 0.2));
  const score = (s, t) => { let x = 0; for (let i = 0; i < dim; i++) x += E[s][i] * E[t][i]; return x; };
  for (let ep = 0; ep < epochs; ep++) {
    for (const [a, b] of train) {
      for (let neg = 0; neg < 2; neg++) {
        const corrupt = Math.floor(rnd() * N);
        if (adj[a].has(corrupt) || corrupt === a) continue;
        const pos = score(a, b), negs = score(a, corrupt);
        const margin = 1 - (pos - negs);
        if (margin <= 0) continue;
        for (let i = 0; i < dim; i++) {
          const ga = E[b][i] - E[corrupt][i];
          const gb = E[a][i], gc = -E[a][i];
          E[a][i] += lr * ga; E[b][i] += lr * gb; E[corrupt][i] += lr * gc;
        }
      }
    }
  }
  return score;
}
const dm = trainDistMult();

// z-normalize a score column over the candidate list so fusion is scale-fair
function fused(sIdx, scoreA, scoreB, wB) {
  const cands = [];
  for (let t = 0; t < N; t++) { if (t === sIdx || adj[sIdx].has(t)) continue; cands.push(t); }
  const a = cands.map((t) => scoreA(sIdx, t)), b = cands.map((t) => scoreB(sIdx, t));
  const zn = (arr) => { const m = arr.reduce((s, x) => s + x, 0) / arr.length; const sd = Math.sqrt(arr.reduce((s, x) => s + (x - m) ** 2, 0) / arr.length) || 1; return arr.map((x) => (x - m) / sd); };
  const za = zn(a), zb = zn(b);
  return cands.map((t, i) => [za[i] + wB * zb[i], t]).sort((x, y) => y[0] - x[0]);
}

// evaluate held-out edges (both directions) with each method
const methods = {
  jina: (s) => rankBy(s, jinaScore),
  commonNeighbors: (s) => rankBy(s, commonNeighbors),
  adamicAdar: (s) => rankBy(s, adamicAdar),
  distMult: (s) => rankBy(s, dm),
  "jina+AA": (s) => fused(s, jinaScore, adamicAdar, 1),
  "jina+distMult": (s) => fused(s, jinaScore, dm, 1),
};
const results = {};
for (const [name, ranker] of Object.entries(methods)) {
  let hit10 = 0, mrrSum = 0, tot = 0;
  const rankCacheM = new Map();
  for (const [a, b] of test) {
    for (const [s, tgt] of [[a, b], [b, a]]) {
      if (adj[s].has(tgt)) continue; // shouldn't happen (test excluded from adj)
      if (!rankCacheM.has(s)) rankCacheM.set(s, ranker(s));
      const ranked = rankCacheM.get(s);
      const pos = ranked.findIndex(([, t]) => t === tgt) + 1;
      if (pos >= 1 && pos <= 10) hit10++;
      if (pos >= 1) mrrSum += 1 / pos;
      tot++;
    }
  }
  results[name] = { recallAt10: +(hit10 / tot).toFixed(3), mrr: +(mrrSum / tot).toFixed(3), evaluated: tot };
}
console.log("\n== held-out link prediction ==");
for (const [k, v] of Object.entries(results)) console.log(`  ${k.padEnd(16)} recall@10 ${v.recallAt10}  MRR ${v.mrr}`);
const degs = adj.map((s) => s.size);
const out = {
  graph: { notes: N, edges: edges.length, avgDegree: +(2 * edges.length / N).toFixed(2),
    isolatedNotes: degs.filter((d) => d === 0).length, medianDegree: degs.sort((a, b) => a - b)[Math.floor(N / 2)] },
  heldOut: test.length, results,
  verdict: results["jina+AA"].recallAt10 > results.jina.recallAt10 + 0.02 || results["jina+distMult"].recallAt10 > results.jina.recallAt10 + 0.02
    ? "structural signal helps — pursue MUSAE/LINE/KGE fusion" : "no meaningful structural lift on this vault — defer graph-embedding branch",
};
writeFileSync("/Users/justus/obsidian_atomized_intermediary/lab/results/graph-fusion-eval.json", JSON.stringify(out, null, 1));
console.log("\nverdict:", out.verdict);
console.log("wrote lab/results/graph-fusion-eval.json");
