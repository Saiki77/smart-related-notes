// Graph x content fusion v2 — tests the DEEPER paper ideas, not DistMult-as-black-box:
//   * MUSAE-lite: multi-scale attributed representation (1-hop, 2-hop neighbor-mean content)
//   * LINE 2nd-order: similarity of shared-neighbor content signatures
//   * NON-OBVIOUS recovery: of held-out links NOT in the source's content-only top-10,
//     how many does fusion pull into top-10? (the links similarity alone misses)
//   * cluster-on-fused: does attributed (content+structure) clustering beat content-only
//     purity 0.605 for concept-search / cartography?
// No leakage: neighbor features for link-prediction use TRAIN adjacency only.
//   node bench/graph-fusion-eval2.mjs
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
const zn = (arr) => { const m = arr.reduce((s, x) => s + x, 0) / arr.length; const sd = Math.sqrt(arr.reduce((s, x) => s + (x - m) ** 2, 0) / arr.length) || 1; return arr.map((x) => (x - m) / sd); };
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
  notes.push({ rel, basename, body, folder: rel.split("/").length > 2 ? rel.split("/")[1] : "_root", v: vec(noteText(basename, body)) });
}
const withV = notes.filter((n) => n.v);
const idx = new Map(withV.map((n, i) => [n, i]));
const byFold = new Map(withV.map((n) => [fold(n.basename), n]));
const N = withV.length, DIM = withV[0].v.length;
const mean = new Array(DIM).fill(0);
for (const n of withV) for (let i = 0; i < DIM; i++) mean[i] += n.v[i];
for (let i = 0; i < DIM; i++) mean[i] /= N;
const C = withV.map((n) => l2(n.v.map((x, i) => x - mean[i]))); // centered content, by index

// edges
const edgeSet = new Set(), edges = [];
for (const n of withV) for (const m of n.body.matchAll(/\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]/g)) {
  const t = byFold.get(fold(m[1]));
  if (!t || t === n) continue;
  const a = Math.min(idx.get(n), idx.get(t)), b = Math.max(idx.get(n), idx.get(t));
  const key = a + "-" + b;
  if (!edgeSet.has(key)) { edgeSet.add(key); edges.push([a, b]); }
}
const rnd = mulberry32(42);
const shuffled = fisherYates(edges, rnd);
const nTest = Math.floor(edges.length * 0.2);
const test = shuffled.slice(0, nTest), train = shuffled.slice(nTest);
const adjTrain = Array.from({ length: N }, () => new Set());
for (const [a, b] of train) { adjTrain[a].add(b); adjTrain[b].add(a); }

// r-hop neighbor-mean of centered content (attributed multi-scale; MUSAE spirit).
// From a given adjacency. Returns per-node normalized vectors, or null if no neighbors.
function hopMeans(adj, maxHop) {
  const scales = [];
  let frontierVecs = C.map((c) => c.slice()); // hop 0 = self content
  const reached = adj.map((s, i) => new Set([i]));
  let cur = adj.map((_, i) => new Set([i]));
  for (let h = 1; h <= maxHop; h++) {
    const next = adj.map(() => new Set());
    for (let i = 0; i < N; i++) for (const u of cur[i]) for (const w of adj[u]) if (!reached[i].has(w)) { next[i].add(w); }
    for (let i = 0; i < N; i++) for (const w of next[i]) reached[i].add(w);
    const hv = new Array(N);
    for (let i = 0; i < N; i++) {
      if (!next[i].size) { hv[i] = null; continue; }
      const m = new Array(DIM).fill(0);
      for (const w of next[i]) for (let d = 0; d < DIM; d++) m[d] += C[w][d];
      hv[i] = l2(m);
    }
    scales.push(hv);
    cur = next;
  }
  return scales; // scales[0] = 1-hop mean, scales[1] = 2-hop ring mean, ...
}
const hops = hopMeans(adjTrain, 2);
const m1 = hops[0], m2 = hops[1];
const cosOrNull = (va, vb) => (va && vb) ? dot(va, vb) : null;

// ---------- link prediction ----------
const adamicAdar = (s, t) => { let a = 0; for (const x of adjTrain[s]) if (adjTrain[t].has(x)) a += 1 / Math.log(1 + adjTrain[x].size); return a; };
function rankGeneric(sIdx, scoreFn) {
  const out = [];
  for (let t = 0; t < N; t++) { if (t === sIdx || adjTrain[sIdx].has(t)) continue; out.push([scoreFn(sIdx, t), t]); }
  out.sort((a, b) => b[0] - a[0]);
  return out;
}
// fused score over a candidate list, z-normalizing each component fairly
function rankFused(sIdx, comps) {
  const cands = [];
  for (let t = 0; t < N; t++) { if (t === sIdx || adjTrain[sIdx].has(t)) continue; cands.push(t); }
  const cols = comps.map(({ fn }) => zn(cands.map((t) => fn(sIdx, t))));
  return cands.map((t, i) => [comps.reduce((s, c, k) => s + c.w * cols[k][i], 0), t]).sort((a, b) => b[0] - a[0]);
}
const scoreContent = (s, t) => dot(C[s], C[t]);
const scoreMs1 = (s, t) => cosOrNull(m1[s], m1[t]) ?? -2; // LINE 2nd-order: shared-neighbor content
const scoreMs2 = (s, t) => cosOrNull(m2[s], m2[t]) ?? -2;

const methods = {
  content: (s) => rankGeneric(s, scoreContent),
  "ms1_2ndorder": (s) => rankGeneric(s, scoreMs1),
  "content+AA": (s) => rankFused(s, [{ fn: scoreContent, w: 1 }, { fn: adamicAdar, w: 1 }]),
  "MUSAE(content+ms1+ms2)": (s) => rankFused(s, [{ fn: scoreContent, w: 1 }, { fn: scoreMs1, w: 0.7 }, { fn: scoreMs2, w: 0.4 }]),
  "MUSAE+AA": (s) => rankFused(s, [{ fn: scoreContent, w: 1 }, { fn: scoreMs1, w: 0.7 }, { fn: scoreMs2, w: 0.4 }, { fn: adamicAdar, w: 0.8 }]),
};
const rankCaches = Object.fromEntries(Object.keys(methods).map((k) => [k, new Map()]));
function rankedOf(name, s) { const c = rankCaches[name]; if (!c.has(s)) c.set(s, methods[name](s)); return c.get(s); }

const linkResults = {};
for (const name of Object.keys(methods)) {
  let hit10 = 0, mrr = 0, tot = 0;
  for (const [a, b] of test) for (const [s, tgt] of [[a, b], [b, a]]) {
    const ranked = rankedOf(name, s);
    const pos = ranked.findIndex(([, t]) => t === tgt) + 1;
    if (pos >= 1 && pos <= 10) hit10++;
    if (pos >= 1) mrr += 1 / pos;
    tot++;
  }
  linkResults[name] = { recallAt10: +(hit10 / tot).toFixed(3), mrr: +(mrr / tot).toFixed(3) };
}

// ---------- THE non-obvious metric ----------
// A held-out edge is "obvious" if the target is already in the source's content-only
// top-10 (plain similarity would suggest it). "non-obvious" otherwise. For the
// non-obvious subset, does the best fusion recover the target into top-10?
const BEST = "MUSAE+AA";
let obviousN = 0, nonObviousN = 0, nonObvRecovered = 0, nonObvContentTop25 = 0;
const nonObviousExamples = [];
for (const [a, b] of test) for (const [s, tgt] of [[a, b], [b, a]]) {
  const cont = rankedOf("content", s);
  const contPos = cont.findIndex(([, t]) => t === tgt) + 1;
  const obvious = contPos >= 1 && contPos <= 10;
  if (obvious) { obviousN++; continue; }
  nonObviousN++;
  const fused = rankedOf(BEST, s);
  const fPos = fused.findIndex(([, t]) => t === tgt) + 1;
  if (fPos >= 1 && fPos <= 10) {
    nonObvRecovered++;
    if (nonObviousExamples.length < 25) nonObviousExamples.push({ from: withV[s].rel, to: withV[tgt].rel, contentRank: contPos || null, fusedRank: fPos });
  }
}

// ---------- cluster on fused vs content (full graph; folder-graph correlation noted) ----------
const adjFull = Array.from({ length: N }, () => new Set());
for (const [a, b] of edges) { adjFull[a].add(b); adjFull[b].add(a); }
const full1 = hopMeans(adjFull, 1)[0];
function attributed(lambda) {
  return C.map((c, i) => {
    const s = full1[i];
    if (!s) return c.slice();
    return l2(c.concat(s.map((x) => x * lambda))); // concat content ⊕ λ·(1-hop neighbor content)
  });
}
function kmeans(vectors, k, seed) {
  let best = null;
  for (let r = 0; r < 5; r++) {
    const rr = mulberry32(seed + r * 1000);
    const cents = [vectors[Math.floor(rr() * vectors.length)]];
    while (cents.length < k) {
      const d = vectors.map((v) => Math.min(...cents.map((c) => 1 - dot(v, c) / ((Math.sqrt(dot(v, v)) || 1) * (Math.sqrt(dot(c, c)) || 1)))));
      const sum = d.reduce((s, x) => s + x, 0); let pick = rr() * sum, ix = 0;
      for (; ix < d.length - 1 && pick > d[ix]; ix++) pick -= d[ix];
      cents.push(vectors[ix]);
    }
    const assign = new Array(vectors.length).fill(0);
    for (let it = 0; it < 40; it++) {
      let ch = 0;
      for (let i = 0; i < vectors.length; i++) { let bi = 0, bs = -2; for (let j = 0; j < k; j++) { const s = dot(vectors[i], cents[j]); if (s > bs) { bs = s; bi = j; } } if (assign[i] !== bi) { assign[i] = bi; ch++; } }
      for (let j = 0; j < k; j++) { const mem = vectors.filter((_, i) => assign[i] === j); if (!mem.length) continue; const m = new Array(mem[0].length).fill(0); for (const v of mem) for (let i = 0; i < v.length; i++) m[i] += v[i]; cents[j] = l2(m); }
      if (!ch) break;
    }
    const inertia = vectors.reduce((s, v, i) => s + (1 - dot(v, cents[i in assign ? assign[i] : 0])), 0);
    if (!best || inertia < best.inertia) best = { assign, inertia };
  }
  return best;
}
function purity(assign, labels) {
  const cl = new Map();
  for (let i = 0; i < labels.length; i++) { if (!cl.has(assign[i])) cl.set(assign[i], new Map()); const m = cl.get(assign[i]); m.set(labels[i], (m.get(labels[i]) ?? 0) + 1); }
  let p = 0; for (const [, m] of cl) p += Math.max(...m.values());
  return +(p / labels.length).toFixed(3);
}
const cartoIdx = withV.map((n, i) => i).filter((i) => withV[i].folder !== "_root" && withV[i].rel.startsWith("real/") && !/^Daily$/.test(withV[i].folder));
const labels = cartoIdx.map((i) => withV[i].folder);
const clusterResults = {};
const contVecs = cartoIdx.map((i) => C[i]);
clusterResults.contentOnly = purity(kmeans(contVecs, 12, 42).assign, labels);
for (const lam of [0.5, 1.0, 1.5]) {
  const av = attributed(lam);
  clusterResults["fused_lambda" + lam] = purity(kmeans(cartoIdx.map((i) => av[i]), 12, 42).assign, labels);
}

const out = {
  graph: { notes: N, edges: edges.length, heldOut: test.length },
  linkPrediction: linkResults,
  nonObvious: {
    obviousEdges: obviousN, nonObviousEdges: nonObviousN,
    nonObviousShare: +(nonObviousN / (obviousN + nonObviousN)).toFixed(3),
    fusionRecoversNonObvious: nonObviousN ? +(nonObvRecovered / nonObviousN).toFixed(3) : null,
    recoveredCount: nonObvRecovered, method: BEST,
    examples: nonObviousExamples.slice(0, 12),
  },
  clusteringPurity: { note: "full-graph attributed clustering; folders correlate with links (report honestly)", ...clusterResults },
};
console.log(JSON.stringify(out, null, 1));
writeFileSync("/Users/justus/obsidian_atomized_intermediary/lab/results/graph-fusion-eval2.json", JSON.stringify(out, null, 1));
console.log("wrote lab/results/graph-fusion-eval2.json");
