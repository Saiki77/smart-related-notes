// Across-the-board retrieval levers — all TRAINING-FREE, measured on the same
// held-out-link harness as graph-fusion-eval2 (seed 42, 20% of real wikilinks).
//
// Content-side (hubness / anisotropy literature):
//   ABTT   all-but-the-top: remove the top-D principal directions after centering
//          (Mu & Viswanath, ICLR'18)
//   CSLS   cross-domain similarity local scaling: sim = 2cos(s,t) - r_k(s) - r_k(t)
//          where r_k(x) = mean cosine to x's k nearest neighbours (Conneau et al.,
//          ICLR'18) — directly penalizes hub notes that are everyone's neighbour
//   MUTUAL mutual-rank symmetrization: score = -(rank_s(t) + rank_t(s))
//
// Structural (parameter-free link-prediction scores):
//   AA     Adamic-Adar (current)      RA   Resource Allocation (1/deg, sharper)
//   KATZ   truncated Katz (3 hops)    PPR  rooted personalized PageRank
//
// Fusion: z-score sum (current) vs Reciprocal Rank Fusion (RRF, Cormack et al.'09).
//
//   node bench/levers-eval.mjs
import { readFileSync, writeFileSync } from "node:fs";
import { relative } from "node:path";
import { walk, stripFront, noteText } from "./jina-cache.mjs";
import { fisherYates } from "./shuffle.mjs";

const VAULT = "/Users/justus/obsidian_atomized_intermediary/lab/vault";
const CACHE_PATH = process.env.JINA_CACHE || (process.env.HOME + "/.cache/srn-lab/jina-cache.json");
const OUT = "/Users/justus/obsidian_atomized_intermediary/lab/results/levers-eval.json";
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
  const body = stripFront(readFileSync(abs, "utf8"));
  notes.push({ rel, basename, body, v: vec(noteText(basename, body)) });
}
const withV = notes.filter((n) => n.v);
const idx = new Map(withV.map((n, i) => [n, i]));
const byFold = new Map(withV.map((n) => [fold(n.basename), n]));
const N = withV.length, DIM = withV[0].v.length;
const mean = new Array(DIM).fill(0);
for (const n of withV) for (let i = 0; i < DIM; i++) mean[i] += n.v[i];
for (let i = 0; i < DIM; i++) mean[i] /= N;
const C = withV.map((n) => l2(n.v.map((x, i) => x - mean[i])));

// ---------- edges + identical 42-seed split as graph-fusion-eval2 ----------
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
console.log(`${N} notes, ${edges.length} edges (${train.length} train / ${test.length} test)`);

// ---------- content transforms ----------
// ABTT: remove top-D principal directions (power iteration + deflation)
function topPCs(vectors, D) {
  const comps = [];
  const r = mulberry32(7);
  for (let c = 0; c < D; c++) {
    let w = l2(new Array(DIM).fill(0).map(() => r() - 0.5));
    for (let it = 0; it < 50; it++) {
      const out = new Array(DIM).fill(0);
      for (const x of vectors) { const s = dot(x, w); for (let i = 0; i < DIM; i++) out[i] += s * x[i]; }
      let nw = out;
      for (const p of comps) { const s = dot(nw, p); nw = nw.map((x, i) => x - s * p[i]); }
      w = l2(nw);
    }
    comps.push(w);
  }
  return comps;
}
function abtt(vectors, D) {
  const pcs = topPCs(vectors, D);
  return vectors.map((v) => {
    let x = v.slice();
    for (const p of pcs) { const s = dot(x, p); x = x.map((y, i) => y - s * p[i]); }
    return l2(x);
  });
}
const C_abtt2 = abtt(C, 2), C_abtt3 = abtt(C, 3);

// full pairwise cosine matrices (N=488 -> fine)
function simMatrix(vecs) {
  const S = Array.from({ length: N }, () => new Float64Array(N));
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) { const s = dot(vecs[i], vecs[j]); S[i][j] = s; S[j][i] = s; }
  return S;
}
const S_base = simMatrix(C), S_abtt2 = simMatrix(C_abtt2), S_abtt3 = simMatrix(C_abtt3);

// CSLS local scaling term r_k(x)
function cslsR(S, k = 10) {
  const r = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const row = [];
    for (let j = 0; j < N; j++) if (j !== i) row.push(S[i][j]);
    row.sort((a, b) => b - a);
    r[i] = row.slice(0, k).reduce((s, x) => s + x, 0) / k;
  }
  return r;
}
const R_base = cslsR(S_base), R_abtt2 = cslsR(S_abtt2);
const cslsSim = (S, R) => (s, t) => 2 * S[s][t] - R[s] - R[t];

// Mutual Proximity (Schnitzer et al., JMLR 2012), independent-Gaussian form:
// MP(x,y) = P(d_xy > mu_x) * P(d_yx > mu_y) using per-note distance mean/std.
// Gives CALIBRATED [0,1] scores -> a principled absolute threshold (precision lever).
function mpMatrix(S) {
  const mu = new Float64Array(N), sd = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    let m = 0, c = 0;
    for (let j = 0; j < N; j++) if (j !== i) { m += 1 - S[i][j]; c++; }
    mu[i] = m / c;
    let v = 0;
    for (let j = 0; j < N; j++) if (j !== i) v += (1 - S[i][j] - mu[i]) ** 2;
    sd[i] = Math.sqrt(v / c) || 1;
  }
  const erf = (x) => { const t = 1 / (1 + 0.3275911 * Math.abs(x)); const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return x >= 0 ? y : -y; };
  const sf = (d, i) => 0.5 * (1 - erf((d - mu[i]) / (sd[i] * Math.SQRT2))); // P(D > d)
  return (s, t) => { const d = 1 - S[s][t]; return sf(d, s) * sf(d, t); };
}
const mpSim = mpMatrix(S_base);

// mutual-rank symmetrization
function rankMatrixRanks(S) {
  const ranks = Array.from({ length: N }, () => new Int32Array(N));
  for (let i = 0; i < N; i++) {
    const order = [];
    for (let j = 0; j < N; j++) if (j !== i) order.push([S[i][j], j]);
    order.sort((a, b) => b[0] - a[0]);
    order.forEach(([, j], pos) => { ranks[i][j] = pos + 1; });
  }
  return ranks;
}
const RANK_base = rankMatrixRanks(S_base);
const mutualSim = (s, t) => -(RANK_base[s][t] + RANK_base[t][s]);

// ---------- structural scores (train adjacency only) ----------
const deg = adjTrain.map((s) => s.size);
const adamicAdar = (s, t) => { let a = 0; for (const x of adjTrain[s]) if (adjTrain[t].has(x)) a += 1 / Math.log(1 + deg[x]); return a; };
const resourceAlloc = (s, t) => { let a = 0; for (const x of adjTrain[s]) if (adjTrain[t].has(x)) a += 1 / Math.max(1, deg[x]); return a; };
// truncated Katz: beta*A + beta^2*A^2 + beta^3*A^3
const KATZ_BETA = 0.05;
const katz = (() => {
  const A2 = Array.from({ length: N }, () => new Map());
  for (let i = 0; i < N; i++) for (const u of adjTrain[i]) for (const w of adjTrain[u]) if (w !== i) A2[i].set(w, (A2[i].get(w) ?? 0) + 1);
  const A3 = Array.from({ length: N }, () => new Map());
  for (let i = 0; i < N; i++) for (const [u, c] of A2[i]) for (const w of adjTrain[u]) if (w !== i) A3[i].set(w, (A3[i].get(w) ?? 0) + c);
  return (s, t) => KATZ_BETA * (adjTrain[s].has(t) ? 1 : 0)
    + KATZ_BETA ** 2 * (A2[s].get(t) ?? 0)
    + KATZ_BETA ** 3 * (A3[s].get(t) ?? 0);
})();
// L3 (Kovacs et al., Nat.Commun. 2019): paths of length 3, degree-normalised.
// Critical at <k>~4.8: most pairs share NO neighbour, so AA/RA are exactly 0 and
// contribute nothing; L3 scores them via x-u-v-y paths.
const l3 = (s, t) => {
  let acc = 0;
  for (const u of adjTrain[s]) for (const v of adjTrain[u]) {
    if (v === s || v === t) continue;
    if (adjTrain[t].has(v)) acc += 1 / Math.sqrt(Math.max(1, deg[u]) * Math.max(1, deg[v]));
  }
  return acc;
};
// coverage diagnostic: fraction of test pairs with zero common neighbours
// rooted personalized PageRank (computed lazily per source, cached)
const pprCache = new Map();
function ppr(s) {
  if (pprCache.has(s)) return pprCache.get(s);
  const alpha = 0.15;
  let p = new Float64Array(N); p[s] = 1;
  for (let it = 0; it < 25; it++) {
    const np = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      if (!p[i]) continue;
      if (deg[i] === 0) { np[s] += (1 - alpha) * p[i]; continue; }
      const share = (1 - alpha) * p[i] / deg[i];
      for (const j of adjTrain[i]) np[j] += share;
    }
    for (let i = 0; i < N; i++) np[i] += alpha * (i === s ? 1 : 0);
    p = np;
  }
  pprCache.set(s, p);
  return p;
}
const pprScore = (s, t) => ppr(s)[t];

// ---------- ranking / fusion ----------
function candidatesOf(s) {
  const out = [];
  for (let t = 0; t < N; t++) if (t !== s && !adjTrain[s].has(t)) out.push(t);
  return out;
}
function rankZ(s, comps) {
  const cands = candidatesOf(s);
  const cols = comps.map(({ fn }) => zn(cands.map((t) => fn(s, t))));
  return cands.map((t, i) => [comps.reduce((acc, c, k) => acc + c.w * cols[k][i], 0), t]).sort((a, b) => b[0] - a[0]);
}
function rankRRF(s, fns, k = 60) {
  const cands = candidatesOf(s);
  const rankPer = fns.map((fn) => {
    const ord = cands.map((t) => [fn(s, t), t]).sort((a, b) => b[0] - a[0]);
    const r = new Map();
    ord.forEach(([, t], i) => r.set(t, i + 1));
    return r;
  });
  return cands.map((t) => [rankPer.reduce((acc, r) => acc + 1 / (k + r.get(t)), 0), t]).sort((a, b) => b[0] - a[0]);
}
const plain = (S) => (s, t) => S[s][t];

const methods = {
  "content (baseline)": (s) => rankZ(s, [{ fn: plain(S_base), w: 1 }]),
  "content+ABTT2": (s) => rankZ(s, [{ fn: plain(S_abtt2), w: 1 }]),
  "content+ABTT3": (s) => rankZ(s, [{ fn: plain(S_abtt3), w: 1 }]),
  "content+CSLS": (s) => rankZ(s, [{ fn: cslsSim(S_base, R_base), w: 1 }]),
  "content+CSLS+ABTT2": (s) => rankZ(s, [{ fn: cslsSim(S_abtt2, R_abtt2), w: 1 }]),
  "content+mutualRank": (s) => rankZ(s, [{ fn: mutualSim, w: 1 }]),
  "AA alone": (s) => rankZ(s, [{ fn: adamicAdar, w: 1 }]),
  "RA alone": (s) => rankZ(s, [{ fn: resourceAlloc, w: 1 }]),
  "Katz alone": (s) => rankZ(s, [{ fn: katz, w: 1 }]),
  "PPR alone": (s) => rankZ(s, [{ fn: pprScore, w: 1 }]),
  "content+AA (current)": (s) => rankZ(s, [{ fn: plain(S_base), w: 1 }, { fn: adamicAdar, w: 1 }]),
  "content+RA": (s) => rankZ(s, [{ fn: plain(S_base), w: 1 }, { fn: resourceAlloc, w: 1 }]),
  "CSLS+AA": (s) => rankZ(s, [{ fn: cslsSim(S_base, R_base), w: 1 }, { fn: adamicAdar, w: 1 }]),
  "CSLS+RA": (s) => rankZ(s, [{ fn: cslsSim(S_base, R_base), w: 1 }, { fn: resourceAlloc, w: 1 }]),
  "CSLS+RA+Katz": (s) => rankZ(s, [{ fn: cslsSim(S_base, R_base), w: 1 }, { fn: resourceAlloc, w: 1 }, { fn: katz, w: 0.6 }]),
  "CSLS+RA+PPR": (s) => rankZ(s, [{ fn: cslsSim(S_base, R_base), w: 1 }, { fn: resourceAlloc, w: 1 }, { fn: pprScore, w: 0.6 }]),
  "CSLS+ABTT2+RA": (s) => rankZ(s, [{ fn: cslsSim(S_abtt2, R_abtt2), w: 1 }, { fn: resourceAlloc, w: 1 }]),
  "L3 alone": (s) => rankZ(s, [{ fn: l3, w: 1 }]),
  "CSLS+L3": (s) => rankZ(s, [{ fn: cslsSim(S_base, R_base), w: 1 }, { fn: l3, w: 1 }]),
  "CSLS+RA+L3": (s) => rankZ(s, [{ fn: cslsSim(S_base, R_base), w: 1 }, { fn: resourceAlloc, w: 1 }, { fn: l3, w: 0.6 }]),
  "CSLS+RA+PPR+L3": (s) => rankZ(s, [{ fn: cslsSim(S_base, R_base), w: 1 }, { fn: resourceAlloc, w: 1 }, { fn: pprScore, w: 0.6 }, { fn: l3, w: 0.6 }]),
  "content+MP": (s) => rankZ(s, [{ fn: mpSim, w: 1 }]),
  "MP+RA": (s) => rankZ(s, [{ fn: mpSim, w: 1 }, { fn: resourceAlloc, w: 1 }]),
  "MP+RA+PPR": (s) => rankZ(s, [{ fn: mpSim, w: 1 }, { fn: resourceAlloc, w: 1 }, { fn: pprScore, w: 0.6 }]),
  "RRF(content, AA)": (s) => rankRRF(s, [plain(S_base), adamicAdar]),
  "RRF(CSLS, RA)": (s) => rankRRF(s, [cslsSim(S_base, R_base), resourceAlloc]),
  "RRF(CSLS, RA, PPR)": (s) => rankRRF(s, [cslsSim(S_base, R_base), resourceAlloc, pprScore]),
};

// ---------- evaluate ----------
const contentTop10 = [];
for (let s = 0; s < N; s++) {
  const sc = [];
  for (let t = 0; t < N; t++) if (t !== s) sc.push([S_base[s][t], t]);
  sc.sort((a, b) => b[0] - a[0]);
  contentTop10.push(new Set(sc.slice(0, 10).map((x) => x[1])));
}
const queries = [];
for (const [a, b] of test) { queries.push([a, b]); queries.push([b, a]); }
const nonObvious = queries.filter(([s, t]) => !contentTop10[s].has(t));

const results = {};
for (const name of Object.keys(methods)) {
  const cacheR = new Map();
  const rankedOf = (s) => { if (!cacheR.has(s)) cacheR.set(s, methods[name](s)); return cacheR.get(s); };
  let hit5 = 0, hit10 = 0, mrr = 0;
  for (const [s, tgt] of queries) {
    const pos = rankedOf(s).findIndex(([, t]) => t === tgt) + 1;
    if (pos >= 1 && pos <= 5) hit5++;
    if (pos >= 1 && pos <= 10) hit10++;
    if (pos >= 1) mrr += 1 / pos;
  }
  let noHit10 = 0;
  for (const [s, tgt] of nonObvious) {
    const pos = rankedOf(s).findIndex(([, t]) => t === tgt) + 1;
    if (pos >= 1 && pos <= 10) noHit10++;
  }
  results[name] = {
    recallAt5: +(hit5 / queries.length).toFixed(3),
    recallAt10: +(hit10 / queries.length).toFixed(3),
    mrr: +(mrr / queries.length).toFixed(3),
    nonObviousRecovery: +(noHit10 / nonObvious.length).toFixed(3),
  };
  console.log(`${name.padEnd(24)} R@5 ${results[name].recallAt5.toFixed(3)}  R@10 ${results[name].recallAt10.toFixed(3)}  MRR ${results[name].mrr.toFixed(3)}  nonObv ${results[name].nonObviousRecovery.toFixed(3)}`);
}

// hubness diagnostics: k-occurrence skewness before/after each content transform
function hubness(S, k = 10) {
  const occ = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const ord = [];
    for (let j = 0; j < N; j++) if (j !== i) ord.push([S[i][j], j]);
    ord.sort((a, b) => b[0] - a[0]);
    for (const [, j] of ord.slice(0, k)) occ[j]++;
  }
  const m = occ.reduce((s, x) => s + x, 0) / N;
  const sd = Math.sqrt(occ.reduce((s, x) => s + (x - m) ** 2, 0) / N) || 1;
  const skew = occ.reduce((s, x) => s + ((x - m) / sd) ** 3, 0) / N;
  return { meanOcc: +m.toFixed(2), maxOcc: Math.max(...occ), skewness: +skew.toFixed(2) };
}
const hub = {
  centered: hubness(S_base), abtt2: hubness(S_abtt2), abtt3: hubness(S_abtt3),
  mp: (() => { const S = Array.from({ length: N }, () => new Float64Array(N)); for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) if (i !== j) S[i][j] = mpSim(i, j); return hubness(S); })(),
  csls: (() => { const f = cslsSim(S_base, R_base); const S = Array.from({ length: N }, () => new Float64Array(N)); for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) if (i !== j) S[i][j] = f(i, j); return hubness(S); })(),
};
console.log("\nhubness (k-occurrence skewness; lower = fewer hub notes):", JSON.stringify(hub));

const zeroCN = queries.filter(([s, t]) => [...adjTrain[s]].filter((x) => adjTrain[t].has(x)).length === 0).length;
console.log(`test pairs with ZERO common neighbours (AA/RA blind): ${zeroCN}/${queries.length} = ${(zeroCN / queries.length).toFixed(3)}`);
const out = { N, edges: edges.length, test: test.length, queries: queries.length, nonObviousQueries: nonObvious.length, zeroCommonNeighbourPairs: zeroCN, results, hubness: hub };
writeFileSync(OUT, JSON.stringify(out, null, 1));
console.log("\nwrote", OUT);
