// PERSONAL METRIC LEARNING — can the ranker learn what YOU mean by "related"?
//
// The plugin uses a frozen, generic embedding: it encodes what the model was
// trained to think is similar, not what this user links together. But the vault
// contains ~1300 labelled examples of the user's own notion of relatedness: the
// links they made by hand.
//
// Idea: a linked pair (a, b) is a statement that a and b belong together DESPITE
// however they differ. So the direction (x_a - x_b) is, for this user, largely
// irrelevant variation. Collect those difference directions, find the ones that
// recur, and suppress them. Nothing is generated and no transformer is trained:
// this is an eigendecomposition over the user's own graph, seconds of linear
// algebra, entirely on device.
//
//   node bench/v4-metric.mjs
//
// Variants
//   frozen      today's ranker (centered cosine)
//   suppress-k  project out the top-k linked-difference directions
//   rca         soften rather than delete them (Relevant Component Analysis:
//               scale each direction by 1/sqrt(lambda + eps) instead of by 0)
//
// Protocol: 20% of links held out, the metric is fit on the REMAINING 80% only,
// and evaluation is on the held-out edges. Fitting on all links and scoring the
// same links would be circular and would look spectacular.
import { readFileSync, writeFileSync, readdirSync, statSync, mkdirSync } from "node:fs";
import { join, relative } from "node:path";
import { fisherYates } from "./shuffle.mjs";

const VAULT = process.env.LAB_VAULT || "/Users/justus/obsidian_atomized_intermediary/lab/vault";
const OUT = "/Users/justus/obsidian_atomized_intermediary/lab/results/v4-metric.json";
const MODEL = process.env.LAB_MODEL || "jinaai/jina-embeddings-v5-text-nano-text-matching";
const CACHE = join(process.env.HOME, ".cache/srn-lab", `v3-${MODEL.replace(/[^a-z0-9]/gi, "_")}.json`);
const RANKS = (process.env.V4_K || "4,8,16,32,64").split(",").map(Number);
const EPS = Number(process.env.V4_EPS ?? 0.15);

const fold = (s) => s.toLowerCase().normalize("NFC").replace(/ä/g, "ae").replace(/ö/g, "oe").replace(/ü/g, "ue").replace(/ß/g, "ss").replace(/\s+/g, " ").trim();
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const l2 = (v) => { const s = Math.sqrt(v.reduce((t, x) => t + x * x, 0)) || 1; return v.map((x) => x / s); };
function mulberry32(a) { return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function walk(dir) {
  const out = [];
  for (const n of readdirSync(dir)) {
    if (n.startsWith(".")) continue;
    const p = join(dir, n), s = statSync(p);
    if (s.isDirectory()) out.push(...walk(p)); else if (n.endsWith(".md")) out.push(p);
  }
  return out;
}
const stripFront = (r) => { const m = r.match(/^---\n[\s\S]*?\n---\n?/); return m ? r.slice(m[0].length) : r; };
const noteText = (n) => (n.basename + "\n\n" + n.body).slice(0, 8000);

// ---------- corpus ----------
const manifest = JSON.parse(readFileSync("/Users/justus/obsidian_atomized_intermediary/lab/corpus-manifest.json", "utf8"));
const INDEXABLE = new Set(manifest.answer_paths.map((p) => "real/" + p));
const cache = JSON.parse(readFileSync(CACHE, "utf8"));
const notes = [];
for (const abs of walk(VAULT)) {
  const rel = relative(VAULT, abs);
  if (rel.startsWith("real/") && !INDEXABLE.has(rel)) continue;
  const basename = rel.replace(/\.md$/, "").split("/").pop();
  if (/(^|\/)Attachments\//.test(rel) || /\.dup$/.test(basename)) continue;
  const body = stripFront(readFileSync(abs, "utf8"));
  const v = cache[noteText({ basename, body })];
  if (v) notes.push({ rel, basename, body, v });
}
const N = notes.length, D = notes[0].v.length;
const mean = new Array(D).fill(0);
for (const n of notes) for (let i = 0; i < D; i++) mean[i] += n.v[i] / N;
const C = notes.map((n) => l2(n.v.map((x, i) => x - mean[i])));
const byFold = new Map(notes.map((n, i) => [fold(n.basename), i]));

// ---------- link graph, split ----------
const uniq = new Set();
for (let s = 0; s < N; s++) {
  for (const m of notes[s].body.matchAll(/\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]/g)) {
    const name = m[1].trim(); if (!name) continue;
    const t = byFold.get(fold(name));
    if (t === undefined || t === s) continue;
    uniq.add(`${Math.min(s, t)}-${Math.max(s, t)}`);
  }
}
const edges = [...uniq].map((k) => k.split("-").map(Number));
const rnd = mulberry32(20260727);
const shuffled = fisherYates(edges, rnd);
const nHold = Math.floor(shuffled.length * 0.2);
const heldOut = shuffled.slice(0, nHold), train = shuffled.slice(nHold);
const adj = Array.from({ length: N }, () => new Set());
for (const [a, b] of train) { adj[a].add(b); adj[b].add(a); }
console.log(`${N} notes, ${edges.length} links: ${train.length} to learn from, ${heldOut.length} held out`);

// ---------- the linked-difference directions ----------
// M = mean over TRAINING links of (x_a - x_b)(x_a - x_b)^T. Its leading
// eigenvectors are the directions along which this user's linked notes most
// often differ, i.e. the variation they consider irrelevant to relatedness.
// Power iteration with deflation: never materialise the 768x768 matrix.
const diffs = train.map(([a, b]) => {
  const d = new Array(D);
  for (let i = 0; i < D; i++) d[i] = C[a][i] - C[b][i];
  return d;
});
function topEigenvectors(vectors, k, seed = 5) {
  const rnd2 = mulberry32(seed);
  const comps = [], vals = [];
  for (let c = 0; c < k; c++) {
    let w = l2(new Array(D).fill(0).map(() => rnd2() - 0.5));
    let lambda = 0;
    for (let iter = 0; iter < 50; iter++) {
      const next = new Array(D).fill(0);
      for (const d of vectors) {
        const s = dot(d, w);
        for (let i = 0; i < D; i++) next[i] += s * d[i];
      }
      for (let i = 0; i < D; i++) next[i] /= vectors.length;
      for (const p of comps) {           // deflate against what we already found
        const s = dot(next, p);
        for (let i = 0; i < D; i++) next[i] -= s * p[i];
      }
      lambda = Math.sqrt(next.reduce((t, x) => t + x * x, 0));
      w = l2(next);
    }
    comps.push(w); vals.push(lambda);
  }
  return { comps, vals };
}
const KMAX = Math.max(...RANKS);
console.log(`fitting ${KMAX} linked-difference directions from ${diffs.length} pairs ...`);
const t0 = Date.now();
const { comps, vals } = topEigenvectors(diffs, KMAX);
const fitMs = Date.now() - t0;
console.log(`fitted in ${fitMs} ms; leading eigenvalues ${vals.slice(0, 5).map((v) => v.toFixed(4)).join(", ")}`);

// ---------- transforms ----------
const suppress = (v, k) => {                       // hard: project the directions out
  const out = v.slice();
  for (let c = 0; c < k; c++) {
    const s = dot(out, comps[c]);
    for (let i = 0; i < D; i++) out[i] -= s * comps[c][i];
  }
  return l2(out);
};
const rca = (v, k) => {                            // soft: shrink rather than delete
  const out = v.slice();
  for (let c = 0; c < k; c++) {
    const s = dot(out, comps[c]);
    const scale = 1 / Math.sqrt(vals[c] / vals[0] + EPS);
    for (let i = 0; i < D; i++) out[i] += s * (scale - 1) * comps[c][i];
  }
  return l2(out);
};

// ---------- evaluate ----------
function recallAt10(space) {
  const bySrc = new Map();
  for (const [a, b] of heldOut) {
    if (!bySrc.has(a)) bySrc.set(a, []); bySrc.get(a).push(b);
    if (!bySrc.has(b)) bySrc.set(b, []); bySrc.get(b).push(a);
  }
  let hit = 0, tot = 0;
  for (const [s, targets] of bySrc) {
    const scored = [];
    for (let t = 0; t < N; t++) { if (t === s || adj[s].has(t)) continue; scored.push([dot(space[s], space[t]), t]); }
    scored.sort((x, y) => y[0] - x[0]);
    const top = new Set(scored.slice(0, 10).map((x) => x[1]));
    for (const t of new Set(targets)) { if (adj[s].has(t)) continue; tot++; if (top.has(t)) hit++; }
  }
  return +(hit / tot).toFixed(4);
}

// ---------- does it COMPOSE with the structural channel? ----------
// The personal metric makes the CONTENT space better; graph fusion adds a
// different channel. If they stack, the architecture is additive; if not, they
// are finding the same signal twice.
const RA = (s, t) => { let a = 0; for (const x of adj[s]) if (adj[t].has(x)) a += 1 / Math.max(1, adj[x].size); return a; };
const zf = (arr) => { const m = arr.reduce((a, b) => a + b, 0) / arr.length; const sd = Math.sqrt(arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length) || 1; return (x) => (x - m) / sd; };
function recallFused(space) {
  const bySrc = new Map();
  for (const [a, b] of heldOut) {
    if (!bySrc.has(a)) bySrc.set(a, []); bySrc.get(a).push(b);
    if (!bySrc.has(b)) bySrc.set(b, []); bySrc.get(b).push(a);
  }
  let hit = 0, tot = 0;
  for (const [s, targets] of bySrc) {
    const cand = [];
    for (let t = 0; t < N; t++) { if (t === s || adj[s].has(t)) continue; cand.push(t); }
    const cs = cand.map((t) => dot(space[s], space[t])), ra = cand.map((t) => RA(s, t));
    const zc = zf(cs), zr = zf(ra);
    const ranked = cand.map((t, i) => [zc(cs[i]) + zr(ra[i]), t]).sort((x, y) => y[0] - x[0]);
    const top = new Set(ranked.slice(0, 10).map((x) => x[1]));
    for (const t of new Set(targets)) { if (adj[s].has(t)) continue; tot++; if (top.has(t)) hit++; }
  }
  return +(hit / tot).toFixed(4);
}

// ---------- learning curve: does it get better with MORE of the user's links? ----------
// This is the property that decides whether it is worth building: a metric that
// plateaus at 1000 links is a tweak, one still climbing is an architecture that
// compounds as the vault grows.
const curve = [];
for (const frac of [0.25, 0.5, 0.75, 1.0]) {
  const subset = diffs.slice(0, Math.max(8, Math.round(diffs.length * frac)));
  const { comps: cc, vals: vv } = topEigenvectors(subset, 16, 5);
  const proj = (v) => {
    const out = v.slice();
    for (let c = 0; c < 16; c++) { const sc = dot(out, cc[c]); for (let i = 0; i < D; i++) out[i] -= sc * cc[c][i]; }
    return l2(out);
  };
  curve.push({ trainLinks: subset.length, recall: recallAt10(C.map(proj)), topEigen: +vv[0].toFixed(4) });
}

const rows = [{ config: "frozen (today)", k: 0, recall: recallAt10(C) }];
for (const k of RANKS) {
  rows.push({ config: `suppress-${k}`, k, recall: recallAt10(C.map((v) => suppress(v, k))) });
  rows.push({ config: `rca-${k}`, k, recall: recallAt10(C.map((v) => rca(v, k))) });
}
rows.sort((a, b) => b.recall - a.recall);

console.log("\n==== PERSONAL METRIC (held-out link recall@10) ====");
console.log(`model ${MODEL}\n`);
for (const r of rows) {
  const base = rows.find((x) => x.k === 0).recall;
  const delta = r.k === 0 ? "" : `  ${r.recall >= base ? "+" : ""}${(r.recall - base).toFixed(4)}`;
  console.log(`  ${r.config.padEnd(18)} ${r.recall.toFixed(4)}${delta}`);
}
console.log("\n==== LEARNING CURVE (suppress-16) ====");
for (const c of curve) console.log(`  ${String(c.trainLinks).padStart(5)} links   recall ${c.recall.toFixed(4)}   leading eigenvalue ${c.topEigen}`);

const best = rows.filter((r) => r.k > 0).sort((a, b) => b.recall - a.recall)[0];
const bestSpace = best.config.startsWith("rca") ? C.map((v) => rca(v, best.k)) : C.map((v) => suppress(v, best.k));
const compose = { frozenFused: recallFused(C), personalFused: recallFused(bestSpace) };
console.log("\n==== COMPOSES WITH THE GRAPH CHANNEL? ====");
console.log(`  frozen + graph fusion    ${compose.frozenFused.toFixed(4)}`);
console.log(`  ${best.config} + graph fusion  ${compose.personalFused.toFixed(4)}   ${compose.personalFused >= compose.frozenFused ? "+" : ""}${(compose.personalFused - compose.frozenFused).toFixed(4)}`);

mkdirSync("/Users/justus/obsidian_atomized_intermediary/lab/results", { recursive: true });
writeFileSync(OUT, JSON.stringify({ model: MODEL, notes: N, edges: edges.length, trainEdges: train.length, fitMs, eigenvalues: vals, rows, curve, compose }, null, 1));
console.log("\nwrote", OUT);
