// MUTUAL PROXIMITY: does calibrating the score help, and does it cost recall?
//
// The remaining v3 item. Two separate claims need separate tests, because a
// calibration that improves comparability but costs recall is not worth having:
//
//   1 RECALL     does MP rank better or worse than raw centered cosine?
//                (CSLS, the other hubness correction, cost jina 0.030 - so the
//                 prior here is genuinely uncertain, not obviously positive)
//   2 CALIBRATION does a FIXED threshold keep its meaning as the vault grows?
//                This is the actual motivation: minSimilarity currently means
//                something different in a 200-note vault than a 20,000-note one.
//
// MP(x,y) = F_x(s_xy) * F_y(s_yx), where F_x is the empirical CDF of x's own
// similarity distribution. Each side asks "what fraction of the vault is FURTHER
// from x than y is", so the output is a probability by construction and a note
// that is everyone's neighbour stops dominating.
//
//   node bench/v3-calibration.mjs
import { readFileSync, writeFileSync, readdirSync, statSync, mkdirSync } from "node:fs";
import { join, relative } from "node:path";
import { fisherYates } from "./shuffle.mjs";

const VAULT = process.env.LAB_VAULT || "/Users/justus/obsidian_atomized_intermediary/lab/vault";
const OUT = "/Users/justus/obsidian_atomized_intermediary/lab/results/v3-calibration.json";
const MODEL = process.env.LAB_MODEL || "jinaai/jina-embeddings-v5-text-nano-text-matching";
const CACHE = join(process.env.HOME, ".cache/srn-lab", `v3-${MODEL.replace(/[^a-z0-9]/gi, "_")}.json`);
const SAMPLE = Number(process.env.CAL_SAMPLE ?? 400);

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

// ---------- links + holdout ----------
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

// ---------- full similarity matrix (494 notes: fine to materialise) ----------
const S = [];
for (let i = 0; i < N; i++) { const row = new Float64Array(N); for (let j = 0; j < N; j++) row[j] = i === j ? -Infinity : dot(C[i], C[j]); S.push(row); }

// Empirical CDF per note. sortedAsc[i] lets us ask "what fraction of the vault is
// LESS similar to i than s is" by binary search.
const sortedAsc = S.map((row) => { const a = Array.from(row).filter((x) => isFinite(x)); a.sort((p, q) => p - q); return a; });
const cdf = (i, s) => {
  const a = sortedAsc[i];
  let lo = 0, hi = a.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (a[mid] < s) lo = mid + 1; else hi = mid; }
  return lo / a.length;
};
const mp = (i, j) => cdf(i, S[i][j]) * cdf(j, S[j][i]);

// Sampled MP: the shippable form. Each note keeps the CDF of its similarities to
// a fixed random sample rather than to the whole vault.
const rs = mulberry32(3);
const sampleIdx = [];
{
  const pool = [...Array(N).keys()];
  for (let i = 0; i < Math.min(SAMPLE, N); i++) sampleIdx.push(pool.splice(Math.floor(rs() * pool.length), 1)[0]);
}
const sortedSample = S.map((row) => { const a = sampleIdx.map((j) => row[j]).filter((x) => isFinite(x)); a.sort((p, q) => p - q); return a; });
const cdfS = (i, s) => {
  const a = sortedSample[i];
  let lo = 0, hi = a.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (a[mid] < s) lo = mid + 1; else hi = mid; }
  return lo / a.length;
};
const mpS = (i, j) => cdfS(i, S[i][j]) * cdfS(j, S[j][i]);

// ---------- 1. recall ----------
const bySrc = new Map();
for (const [a, b] of heldOut) {
  if (!bySrc.has(a)) bySrc.set(a, []); bySrc.get(a).push(b);
  if (!bySrc.has(b)) bySrc.set(b, []); bySrc.get(b).push(a);
}
const RA = (s, t) => { let a = 0; for (const x of adj[s]) if (adj[t].has(x)) a += 1 / Math.max(1, adj[x].size); return a; };
const zf = (arr) => { const m = arr.reduce((a, b) => a + b, 0) / arr.length; const sd = Math.sqrt(arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length) || 1; return (x) => (x - m) / sd; };
function recall(scoreFn, withGraph) {
  let hit = 0, tot = 0;
  const top1 = new Array(N).fill(0);
  for (const [s, targets] of bySrc) {
    const cand = [];
    for (let t = 0; t < N; t++) { if (t === s || adj[s].has(t)) continue; cand.push(t); }
    let ranked;
    if (withGraph) {
      const cs = cand.map((t) => scoreFn(s, t)), ra = cand.map((t) => RA(s, t));
      const zc = zf(cs), zr = zf(ra);
      ranked = cand.map((t, i) => [zc(cs[i]) + zr(ra[i]), t]).sort((x, y) => y[0] - x[0]);
    } else {
      ranked = cand.map((t) => [scoreFn(s, t), t]).sort((x, y) => y[0] - x[0]);
    }
    if (ranked.length) top1[ranked[0][1]]++;
    const top = new Set(ranked.slice(0, 10).map((x) => x[1]));
    for (const t of new Set(targets)) { if (adj[s].has(t)) continue; tot++; if (top.has(t)) hit++; }
  }
  const m = top1.reduce((a, b) => a + b, 0) / N;
  const sd = Math.sqrt(top1.reduce((a, x) => a + (x - m) ** 2, 0) / N) || 1;
  const skew = top1.reduce((a, x) => a + ((x - m) / sd) ** 3, 0) / N;
  return { recall: +(hit / tot).toFixed(4), hubSkew: +skew.toFixed(3) };
}
const cos = (i, j) => S[i][j];
const recallRows = [
  { config: "cosine (today)", ...recall(cos, false) },
  { config: "MP exact", ...recall(mp, false) },
  { config: `MP sampled(${SAMPLE})`, ...recall(mpS, false) },
  { config: "cosine + graph (3.0 shipped)", ...recall(cos, true) },
  { config: "MP exact + graph", ...recall(mp, true) },
  { config: `MP sampled(${SAMPLE}) + graph`, ...recall(mpS, true) },
];

// ---------- 2. calibration ----------
// The real question: if a user sets a threshold, does it keep meaning the same
// thing as the vault grows? Take nested vault sizes and, for a FIXED threshold,
// record what fraction of pairs survive. A calibrated score holds that fraction
// steady; an uncalibrated one drifts.
function survivalCurve(scoreFn, threshold, sizes) {
  const out = [];
  for (const size of sizes) {
    let kept = 0, total = 0;
    for (let i = 0; i < size; i++) {
      for (let j = i + 1; j < size; j++) { total++; if (scoreFn(i, j) >= threshold) kept++; }
    }
    out.push({ vaultSize: size, keptFraction: +(kept / total).toFixed(4) });
  }
  return out;
}
const sizes = [100, 200, 350, N];
const drift = (curve) => {
  const v = curve.map((c) => c.keptFraction);
  return +(Math.max(...v) - Math.min(...v)).toFixed(4);
};
const cosCurve = survivalCurve(cos, 0.2, sizes);          // the plugin's default minSimilarity
const mpCurve = survivalCurve(mp, 0.8, sizes);            // an equivalently selective MP cut
const calib = { cosine: { threshold: 0.2, curve: cosCurve, drift: drift(cosCurve) },
                mp: { threshold: 0.8, curve: mpCurve, drift: drift(mpCurve) } };

console.log(`\n==== MUTUAL PROXIMITY ====`);
console.log(`model ${MODEL} | ${N} notes | ${heldOut.length} held-out links\n`);
console.log("1. RECALL");
for (const r of recallRows) console.log(`   ${r.config.padEnd(30)} recall ${r.recall.toFixed(4)}   hub-skew ${String(r.hubSkew).padStart(7)}`);
console.log("\n2. CALIBRATION (does a fixed threshold keep its meaning as the vault grows?)");
console.log(`   cosine >= 0.2 :  ${cosCurve.map((c) => `${c.vaultSize}n ${c.keptFraction}`).join("   ")}   drift ${calib.cosine.drift}`);
console.log(`   MP     >= 0.8 :  ${mpCurve.map((c) => `${c.vaultSize}n ${c.keptFraction}`).join("   ")}   drift ${calib.mp.drift}`);
mkdirSync("/Users/justus/obsidian_atomized_intermediary/lab/results", { recursive: true });
writeFileSync(OUT, JSON.stringify({ model: MODEL, notes: N, recallRows, calib }, null, 1));
console.log("\nwrote", OUT);
