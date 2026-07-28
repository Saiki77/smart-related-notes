// THOROUGH structural channel study. Extends v5 with the families it did not cover,
// and splits every result by PROVENANCE, which is the check v5 was missing.
//
//   node bench/v6-channels.mjs [seeds]
//
// v5 measured eight channels on the expanded vault and found none of them passed the
// admission test. Half those edges are synthetic, written by agents told to make
// plausible associative links, so the finding needed a control: does it hold on the
// authentic half alone? Every number below is reported for ALL edges and for REAL-only
// edges, where both endpoints are notes from the user's own export.
//
// New channels over v5:
//   katz        weighted count of all paths, beta-damped        (diffusion family)
//   ppr         rooted PageRank, the thing 3.0 uses to NOMINATE (as a score this time)
//   heat        heat kernel diffusion, e^{-tL} approximated     (diffusion family)
//   simrank     one iteration: similar if neighbours are similar
//   lp          local path, CN plus a damped 3-hop term
//   pa          preferential attachment, deliberately a NULL: degree product only
//   community   label propagation, then same-community bonus
//   tdecayRA    RA over edges weighted by how recently both notes were written
//   triangle    pairs that close an open triangle, scored by how far apart in TIME
//               the two existing sides were written. The "triangle you never closed".
//
// v7 adds the CONTENT channel from the cached whole-note vectors, which is the only
// question that decides anything: PPR beat RA on structure alone, but the tags channel
// also looked useful alone and turned out to be re-describing what content already knew.
// A structural channel only earns a place if it survives NEXT TO content.
//
// pa is included on purpose. A channel that cannot beat degree-product is not finding
// structure, it is finding popularity, and several proposals in the backlog are at risk
// of exactly that.
import { readFileSync, readdirSync, statSync, writeFileSync, mkdirSync } from "node:fs";
import { join, relative } from "node:path";
import { fisherYates } from "./shuffle.mjs";

const VAULT = "/Users/justus/obsidian_atomized_intermediary/lab/vault";
const OUT = "/Users/justus/obsidian_atomized_intermediary/lab/results/v7-fusion.json";
const MODEL = "jinaai/jina-embeddings-v5-text-nano-text-matching";
const CACHE = join(process.env.HOME, ".cache/srn-lab", `v3-${MODEL.replace(/[^a-z0-9]/gi, "_")}.json`);
const SEEDS = Number(process.argv[2] ?? 8);
const HOLDOUT = 0.2, K = 10;

function walk(d, a = []) {
  for (const n of readdirSync(d)) {
    if (n.startsWith(".")) continue;
    const p = join(d, n);
    if (statSync(p).isDirectory()) walk(p, a); else if (n.endsWith(".md")) a.push(p);
  }
  return a;
}
const notes = walk(VAULT).map((abs) => {
  const rel = relative(VAULT, abs), raw = readFileSync(abs, "utf8");
  const front = (raw.match(/^---\n([\s\S]*?)\n---/) ?? [, ""])[1];
  const scalar = (k) => (front.match(new RegExp(`^${k}: *(.+)$`, "m")) ?? [])[1]?.trim() ?? null;
  const title = rel.slice(rel.lastIndexOf("/") + 1, -3);
  const ds = scalar("date") ?? (title.match(/\d{4}-\d{2}-\d{2}/) ?? [])[0] ?? null;
  return {
    title, real: !rel.startsWith("gen/"),
    day: ds ? Math.round(Date.parse(ds + "T00:00:00Z") / 86400000) : null,
    targets: [...raw.matchAll(/\[\[([^\]|#]+)/g)].map((m) => m[1].trim()),
  };
});
const byTitle = new Map();
notes.forEach((n, i) => { if (!byTitle.has(n.title)) byTitle.set(n.title, i); });
const N = notes.length;
const edgeSet = new Set();
for (let i = 0; i < N; i++) for (const t of notes[i].targets) {
  const j = byTitle.get(t);
  if (j === undefined || j === i) continue;
  edgeSet.add(i < j ? `${i},${j}` : `${j},${i}`);
}
const edges = [...edgeSet].map((s) => s.split(",").map(Number));
const isRealEdge = ([a, b]) => notes[a].real && notes[b].real;
// median day, used to date notes that carry none, so tdecayRA has something to work with
const days = notes.map((n) => n.day).filter((d) => d != null).sort((a, b) => a - b);
const MEDDAY = days[Math.floor(days.length / 2)] ?? 0;
const dayOf = (i) => notes[i].day ?? MEDDAY;

// ---- content channel: cosine over mean-centered whole-note vectors
const cache = JSON.parse(readFileSync(CACHE, "utf8"));
const strip = (r) => { const m = r.match(/^---\n[\s\S]*?\n---\n?/); return m ? r.slice(m[0].length) : r; };
const vecs = new Array(N).fill(null);
{
  const filesAgain = walk(VAULT);
  for (let i = 0; i < filesAgain.length; i++) {
    const f = filesAgain[i];
    const base = f.slice(f.lastIndexOf("/") + 1, -3);
    const key = (base + "\n\n" + strip(readFileSync(f, "utf8"))).slice(0, 8000);
    const idx = byTitle.get(base);
    const v = cache[key];
    if (v && idx !== undefined && vecs[idx] === null) vecs[idx] = Float64Array.from(v);
  }
  // mean-center, the way the plugin does, so the corpus noise floor goes away
  const present = vecs.filter(Boolean);
  const D = present[0]?.length ?? 0;
  const mu = new Float64Array(D);
  for (const v of present) for (let d = 0; d < D; d++) mu[d] += v[d] / present.length;
  for (const v of present) { let s2 = 0;
    for (let d = 0; d < D; d++) { v[d] -= mu[d]; s2 += v[d] * v[d]; }
    const n = Math.sqrt(s2) || 1; for (let d = 0; d < D; d++) v[d] /= n; }
  console.log(`content vectors: ${present.length}/${N} notes`);
}
// top-M by cosine, so content is a ranking not an O(N^2) dump
function chContent(i, adj) {
  const o = new Map(); const a = vecs[i];
  if (!a) return o;
  for (let j = 0; j < N; j++) {
    if (j === i || !vecs[j] || adj[i].has(j)) continue;
    let d = 0; const b = vecs[j];
    for (let k = 0; k < a.length; k++) d += a[k] * b[k];
    if (d > 0.15) o.set(j, d);
  }
  return o;
}

const buildAdj = (E) => { const a = Array.from({ length: N }, () => new Set());
  for (const [x, y] of E) { a[x].add(y); a[y].add(x); } return a; };

// ------------------------------------------------------------------ channels
const nb = (adj, i) => adj[i];
function chRA(i, adj) { const o = new Map();
  for (const x of nb(adj, i)) { const d = adj[x].size; if (!d) continue;
    for (const j of adj[x]) if (j !== i && !adj[i].has(j)) o.set(j, (o.get(j) ?? 0) + 1 / d); } return o; }
function chLP(i, adj) { const o = chCN(i, adj);
  for (const x of nb(adj, i)) for (const y of adj[x]) for (const j of adj[y])
    if (j !== i && !adj[i].has(j)) o.set(j, (o.get(j) ?? 0) + 0.01); return o; }
function chCN(i, adj) { const o = new Map();
  for (const x of nb(adj, i)) for (const j of adj[x]) if (j !== i && !adj[i].has(j)) o.set(j, (o.get(j) ?? 0) + 1); return o; }
// Katz: damped walks, truncated at 4 hops (beyond that beta^k is negligible at beta=.05)
function chKatz(i, adj) {
  let front = new Map([[i, 1]]); const o = new Map(); const BETA = 0.05;
  for (let h = 1; h <= 4; h++) {
    const next = new Map();
    for (const [u, w] of front) for (const v of adj[u]) next.set(v, (next.get(v) ?? 0) + w);
    for (const [v, w] of next) if (v !== i && !adj[i].has(v)) o.set(v, (o.get(v) ?? 0) + Math.pow(BETA, h) * w);
    front = next;
  }
  return o;
}
// rooted PageRank, power iteration with restart
function chPPR(i, adj) {
  const ALPHA = 0.15, STEPS = 6;
  let p = new Map([[i, 1]]);
  const acc = new Map();
  for (let s = 0; s < STEPS; s++) {
    const q = new Map();
    for (const [u, w] of p) { const d = adj[u].size; if (!d) continue;
      const share = (w * (1 - ALPHA)) / d;
      for (const v of adj[u]) q.set(v, (q.get(v) ?? 0) + share); }
    q.set(i, (q.get(i) ?? 0) + ALPHA);
    p = q;
    for (const [v, w] of p) if (v !== i && !adj[i].has(v)) acc.set(v, (acc.get(v) ?? 0) + w);
  }
  return acc;
}
// heat kernel: sum_k (-t)^k L^k / k!, truncated; approximated on the normalised walk
function chHeat(i, adj) {
  const T = 2, TERMS = 4;
  let cur = new Map([[i, 1]]); const o = new Map(); let coef = 1;
  for (let k = 1; k <= TERMS; k++) {
    const nx = new Map();
    for (const [u, w] of cur) { const d = adj[u].size || 1;
      for (const v of adj[u]) nx.set(v, (nx.get(v) ?? 0) + w / d); }
    coef = (coef * T) / k;
    for (const [v, w] of nx) if (v !== i && !adj[i].has(v)) o.set(v, (o.get(v) ?? 0) + Math.exp(-T) * coef * w);
    cur = nx;
  }
  return o;
}
// SimRank, single iteration seeded on common neighbours
function chSimRank(i, adj) {
  const o = new Map(); const C = 0.8;
  for (const x of nb(adj, i)) for (const j of adj[x]) {
    if (j === i || adj[i].has(j)) continue;
    const di = adj[i].size || 1, dj = adj[j].size || 1;
    o.set(j, (o.get(j) ?? 0) + C / (di * dj));
  }
  return o;
}
function chPA(i, adj) { const o = new Map();
  for (const x of nb(adj, i)) for (const j of adj[x]) if (j !== i && !adj[i].has(j)) o.set(j, adj[i].size * adj[j].size);
  return o; }
// label propagation communities, recomputed per split
let COMM = null;
function communities(adj) {
  const lab = Array.from({ length: N }, (_, i) => i);
  for (let it = 0; it < 6; it++) {
    for (let i = 0; i < N; i++) {
      const c = new Map();
      for (const j of adj[i]) c.set(lab[j], (c.get(lab[j]) ?? 0) + 1);
      let best = lab[i], bn = -1;
      for (const [l, n] of c) if (n > bn) { bn = n; best = l; }
      lab[i] = best;
    }
  }
  return lab;
}
function chCommunity(i, adj) {
  const o = new Map(); const size = new Map();
  for (const l of COMM) size.set(l, (size.get(l) ?? 0) + 1);
  for (let j = 0; j < N; j++) if (j !== i && !adj[i].has(j) && COMM[j] === COMM[i])
    o.set(j, 1 / Math.log(1 + (size.get(COMM[i]) ?? 2)));
  return o;
}
// RA over edges weighted by recency: a shared neighbour you both linked recently counts
// for more than one from two years ago.
function chTdecayRA(i, adj) {
  const o = new Map();
  for (const x of nb(adj, i)) { const d = adj[x].size; if (!d) continue;
    for (const j of adj[x]) {
      if (j === i || adj[i].has(j)) continue;
      const gap = Math.abs(dayOf(i) - dayOf(j));
      o.set(j, (o.get(j) ?? 0) + (1 / d) * Math.exp(-gap / 400));
    } }
  return o;
}
// "The triangle you never closed", scored by how far apart the two existing sides were
// written. A triple assembled across a year is a gap; one from a single afternoon is not.
function chTriangle(i, adj) {
  const o = new Map();
  for (const x of nb(adj, i)) for (const j of adj[x]) {
    if (j === i || adj[i].has(j)) continue;
    const spread = Math.abs(dayOf(i) - dayOf(j)) + Math.abs(dayOf(i) - dayOf(x));
    o.set(j, Math.max(o.get(j) ?? 0, Math.log(1 + spread) / (adj[x].size || 1)));
  }
  return o;
}
const CH = { content: chContent, RA: chRA, katz: chKatz, ppr: chPPR, heat: chHeat, simrank: chSimRank,
             lp: chLP, pa: chPA, community: chCommunity, tdecayRA: chTdecayRA, triangle: chTriangle };
const NAMES = Object.keys(CH);

// ------------------------------------------------------------------ scoring
const lcg = (s0) => { let s = s0 >>> 0 || 1; return () => ((s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff); };
function znorm(m) { const v = [...m.values()]; if (v.length < 2) return new Map();
  const mu = v.reduce((a, b) => a + b, 0) / v.length;
  const sd = Math.sqrt(v.reduce((a, b) => a + (b - mu) ** 2, 0) / v.length) || 1;
  return new Map([...m].map(([k, x]) => [k, (x - mu) / sd])); }
function evaluate(f, test, adj) {
  const bySrc = new Map();
  for (const [a, b] of test) { (bySrc.get(a) ?? bySrc.set(a, []).get(a)).push(b);
                               (bySrc.get(b) ?? bySrc.set(b, []).get(b)).push(a); }
  let hit = 0, tot = 0, fired = 0, srcs = 0, chit = 0, ctot = 0;
  for (const [i, want] of bySrc) {
    srcs++;
    const sc = f(i, adj);
    const top = [...sc.entries()].sort((x, y) => y[1] - x[1]).slice(0, K).map(([j]) => j);
    if (sc.size) fired++;
    for (const w of want) { tot++; if (top.includes(w)) hit++;
      if (sc.size) { ctot++; if (top.includes(w)) chit++; } }
  }
  return { recall: tot ? hit / tot : 0, fires: srcs ? fired / srcs : 0, cond: ctot ? chit / ctot : 0 };
}
const push = (o, k, v) => { (o[k] ??= []).push(v); };
const acc = { all: {}, real: {}, marg: {}, margReal: {}, fires: {}, cond: {} };

for (let s = 0; s < SEEDS; s++) {
  const rnd = lcg(20260728 + s * 7919);
  const sh = fisherYates(edges, rnd);
  const cut = Math.floor(sh.length * HOLDOUT);
  const test = sh.slice(0, cut), train = sh.slice(cut);
  const testReal = test.filter(isRealEdge);
  const adj = buildAdj(train);
  COMM = communities(adj);

  const base = evaluate((i, a) => znorm(CH.content(i, a)), test, adj).recall;
  const baseReal = evaluate((i, a) => znorm(CH.content(i, a)), testReal, adj).recall;
  for (const n of NAMES) {
    const e = evaluate(CH[n], test, adj);
    push(acc.all, n, e.recall); push(acc.fires, n, e.fires); push(acc.cond, n, e.cond);
    push(acc.real, n, evaluate(CH[n], testReal, adj).recall);
    if (n === "content") continue;
    const fus = (i, a) => { const o = new Map(znorm(CH.content(i, a)));
      for (const [k, v] of znorm(CH[n](i, a))) o.set(k, (o.get(k) ?? 0) + v); return o; };
    push(acc.marg, n, evaluate(fus, test, adj).recall - base);
    push(acc.margReal, n, evaluate(fus, testReal, adj).recall - baseReal);
  }
  // the combinations that actually decide what ships
  const mix = (...names) => (i, a) => { const o = new Map();
    for (const nm of names) for (const [k, v] of znorm(CH[nm](i, a))) o.set(k, (o.get(k) ?? 0) + v);
    return o; };
  for (const combo of [["content"], ["content","RA"], ["content","ppr"], ["content","RA","ppr"],
                       ["content","RA","ppr","heat"], ["RA","ppr"]]) {
    const key = combo.join("+");
    push(acc.combo ??= {}, key, evaluate(mix(...combo), test, adj).recall);
    push(acc.comboReal ??= {}, key, evaluate(mix(...combo), testReal, adj).recall);
  }
}
const stat = (a) => { const mu = a.reduce((x, y) => x + y, 0) / a.length;
  return { mean: +mu.toFixed(4), sd: +Math.sqrt(a.reduce((x, y) => x + (y - mu) ** 2, 0) / a.length).toFixed(4) }; };
const S = (o) => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, stat(v)]));
const all = S(acc.all), real = S(acc.real), marg = S(acc.marg), margReal = S(acc.margReal),
      fires = S(acc.fires), cond = S(acc.cond);

const nReal = edges.filter(isRealEdge).length;
console.log(`\n==== THOROUGH STRUCTURAL STUDY ====`);
console.log(`${N} notes | ${edges.length} edges, of which ${nReal} are real-to-real | ${SEEDS} seeds | recall@${K}\n`);
console.log("  channel alone            ALL edges        REAL-only edges");
for (const n of NAMES.sort((a, b) => all[b].mean - all[a].mean))
  console.log(`    ${n.padEnd(12)} ${all[n].mean.toFixed(4)} +- ${all[n].sd.toFixed(4)}   ${real[n].mean.toFixed(4)} +- ${real[n].sd.toFixed(4)}`);
console.log("\n  MARGINAL on top of CONTENT    ALL              REAL-only          verdict (2 SD)");
for (const n of Object.keys(marg).sort((a, b) => marg[b].mean - marg[a].mean)) {
  const v = marg[n].mean > 2 * marg[n].sd ? "PASSES" : marg[n].mean > 0 ? "noise" : "hurts";
  const vr = margReal[n].mean > 2 * margReal[n].sd ? "PASSES" : margReal[n].mean > 0 ? "noise" : "hurts";
  console.log(`    ${n.padEnd(12)} ${(marg[n].mean >= 0 ? "+" : "") + marg[n].mean.toFixed(4)} +-${marg[n].sd.toFixed(4)}   ${(margReal[n].mean >= 0 ? "+" : "") + margReal[n].mean.toFixed(4)} +-${margReal[n].sd.toFixed(4)}   ${v} / ${vr}`);
}
const combo = S(acc.combo), comboReal = S(acc.comboReal);
console.log("\n  COMBINATIONS            ALL              REAL-only");
for (const k of Object.keys(combo).sort((a, b) => combo[b].mean - combo[a].mean))
  console.log(`    ${k.padEnd(24)} ${combo[k].mean.toFixed(4)} +-${combo[k].sd.toFixed(4)}   ${comboReal[k].mean.toFixed(4)} +-${comboReal[k].sd.toFixed(4)}`);
console.log("\n  coverage: fires on / recall where it fires");
for (const n of NAMES.sort((a, b) => cond[b].mean - cond[a].mean))
  console.log(`    ${n.padEnd(12)} ${(fires[n].mean * 100).toFixed(0).padStart(3)}%   ${cond[n].mean.toFixed(4)}`);

mkdirSync("/Users/justus/obsidian_atomized_intermediary/lab/results", { recursive: true });
writeFileSync(OUT, JSON.stringify({ notes: N, edges: edges.length, realEdges: nReal, seeds: SEEDS,
  all, real, combo: acc.combo, comboReal: acc.comboReal, marginal: marg, marginalReal: margReal, fires, cond }, null, 1));
console.log("\nwrote", OUT);
