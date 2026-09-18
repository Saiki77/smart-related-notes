// How should regions be FORMED? The shipped map is hard spherical k-means over note vectors:
// one region per note, topic-only, k <= 16, recomputed cold. This probe tests the alternative
// on the lab vault: a fused NOTE GRAPH (semantic kNN + wikilinks + hub co-citation, optionally
// folders / tags as a prior) clustered into OVERLAPPING communities via ego-splitting
// (Epasto, Lattanzi, Paes Leme, KDD'17) + Louvain.
//
// Ground truth = groupings the user made themselves, which overlap: tags (incl. small
// project-like ones such as a trip or a work project), leaf folders, MOC member lists.
// A family is only scored when it was NOT an input channel (no circular wins).
//   [1] quality     best-match F1 per GT group, by method and by which channels exist
//   [2] plant       a 14-note "thesis" drawn from different folders, bound only by a hub +
//                   co-editing: does it become its own small region WITHOUT leaving its topics?
//   [3] incremental file 10% new notes locally vs a full recompute
//   node --experimental-strip-types bench/region-formation-probe.mjs
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative, dirname } from "node:path";
import { buildVaultMap } from "../src/vault-map.ts";

const VAULT = "/Users/justus/obsidian_atomized_intermediary/lab/vault";
const MODEL = process.env.LAB_MODEL || "jinaai/jina-embeddings-v5-text-nano-text-matching";
const CACHE = join(process.env.HOME, ".cache/srn-lab", `v3-${MODEL.replace(/[^a-z0-9]/gi, "_")}.json`);

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

function tagsOf(raw) {
  const fm = (raw.match(/^---\n([\s\S]*?)\n---/) || [, ""])[1];
  let out = [], m = fm.match(/^tags:\s*\[(.*?)\]/m);
  if (m) out = m[1].split(",");
  else { m = fm.match(/^tags:\s*\n((?:[ \t]*-[ \t]*.+\n?)+)/m); if (m) out = m[1].split("\n").map((l) => l.replace(/^[ \t]*-[ \t]*/, "")); }
  return [...new Set(out.map((t) => t.trim().replace(/^["'#]+|["']+$/g, "").toLowerCase()).filter(Boolean))];
}
const linksOf = (raw) => { const s = new Set(); for (const m of raw.matchAll(/\[\[([^\]]+)\]\]/g)) s.add(m[1].split("|")[0].split("#")[0].trim().toLowerCase()); return s; };

const notes = [];
for (const abs of walk(VAULT)) {
  const rel = relative(VAULT, abs);
  if (rel.startsWith("real/") && !INDEXABLE.has(rel)) continue;
  const basename = rel.replace(/\.md$/, "").split("/").pop();
  if (/(^|\/)Attachments\//.test(rel) || /\.dup$/.test(basename) || /^Vault Insights/.test(basename)) continue;
  const raw = readFileSync(abs, "utf8");
  const v = cache[noteText({ basename, body: stripFront(raw) })];
  if (v) notes.push({ rel, basename, vec: v, raw });
}
const N = notes.length, D = notes[0].vec.length;
const norm = (v) => { let s = 0; for (const x of v) s += x * x; s = Math.sqrt(s) || 1; const o = new Float32Array(v.length); for (let i = 0; i < v.length; i++) o[i] = v[i] / s; return o; };
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const mean = new Float32Array(D);
for (const n of notes) for (let i = 0; i < D; i++) mean[i] += n.vec[i] / N;
const V = notes.map((n) => norm(n.vec.map((x, i) => x - mean[i])));
let seed = 918; const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };

// ---------- structure the user made ----------
const byBase = new Map(); notes.forEach((n, i) => { const k = n.basename.toLowerCase(); if (!byBase.has(k)) byBase.set(k, i); });
const out = notes.map((n, i) => [...linksOf(n.raw)].map((l) => byBase.get(l)).filter((j) => j !== undefined && j !== i));
const tags = notes.map((n) => tagsOf(n.raw));
const folder = notes.map((n) => dirname(n.rel));
const isHub = notes.map((n, i) => /\bMOC\b|uebersicht|übersicht|\bindex\b/i.test(n.basename) || out[i].length >= 15);

const group = (keyFn, lo, hi) => { const m = new Map(); notes.forEach((_, i) => { for (const k of keyFn(i)) { if (!m.has(k)) m.set(k, new Set()); m.get(k).add(i); } }); return [...m.entries()].filter(([, s]) => s.size >= lo && s.size <= hi).map(([name, set]) => ({ name, set })); };
const GT = {
  tags: group((i) => tags[i], 8, 80),
  folders: group((i) => [folder[i]], 8, 400),
  mocs: notes.map((n, i) => ({ name: n.basename, set: new Set(out[i]) })).filter((g, i) => isHub[i] && g.set.size >= 8),
};
GT.small = GT.tags.filter((g) => g.set.size <= 25);

// ---------- semantic kNN (the only channel every vault has) ----------
const KNN = 10;
const knn = V.map((v, i) => { const s = []; for (let j = 0; j < N; j++) if (j !== i) s.push([dot(v, V[j]), j]); s.sort((a, b) => b[0] - a[0]); return s.slice(0, KNN); });
const inKnn = knn.map((l) => new Set(l.map((x) => x[1])));

function buildGraph(n, ch, extra = [], hide = null) { // ch = { links, hubs, folders, tags }; returns array of Map (weighted, undirected)
  const adj = Array.from({ length: n }, () => new Map());
  const add = (a, b, w) => { if (a === b || a >= n || b >= n || w <= 0) return; adj[a].set(b, (adj[a].get(b) || 0) + w); adj[b].set(a, (adj[b].get(a) || 0) + w); };
  for (let i = 0; i < n; i++) for (const [s, j] of knn[i]) { if (j >= n) continue; if (inKnn[j].has(i)) { if (i < j) add(i, j, s); } else add(i, j, 0.5 * s); }
  if (ch.links) for (let i = 0; i < n; i++) { if (isHub[i] && !ch.hubs) continue; for (const j of out[i]) if (!(isHub[j] && !ch.hubs) && !(hide && hide.has(Math.min(i, j) + ":" + Math.max(i, j)))) add(i, j, isHub[i] || isHub[j] ? 0.3 : 1.0); }
  if (ch.hubs) for (let h = 0; h < n; h++) if (isHub[h]) { const m = out[h].filter((j) => j < n), w = 1 / Math.log2(1 + m.length); for (let a = 0; a < m.length; a++) for (let b = a + 1; b < m.length; b++) add(m[a], m[b], w); }
  const prior = (groups, wSmall) => { for (const g of groups) { const m = [...g].filter((i) => i < n); if (m.length < 2) continue; if (m.length <= 30) for (let a = 0; a < m.length; a++) for (let b = a + 1; b < m.length; b++) add(m[a], m[b], wSmall); } };
  if (ch.folders) prior(group((i) => [folder[i]], 2, 1e9).map((g) => g.set), 0.25);
  if (ch.tags) prior(group((i) => tags[i], 2, 1e9).map((g) => g.set), 0.2);
  for (const [a, b, w] of extra) add(a, b, w);
  return adj;
}

// ---------- Louvain (deterministic order, resolution gamma) ----------
function louvain(n, adjIn, gamma = 1) {
  let adj = adjIn.map((m) => new Map(m)), self = new Array(n).fill(0), nodeOf = Array.from({ length: n }, (_, i) => i);
  for (let level = 0; level < 12; level++) {
    const M = adj.length;
    const deg = adj.map((m, i) => { let s = 2 * self[i]; for (const w of m.values()) s += w; return s; });
    const m2 = deg.reduce((a, b) => a + b, 0); if (m2 === 0) break;
    const comm = Array.from({ length: M }, (_, i) => i), tot = deg.slice();
    let improved = false;
    for (let pass = 0, moved = true; moved && pass < 40; pass++) {
      moved = false;
      for (let i = 0; i < M; i++) {
        const ci = comm[i], wTo = new Map();
        for (const [j, w] of adj[i]) wTo.set(comm[j], (wTo.get(comm[j]) || 0) + w);
        tot[ci] -= deg[i];
        let best = ci, bestGain = (wTo.get(ci) || 0) - gamma * tot[ci] * deg[i] / m2;
        for (const [c, w] of wTo) { const g = w - gamma * tot[c] * deg[i] / m2; if (g > bestGain + 1e-12) { bestGain = g; best = c; } }
        tot[best] += deg[i];
        if (best !== ci) { comm[i] = best; moved = true; improved = true; }
      }
    }
    if (!improved) break;
    const ids = new Map(); for (let i = 0; i < M; i++) if (!ids.has(comm[i])) ids.set(comm[i], ids.size);
    const nadj = Array.from({ length: ids.size }, () => new Map()), nself = new Array(ids.size).fill(0);
    for (let i = 0; i < M; i++) { const a = ids.get(comm[i]); nself[a] += self[i]; for (const [j, w] of adj[i]) { const b = ids.get(comm[j]); if (a === b) nself[a] += w / 2; else nadj[a].set(b, (nadj[a].get(b) || 0) + w); } }
    nodeOf = nodeOf.map((s) => ids.get(comm[s]));
    if (ids.size === M) break;
    adj = nadj; self = nself;
  }
  return nodeOf;
}

// ---------- ego-splitting: a note gets one persona per distinct neighbourhood ----------
const MIN_PERSONA = 3;
function personasOf(u, adj) { // neighbour -> local persona id
  const nb = [...adj[u].keys()], res = new Map();
  if (nb.length < 2 * MIN_PERSONA) { for (const v of nb) res.set(v, 0); return res; }
  const idx = new Map(nb.map((v, i) => [v, i])), ego = nb.map(() => new Map());
  for (const v of nb) for (const [x, w] of adj[v]) if (idx.has(x)) ego[idx.get(v)].set(idx.get(x), w);
  const part = louvain(nb.length, ego, 1), size = new Map();
  for (const p of part) size.set(p, (size.get(p) || 0) + 1);
  const main = [...size.entries()].sort((a, b) => b[1] - a[1])[0][0], remap = new Map();
  for (const [p, s] of size) remap.set(p, s >= MIN_PERSONA ? p : main);
  const local = new Map(); nb.forEach((v, i) => { const p = remap.get(part[i]); if (!local.has(p)) local.set(p, local.size); res.set(v, local.get(p)); });
  return res;
}
function egoSplitCover(n, adj, gamma, minSize = 4) {
  const per = Array.from({ length: n }, (_, u) => personasOf(u, adj)), base = [0];
  for (let u = 0; u < n; u++) base.push(base[u] + Math.max(1, new Set(per[u].values()).size));
  const P = base[n], padj = Array.from({ length: P }, () => new Map());
  for (let u = 0; u < n; u++) for (const [v, w] of adj[u]) if (u < v) { const a = base[u] + per[u].get(v), b = base[v] + per[v].get(u); padj[a].set(b, (padj[a].get(b) || 0) + w); padj[b].set(a, (padj[b].get(a) || 0) + w); }
  const pc = louvain(P, padj, gamma), members = new Map();
  for (let u = 0; u < n; u++) for (let p = base[u]; p < base[u + 1]; p++) { if (!members.has(pc[p])) members.set(pc[p], new Set()); members.get(pc[p]).add(u); }
  return [...members.values()].filter((s) => s.size >= minSize);
}
const hardCover = (assign, minSize = 4) => { const m = new Map(); assign.forEach((c, i) => { if (!m.has(c)) m.set(c, new Set()); m.get(c).add(i); }); return [...m.values()].filter((s) => s.size >= minSize); };

// ---------- k-means baselines (16 = shipped code; 40 = same algorithm, fairer to small groups) ----------
function kmeans(k, iters = 30) {
  const C = [V[Math.floor(rnd() * N)]];
  while (C.length < k) { const d = V.map((v) => Math.max(0, 1 - Math.max(...C.map((c) => dot(v, c))))), t = d.reduce((a, b) => a + b, 0); let p = rnd() * t, i = 0; for (; i < N - 1 && p > d[i]; i++) p -= d[i]; C.push(V[i]); }
  let cent = C.map((c) => Float32Array.from(c)), assign = new Array(N).fill(0);
  for (let it = 0; it < iters; it++) {
    assign = V.map((v) => { let b = 0, bs = -2; cent.forEach((c, j) => { const s = dot(v, c); if (s > bs) { bs = s; b = j; } }); return b; });
    const sums = cent.map(() => new Float32Array(D)); assign.forEach((c, i) => { for (let d = 0; d < D; d++) sums[c][d] += V[i][d]; });
    cent = sums.map((s, j) => (s.some((x) => x !== 0) ? norm(s) : cent[j]));
  }
  return { assign, cent };
}
const softCover = (cent, delta = 0.05) => { const sets = cent.map(() => new Set()); V.forEach((v, i) => { const s = cent.map((c) => dot(v, c)), best = Math.max(...s); s.forEach((x, j) => { if (x >= best - delta) sets[j].add(i); }); }); return sets.filter((s) => s.size >= 4); };

// ---------- scoring ----------
const f1 = (a, b) => { let inter = 0; for (const x of a) if (b.has(x)) inter++; return inter === 0 ? 0 : (2 * inter) / (a.size + b.size); };
const bestF1 = (gt, cover) => gt.length === 0 ? NaN : gt.reduce((s, g) => s + Math.max(0, ...cover.map((c) => f1(g.set, c))), 0) / gt.length;
function describe(cover, n = N) {
  const count = new Array(n).fill(0); for (const c of cover) for (const i of c) count[i]++;
  const sizes = cover.map((c) => c.size).sort((a, b) => a - b);
  return { regions: cover.length, median: sizes[Math.floor(sizes.length / 2)] || 0, smallest: sizes[0] || 0, multi: count.filter((x) => x >= 2).length / n, loose: count.filter((x) => x === 0).length / n };
}
const fmt = (x) => (Number.isNaN(x) ? "  –  " : x.toFixed(3));
function row(name, cover, skip = {}) {
  const d = describe(cover);
  console.log(`  ${name.padEnd(34)} ${String(d.regions).padStart(4)} ${String(d.median).padStart(5)} ${(100 * d.multi).toFixed(0).padStart(5)}% ${(100 * d.loose).toFixed(0).padStart(5)}%   ${skip.tags ? "  –  " : fmt(bestF1(GT.tags, cover))}  ${skip.tags ? "  –  " : fmt(bestF1(GT.small, cover))}  ${skip.folders ? "  –  " : fmt(bestF1(GT.folders, cover))}  ${skip.mocs ? "  –  " : fmt(bestF1(GT.mocs, cover))}`);
}

console.log(`\n==== REGION FORMATION PROBE  (${MODEL}) ====`);
console.log(`${N} notes · ${isHub.filter(Boolean).length} hubs · GT groups: ${GT.tags.length} tags (${GT.small.length} small, 8-25 notes) · ${GT.folders.length} folders · ${GT.mocs.length} MOCs`);
console.log(`\n[1] best-match F1 against the user's own groups (higher = "my group shows up as a region"). – = channel was an input`);
console.log(`  ${"method".padEnd(34)} ${"regs".padStart(4)} ${"med".padStart(5)} ${"in>=2".padStart(6)} ${"loose".padStart(6)}   tags   small  folder  MOC`);
const shipped = buildVaultMap(notes.map((n, i) => ({ path: n.rel, title: n.basename, vec: V[i] })));
const shippedAssign = notes.map((n) => shipped.points.find((p) => p.path === n.rel).cluster);
row("k-means 16, hard (SHIPPED)", hardCover(shippedAssign));
const km16 = kmeans(16), km40 = kmeans(40);
row("k-means 16, soft 0.05", softCover(km16.cent));
row("k-means 40, hard", hardCover(km40.assign));
row("k-means 40, soft 0.05", softCover(km40.cent));
const CONFIGS = [
  ["no links, no folders (text only)", {}, {}],
  ["+ links", { links: 1 }, {}],
  ["+ links + hubs", { links: 1, hubs: 1 }, { mocs: 1 }],
  ["+ links + hubs + folders", { links: 1, hubs: 1, folders: 1 }, { mocs: 1, folders: 1 }],
  ["+ links + hubs + folders + tags", { links: 1, hubs: 1, folders: 1, tags: 1 }, { mocs: 1, folders: 1, tags: 1 }],
];
const timing = {};
for (const [label, ch, skip] of CONFIGS) {
  console.log(`  -- graph: ${label}`);
  const adj = buildGraph(N, ch);
  for (const g of [1, 3]) row(`   louvain, hard, gamma ${g}`, hardCover(louvain(N, adj, g)), skip);
  for (const g of [1, 3]) { const t0 = Date.now(); const c = egoSplitCover(N, adj, g); timing[`${label} g${g}`] = Date.now() - t0; row(`   ego-split, overlapping, gamma ${g}`, c, skip); }
}
console.log(`  full ego-split build times (ms): ${Object.values(timing).join(", ")}`);

// ---------- [2] the planted thesis ----------
console.log(`\n[2] planted "thesis": 14 notes drawn from different folders, bound ONLY by a hub note + co-editing`);
const bigFolders = GT.folders.filter((g) => g.set.size >= 20).map((g) => [...g.set]);
const plant = new Set(); for (let t = 0; plant.size < 14 && t < 500; t++) { const f = bigFolders[t % bigFolders.length]; const i = f[Math.floor(rnd() * f.length)]; if (!isHub[i]) plant.add(i); }
const pm = [...plant], hubW = 1 / Math.log2(1 + pm.length);
const plantEdges = (time) => { const e = []; for (let a = 0; a < pm.length; a++) for (let b = a + 1; b < pm.length; b++) e.push([pm[a], pm[b], hubW + (time ? 0.3 : 0)]); return e; };
console.log(`    members come from ${new Set(pm.map((i) => folder[i])).size} folders. F1 of the best-matching region, and how many members ALSO keep a second (topic) region:`);
const plantRow = (name, cover) => { const best = cover.map((c) => [f1(plant, c), c]).sort((a, b) => b[0] - a[0])[0]; const cnt = pm.map((i) => cover.filter((c) => c.has(i)).length); console.log(`    ${name.padEnd(44)} F1 ${best[0].toFixed(2)}  (region size ${best[1].size})   members also in a topic region: ${cnt.filter((x) => x >= 2).length}/${pm.length}`); };
plantRow("k-means 16 hard (shipped; cannot see links)", hardCover(shippedAssign));
for (const time of [0, 1]) {
  const adj = buildGraph(N, { links: 1, hubs: 1 }, plantEdges(time));
  plantRow(`louvain hard g3, hub${time ? " + co-editing" : ""}`, hardCover(louvain(N, adj, 3)));
  plantRow(`ego-split g3, hub${time ? " + co-editing" : ""}`, egoSplitCover(N, adj, 3));
}

// ---------- [3] incremental filing vs full recompute ----------
console.log(`\n[3] incremental: build on 90%, file the last 10% LOCALLY (only the new note is touched), compare with a full rebuild`);
{
  const ch = { links: 1, hubs: 1 }, G = 3;
  const adjAll = buildGraph(N, ch);
  const t0 = Date.now(); const full = egoSplitCover(N, adjAll, G); const tFull = Date.now() - t0;
  const order = notes.map((_, i) => i).sort(() => rnd() - 0.5), newSet = new Set(order.slice(Math.round(N * 0.9)));
  const adjOld = adjAll.map((m, i) => (newSet.has(i) ? new Map() : new Map([...m].filter(([j]) => !newSet.has(j)))));
  const cover = egoSplitCover(N, adjOld, G).map((s) => new Set(s));
  const commsOf = Array.from({ length: N }, () => []); cover.forEach((c, k) => { for (const i of c) commsOf[i].push(k); });
  const t1 = Date.now(); let filed = 0, loose = 0;
  for (const u of newSet) {
    const live = new Map([...adjAll[u]].filter(([j]) => !newSet.has(j) || commsOf[j].length > 0));
    const tmp = adjAll.map((m, i) => (i === u ? live : m)), per = personasOf(u, tmp), groups = new Map();
    for (const [v, p] of per) { if (!groups.has(p)) groups.set(p, []); groups.get(p).push(v); }
    for (const vs of groups.values()) {
      const vote = new Map(); for (const v of vs) for (const k of commsOf[v]) vote.set(k, (vote.get(k) || 0) + live.get(v) / commsOf[v].length);
      const top = [...vote.entries()].sort((a, b) => b[1] - a[1])[0];
      if (top && !commsOf[u].includes(top[0])) { commsOf[u].push(top[0]); cover[top[0]].add(u); }
    }
    if (commsOf[u].length) filed++; else loose++;
  }
  const tInc = Date.now() - t1;
  const sym = (A, B) => (A.reduce((s, a) => s + Math.max(0, ...B.map((b) => f1(a, b))), 0) / A.length + B.reduce((s, b) => s + Math.max(0, ...A.map((a) => f1(a, b))), 0) / B.length) / 2;
  console.log(`    full rebuild: ${tFull} ms for ${N} notes.   incremental: ${(tInc / newSet.size).toFixed(1)} ms per new note (${newSet.size} notes, ${filed} filed, ${loose} left loose)`);
  console.log(`    agreement between the two covers (symmetric best-match F1): ${sym(cover, full).toFixed(3)}`);
  console.log(`    quality vs the user's groups   full: tags ${fmt(bestF1(GT.tags, full))} small ${fmt(bestF1(GT.small, full))} folders ${fmt(bestF1(GT.folders, full))}`);
  console.log(`                            incremental: tags ${fmt(bestF1(GT.tags, cover))} small ${fmt(bestF1(GT.small, cover))} folders ${fmt(bestF1(GT.folders, cover))}`);
}

// ---------- examples a human can judge ----------
console.log(`\n[4] what the small user groups look like as regions (graph: text + links + hubs, ego-split gamma 3)`);
{
  const cover = egoSplitCover(N, buildGraph(N, { links: 1, hubs: 1 }), 3);
  const km = hardCover(shippedAssign);
  for (const g of GT.small.slice().sort((a, b) => a.set.size - b.set.size).filter((_, i) => i % 4 === 0).slice(0, 12)) {
    const b = cover.map((c) => [f1(g.set, c), c]).sort((x, y) => y[0] - x[0])[0], k = Math.max(...km.map((c) => f1(g.set, c)));
    console.log(`    #${g.name.padEnd(18)} ${String(g.set.size).padStart(3)} notes in ${new Set([...g.set].map((i) => folder[i])).size} folders   region F1 ${b[0].toFixed(2)} (size ${b[1].size})   shipped k-means F1 ${k.toFixed(2)}`);
  }
  const count = new Array(N).fill(0); for (const c of cover) for (const i of c) count[i]++;
  const label = (c) => { const t = new Map(); for (const i of c) for (const x of tags[i]) t.set(x, (t.get(x) || 0) + 1); return [...t.entries()].filter(([x]) => !["concept", "daily"].includes(x)).sort((a, b) => b[1] - a[1]).slice(0, 2).map(([x]) => x).join("+") || "?"; };
  console.log(`\n    notes that sit in several regions (region shown by its dominant tags):`);
  notes.map((n, i) => [count[i], i]).filter(([c]) => c >= 2).sort((a, b) => b[0] - a[0]).filter((_, i) => i % 9 === 0).slice(0, 10).forEach(([, i]) => console.log(`      ${notes[i].basename.slice(0, 44).padEnd(44)} -> ${cover.filter((c) => c.has(i)).map((c) => `${label(c)} (${c.size})`).join("  |  ")}`));
}

// ---------- [5] consolidation: the pass that would run while the app is idle ----------
// Raw ego-split output is over-split: near-duplicate regions and thin memberships. Two local rules:
// merge regions whose members mostly coincide (Jaccard), and keep a note in a secondary region only
// when a real share of its connections lives there. Nested regions are KEPT (they are the hierarchy).
function consolidate(coverIn, adj, mergeJ = 0.5, minShare = 0.2, minSize = 4) {
  let sets = coverIn.map((c) => new Set(c));
  for (let again = true; again; ) {
    again = false;
    outer: for (let a = 0; a < sets.length; a++) for (let b = a + 1; b < sets.length; b++) {
      let inter = 0; for (const x of sets[a]) if (sets[b].has(x)) inter++;
      if (inter / (sets[a].size + sets[b].size - inter) >= mergeJ) { for (const x of sets[b]) sets[a].add(x); sets.splice(b, 1); again = true; break outer; }
    }
  }
  const total = adj.map((m) => { let t = 0; for (const w of m.values()) t += w; return t || 1; });
  const share = sets.map((c) => new Map([...c].map((i) => { let t = 0; for (const [j, w] of adj[i]) if (c.has(j)) t += w; return [i, t / total[i]]; })));
  const bestOf = new Map(); sets.forEach((c, k) => { for (const i of c) { const s = share[k].get(i); if (!bestOf.has(i) || s > bestOf.get(i)[0]) bestOf.set(i, [s, k]); } });
  sets = sets.map((c, k) => new Set([...c].filter((i) => share[k].get(i) >= minShare || bestOf.get(i)[1] === k)));
  return sets.filter((c) => c.size >= minSize);
}
console.log(`\n[5] consolidation (merge near-duplicate regions, drop thin memberships). graph: text + links + hubs, gamma 3`);
console.log(`  ${"".padEnd(34)} ${"regs".padStart(4)} ${"med".padStart(5)} ${"in>=2".padStart(6)} ${"loose".padStart(6)}   tags   small  folder  MOC`);
{
  const adj = buildGraph(N, { links: 1, hubs: 1 }), raw = egoSplitCover(N, adj, 3);
  row("raw ego-split", raw, { mocs: 1 });
  for (const [j, sh] of [[0.5, 0.2], [0.4, 0.25], [0.3, 0.3]]) row(`consolidated J>=${j}, share>=${sh}`, consolidate(raw, adj, j, sh), { mocs: 1 });
  const adjT = buildGraph(N, {}), rawT = egoSplitCover(N, adjT, 3);
  row("text only: raw", rawT); row("text only: consolidated 0.4/0.25", consolidate(rawT, adjT, 0.4, 0.25));
  const adjP = buildGraph(N, { links: 1, hubs: 1 }, plantEdges(1)), adjH = buildGraph(N, { links: 1, hubs: 1 }, plantEdges(0));
  plantRow("plant after consolidation, hub + co-editing", consolidate(egoSplitCover(N, adjP, 3), adjP, 0.4, 0.25));
  plantRow("plant after consolidation, hub only", consolidate(egoSplitCover(N, adjH, 3), adjH, 0.4, 0.25));
}

// ---------- [6] can the vault tune itself by replaying its own links? (the Dream-RSI idea) ----------
// No labels on a real vault. But the links the user already made are recorded outcomes: hide 20% of
// them, form regions under many settings, and score each setting by whether hidden-link pairs end up
// sharing (small) regions more than random pairs do (AUC). If that label-free replay score ranks
// settings the same way the ground truth does, the plugin can pick its own settings per vault.
console.log(`\n[6] replay validity: does "recover my hidden links" rank settings like the ground truth does?`);
{
  const pairs = []; for (let i = 0; i < N; i++) for (const j of out[i]) if (!isHub[i] && !isHub[j] && i < j) pairs.push([i, j]);
  const linked = new Set(pairs.map(([a, b]) => a + ":" + b));
  const shuffled = pairs.slice().sort(() => rnd() - 0.5), held = shuffled.slice(0, Math.round(pairs.length * 0.2)), hide = new Set(held.map(([a, b]) => a + ":" + b));
  const neg = []; while (neg.length < held.length) { const a = Math.floor(rnd() * N), b = Math.floor(rnd() * N); if (a !== b && !linked.has(Math.min(a, b) + ":" + Math.max(a, b))) neg.push([a, b]); }
  const auc = (cover) => {
    const of = Array.from({ length: N }, () => []); cover.forEach((c, k) => { for (const i of c) of[i].push(k); });
    const sc = ([a, b]) => { let s = 0; for (const k of of[a]) if (cover[k].has(b)) s += 1 / Math.log2(1 + cover[k].size); return s; };
    const P = held.map(sc), Q = neg.map(sc); let win = 0; for (const p of P) for (const q of Q) win += p > q ? 1 : p === q ? 0.5 : 0; return win / (P.length * Q.length);
  };
  const rows = [];
  for (const [label, ch] of [["text", {}], ["text+links", { links: 1 }], ["text+links+hubs", { links: 1, hubs: 1 }]]) {
    const adj = buildGraph(N, ch, [], hide);
    for (const g of [0.5, 1, 2, 3, 4, 6]) { const c = consolidate(egoSplitCover(N, adj, g), adj, 0.4, 0.25); rows.push({ s: `${label} g${g}`, regs: c.length, auc: auc(c), tags: bestF1(GT.tags, c), small: bestF1(GT.small, c), folders: bestF1(GT.folders, c) }); }
  }
  const rank = (a) => { const o = a.map((x, i) => [x, i]).sort((p, q) => p[0] - q[0]), r = new Array(a.length); o.forEach(([, i], k) => (r[i] = k)); return r; };
  const spearman = (x, y) => { const rx = rank(x), ry = rank(y), n = x.length, mx = (n - 1) / 2; let num = 0, dx = 0, dy = 0; for (let i = 0; i < n; i++) { num += (rx[i] - mx) * (ry[i] - mx); dx += (rx[i] - mx) ** 2; dy += (ry[i] - mx) ** 2; } return num / Math.sqrt(dx * dy); };
  console.log(`    ${held.length} hidden links, ${rows.length} settings`);
  for (const r of rows) console.log(`    ${r.s.padEnd(22)} regions ${String(r.regs).padStart(3)}   replay AUC ${r.auc.toFixed(3)}   tags ${r.tags.toFixed(3)}  small ${r.small.toFixed(3)}  folders ${r.folders.toFixed(3)}`);
  const A = rows.map((r) => r.auc);
  console.log(`    Spearman(replay AUC, GT F1):  tags ${spearman(A, rows.map((r) => r.tags)).toFixed(2)}   small ${spearman(A, rows.map((r) => r.small)).toFixed(2)}   folders ${spearman(A, rows.map((r) => r.folders)).toFixed(2)}`);
  const pick = rows.slice().sort((a, b) => b.auc - a.auc)[0], bestTags = Math.max(...rows.map((r) => r.tags));
  console.log(`    replay would pick "${pick.s}": tags F1 ${pick.tags.toFixed(3)} vs best possible ${bestTags.toFixed(3)} (regret ${(bestTags - pick.tags).toFixed(3)})`);
}
