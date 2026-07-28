// PRELIMINARY CHANNEL STUDY on the expanded 1054-note vault.
//
//   node bench/v5-channels.mjs [seeds]
//
// Answers two questions the roadmap turns on:
//
//   1. Does each candidate channel predict held-out links at all, on its own?
//   2. Do they INTERACT? The admission rule says a channel only earns a place if it is
//      measurably independent of the ones already there, so what matters is not each
//      channel's recall but its rank correlation with the others and its MARGINAL gain
//      when fused on top of what already ships.
//
// Every channel here is structural: link graph, unresolved links, frontmatter entities,
// dates. None of them need the embedding model, which is why this can run before the
// 526 new notes are embedded. Content is therefore absent from these numbers; the
// question asked is how the new structural channels relate to the SHIPPED graph channel
// (Resource Allocation) and to each other.
//
// Split is seeded Fisher-Yates via ./shuffle.mjs. The old comparator "shuffle" biased
// every previous number in this project.
import { readFileSync, readdirSync, statSync, writeFileSync, mkdirSync } from "node:fs";
import { join, relative } from "node:path";
import { fisherYates } from "./shuffle.mjs";

const VAULT = "/Users/justus/obsidian_atomized_intermediary/lab/vault";
const OUT = "/Users/justus/obsidian_atomized_intermediary/lab/results/v5-channels.json";
const SEEDS = Number(process.argv[2] ?? 8);
const HOLDOUT = 0.2, K = 10;

// ------------------------------------------------------------------ load
function walk(dir, acc = []) {
  for (const name of readdirSync(dir)) {
    if (name.startsWith(".")) continue;
    const abs = join(dir, name);
    if (statSync(abs).isDirectory()) walk(abs, acc);
    else if (name.endsWith(".md")) acc.push(abs);
  }
  return acc;
}
const files = walk(VAULT);
const notes = files.map((abs) => {
  const rel = relative(VAULT, abs);
  const raw = readFileSync(abs, "utf8");
  const fm = raw.match(/^---\n([\s\S]*?)\n---/);
  const front = fm ? fm[1] : "";
  const list = (key) => {
    const m = front.match(new RegExp(`^${key}:\\n((?:  - .+\\n?)+)`, "m"));
    return m ? m[1].split("\n").map((l) => l.replace(/^\s*-\s*/, "").trim()).filter(Boolean) : [];
  };
  const scalar = (key) => (front.match(new RegExp(`^${key}: *(.+)$`, "m")) ?? [])[1]?.trim() ?? null;
  const title = rel.slice(rel.lastIndexOf("/") + 1, -3);
  const dateStr = scalar("date") ?? (title.match(/\d{4}-\d{2}-\d{2}/) ?? [])[0] ?? null;
  return {
    rel, title,
    people: list("people"),
    tags: list("tags"),
    place: scalar("place"),
    day: dateStr ? Math.round(Date.parse(dateStr + "T00:00:00Z") / 86400000) : null,
    targets: [...raw.matchAll(/\[\[([^\]|#]+)/g)].map((m) => m[1].trim()),
  };
});
const byTitle = new Map();
for (let i = 0; i < notes.length; i++) if (!byTitle.has(notes[i].title)) byTitle.set(notes[i].title, i);
const N = notes.length;

// resolved edges (undirected, deduped) and unresolved targets per note
const edgeSet = new Set(), ghost = notes.map(() => new Set());
for (let i = 0; i < N; i++) {
  for (const t of notes[i].targets) {
    const j = byTitle.get(t);
    if (j === undefined) { ghost[i].add(t); continue; }
    if (j === i) continue;
    edgeSet.add(i < j ? `${i},${j}` : `${j},${i}`);
  }
}
const edges = [...edgeSet].map((s) => s.split(",").map(Number));

// entity and tag incidence, for the entity channel
const peopleOf = notes.map((n) => new Set(n.people));
const placeOf = notes.map((n) => (n.place ? n.place : null));
const tagsOf = notes.map((n) => new Set(n.tags));

// ------------------------------------------------------------- channels
// Each channel: given a source i and a train-graph adjacency, return Map(j -> score).
function buildAdj(trainEdges) {
  const adj = Array.from({ length: N }, () => new Set());
  for (const [a, b] of trainEdges) { adj[a].add(b); adj[b].add(a); }
  return adj;
}
// degree-penalised shared neighbours: the shipped channel
function chRA(i, adj) {
  const out = new Map();
  for (const x of adj[i]) { const dx = adj[x].size; if (dx < 1) continue;
    for (const j of adj[x]) if (j !== i && !adj[i].has(j)) out.set(j, (out.get(j) ?? 0) + 1 / dx); }
  return out;
}
function chAA(i, adj) {
  const out = new Map();
  for (const x of adj[i]) { const dx = adj[x].size; if (dx < 2) continue;
    for (const j of adj[x]) if (j !== i && !adj[i].has(j)) out.set(j, (out.get(j) ?? 0) + 1 / Math.log(dx)); }
  return out;
}
function chCN(i, adj) {
  const out = new Map();
  for (const x of adj[i]) for (const j of adj[x]) if (j !== i && !adj[i].has(j)) out.set(j, (out.get(j) ?? 0) + 1);
  return out;
}
// GHOST LINKS: two notes that both point at a title neither of them wrote. The shipped
// channel cannot see this at all, because RA runs over RESOLVED links only.
const ghostIndex = new Map();  // phantom title -> [note ids]
for (let i = 0; i < N; i++) for (const t of ghost[i]) {
  if (!ghostIndex.has(t)) ghostIndex.set(t, []);
  ghostIndex.get(t).push(i);
}
function chGhost(i) {
  const out = new Map();
  for (const t of ghost[i]) {
    const holders = ghostIndex.get(t) ?? [];
    if (holders.length < 2 || holders.length > 60) continue;   // a phantom everyone cites says nothing
    for (const j of holders) if (j !== i) out.set(j, (out.get(j) ?? 0) + 1 / holders.length);
  }
  return out;
}
// ENTITY: shared people, degree-penalised the same way RA penalises hub neighbours
const personIndex = new Map();
for (let i = 0; i < N; i++) for (const p of peopleOf[i]) {
  if (!personIndex.has(p)) personIndex.set(p, []);
  personIndex.get(p).push(i);
}
function chPeople(i) {
  const out = new Map();
  for (const p of peopleOf[i]) {
    const holders = personIndex.get(p) ?? [];
    if (holders.length < 2) continue;
    for (const j of holders) if (j !== i) out.set(j, (out.get(j) ?? 0) + 1 / Math.log(1 + holders.length));
  }
  return out;
}
const placeIndex = new Map();
for (let i = 0; i < N; i++) if (placeOf[i]) {
  if (!placeIndex.has(placeOf[i])) placeIndex.set(placeOf[i], []);
  placeIndex.get(placeOf[i]).push(i);
}
function chPlace(i) {
  const out = new Map();
  const holders = placeOf[i] ? (placeIndex.get(placeOf[i]) ?? []) : [];
  for (const j of holders) if (j !== i) out.set(j, 1 / Math.log(1 + holders.length));
  return out;
}
function chTags(i) {
  const out = new Map();
  for (let j = 0; j < N; j++) {
    if (j === i || tagsOf[i].size === 0) continue;
    let sh = 0; for (const t of tagsOf[i]) if (tagsOf[j].has(t)) sh++;
    if (sh) out.set(j, sh / Math.sqrt(tagsOf[i].size * (tagsOf[j].size || 1)));
  }
  return out;
}
// TEMPORAL: notes written close together, on the theory that a week's work coheres
function chTime(i) {
  const out = new Map();
  if (notes[i].day == null) return out;
  for (let j = 0; j < N; j++) {
    if (j === i || notes[j].day == null) continue;
    const d = Math.abs(notes[i].day - notes[j].day);
    if (d > 30) continue;
    out.set(j, Math.exp(-d / 7));
  }
  return out;
}
const CHANNELS = {
  RA:      (i, adj) => chRA(i, adj),
  AA:      (i, adj) => chAA(i, adj),
  CN:      (i, adj) => chCN(i, adj),
  ghost:   (i) => chGhost(i),
  people:  (i) => chPeople(i),
  place:   (i) => chPlace(i),
  tags:    (i) => chTags(i),
  time:    (i) => chTime(i),
};
const NAMES = Object.keys(CHANNELS);

// ------------------------------------------------------------- scoring
function lcg(seed) { let s = seed >>> 0 || 1; return () => ((s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff); }
// z-normalise a Map's values, so channels of different scale can be summed
function znorm(m) {
  const v = [...m.values()];
  if (v.length < 2) return new Map();
  const mu = v.reduce((a, b) => a + b, 0) / v.length;
  const sd = Math.sqrt(v.reduce((a, b) => a + (b - mu) ** 2, 0) / v.length) || 1;
  return new Map([...m].map(([k, x]) => [k, (x - mu) / sd]));
}
function recallAt(scoreFn, test, adj) {
  let hit = 0, tot = 0;
  const bySource = new Map();
  for (const [a, b] of test) {
    if (!bySource.has(a)) bySource.set(a, []); bySource.get(a).push(b);
    if (!bySource.has(b)) bySource.set(b, []); bySource.get(b).push(a);
  }
  for (const [i, wanted] of bySource) {
    const sc = scoreFn(i, adj);
    const top = [...sc.entries()].sort((x, y) => y[1] - x[1]).slice(0, K).map(([j]) => j);
    for (const w of wanted) { tot++; if (top.includes(w)) hit++; }
  }
  return tot ? hit / tot : 0;
}
// Spearman between two channels over a common candidate pool, averaged across sources
function correlation(fa, fb, adj, rnd) {
  const sample = [];
  for (let t = 0; t < 120; t++) sample.push(Math.floor(rnd() * N));
  const rhos = [];
  for (const i of sample) {
    const A = fa(i, adj), B = fb(i, adj);
    const keys = [...new Set([...A.keys(), ...B.keys()])];
    if (keys.length < 8) continue;
    const rank = (m) => { const s = keys.map((k) => [k, m.get(k) ?? 0]).sort((x, y) => y[1] - x[1]);
      const r = new Map(); s.forEach(([k], idx) => r.set(k, idx)); return r; };
    const ra = rank(A), rb = rank(B);
    const n = keys.length;
    let d2 = 0; for (const k of keys) d2 += (ra.get(k) - rb.get(k)) ** 2;
    rhos.push(1 - (6 * d2) / (n * (n * n - 1)));
  }
  return rhos.length ? rhos.reduce((a, b) => a + b, 0) / rhos.length : NaN;
}

// A channel that fires on 4% of pairs and is excellent there is a PANEL, not a ranking
// channel: averaging its recall over every held-out edge buries it. So measure coverage
// (how often it produces any candidate at all) and recall CONDITIONAL on firing.
function coverage(f, test, adj) {
  const bySource = new Map();
  for (const [a, b] of test) {
    if (!bySource.has(a)) bySource.set(a, []); bySource.get(a).push(b);
    if (!bySource.has(b)) bySource.set(b, []); bySource.get(b).push(a);
  }
  let fired = 0, srcs = 0, hit = 0, tot = 0;
  for (const [i, wanted] of bySource) {
    srcs++;
    const sc = f(i, adj);
    if (sc.size === 0) continue;
    fired++;
    const top = [...sc.entries()].sort((x, y) => y[1] - x[1]).slice(0, K).map(([j]) => j);
    for (const w of wanted) { tot++; if (top.includes(w)) hit++; }
  }
  return { fires: fired / srcs, condRecall: tot ? hit / tot : 0 };
}

const acc = { solo: {}, fused: {}, marginal: {}, corr: {}, cov: {}, cond: {} };
const push = (o, k, v) => { (o[k] ??= []).push(v); };

for (let s = 0; s < SEEDS; s++) {
  const rnd = lcg(20260728 + s * 7919);
  const sh = fisherYates(edges, rnd);
  const cut = Math.floor(sh.length * HOLDOUT);
  const test = sh.slice(0, cut), train = sh.slice(cut);
  const adj = buildAdj(train);

  for (const n of NAMES) {
    push(acc.solo, n, recallAt(CHANNELS[n], test, adj));
    const c = coverage(CHANNELS[n], test, adj);
    push(acc.cov, n, c.fires); push(acc.cond, n, c.condRecall);
  }

  // marginal: RA alone, then RA + each other channel, z-normalised rank fusion
  const raOnly = (i, a) => znorm(CHANNELS.RA(i, a));
  const base = recallAt(raOnly, test, adj);
  push(acc.fused, "RA", base);
  for (const n of NAMES) {
    if (n === "RA") continue;
    const fused = (i, a) => {
      const A = znorm(CHANNELS.RA(i, a)), B = znorm(CHANNELS[n](i, a));
      const out = new Map(A);
      for (const [k, v] of B) out.set(k, (out.get(k) ?? 0) + v);
      return out;
    };
    const r = recallAt(fused, test, adj);
    push(acc.fused, `RA+${n}`, r);
    push(acc.marginal, n, r - base);
  }
  // all-structural fusion
  const allF = (i, a) => {
    const out = new Map();
    for (const n of NAMES) for (const [k, v] of znorm(CHANNELS[n](i, a))) out.set(k, (out.get(k) ?? 0) + v);
    return out;
  };
  push(acc.fused, "all", recallAt(allF, test, adj));

  if (s === 0) for (let x = 0; x < NAMES.length; x++) for (let y = x + 1; y < NAMES.length; y++)
    acc.corr[`${NAMES[x]}_vs_${NAMES[y]}`] = correlation(CHANNELS[NAMES[x]], CHANNELS[NAMES[y]], adj, lcg(99));
}

const stat = (a) => {
  const mu = a.reduce((x, y) => x + y, 0) / a.length;
  const sd = Math.sqrt(a.reduce((x, y) => x + (y - mu) ** 2, 0) / a.length);
  return { mean: +mu.toFixed(4), sd: +sd.toFixed(4) };
};
const solo = Object.fromEntries(Object.entries(acc.solo).map(([k, v]) => [k, stat(v)]));
const fused = Object.fromEntries(Object.entries(acc.fused).map(([k, v]) => [k, stat(v)]));
const marginal = Object.fromEntries(Object.entries(acc.marginal).map(([k, v]) => [k, stat(v)]));

console.log(`\n==== STRUCTURAL CHANNEL STUDY (no embeddings) ====`);
console.log(`${N} notes | ${edges.length} undirected edges | ${SEEDS} seeds | recall@${K}, mean +- SD\n`);
console.log("  channel alone");
for (const [k, v] of Object.entries(solo).sort((a, b) => b[1].mean - a[1].mean))
  console.log(`    ${k.padEnd(10)} ${v.mean.toFixed(4)} +- ${v.sd.toFixed(4)}`);
console.log("\n  fused with the shipped RA channel");
for (const [k, v] of Object.entries(fused).sort((a, b) => b[1].mean - a[1].mean))
  console.log(`    ${k.padEnd(10)} ${v.mean.toFixed(4)} +- ${v.sd.toFixed(4)}`);
console.log("\n  MARGINAL gain on top of RA (this is the admission test)");
for (const [k, v] of Object.entries(marginal).sort((a, b) => b[1].mean - a[1].mean)) {
  const verdict = v.mean > 2 * v.sd ? "PASSES" : v.mean > 0 ? "within noise" : "hurts";
  console.log(`    ${k.padEnd(10)} ${v.mean >= 0 ? "+" : ""}${v.mean.toFixed(4)} +- ${v.sd.toFixed(4)}   ${verdict}`);
}
console.log("\n  coverage and recall WHERE THE CHANNEL FIRES (panel vs ranking channel)");
const cov = Object.fromEntries(Object.entries(acc.cov).map(([k, v]) => [k, stat(v)]));
const cond = Object.fromEntries(Object.entries(acc.cond).map(([k, v]) => [k, stat(v)]));
for (const k of Object.keys(cov).sort((a, b) => cond[b].mean - cond[a].mean))
  console.log(`    ${k.padEnd(10)} fires on ${(cov[k].mean * 100).toFixed(0).padStart(3)}% of notes   recall there ${cond[k].mean.toFixed(4)}`);
console.log("\n  rank correlation between channels (lower = more independent)");
for (const [k, v] of Object.entries(acc.corr).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 14))
  console.log(`    ${k.padEnd(24)} ${Number.isFinite(v) ? v.toFixed(3) : "n/a"}`);

mkdirSync("/Users/justus/obsidian_atomized_intermediary/lab/results", { recursive: true });
writeFileSync(OUT, JSON.stringify({ notes: N, edges: edges.length, seeds: SEEDS, solo, fused, marginal, corr: acc.corr, coverage: acc.cov, conditional: acc.cond }, null, 1));
console.log("\nwrote", OUT);
