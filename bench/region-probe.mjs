// Would named "regions" give the sidebar and the map real structure? Measures, on the
// lab vault and against the SHIPPED src/vault-map.ts: how a note's top-10 splits across
// regions, how faithful the global-PCA scatter is, and whether a territory layout
// (centroid MDS + local PCA per region) is more faithful.
//   node --experimental-strip-types bench/region-probe.mjs
//   LAB_MODEL="jinaai/jina-embeddings-v5-text-nano-text-matching" node --experimental-strip-types bench/region-probe.mjs
// Uses the SHIPPED src/vault-map.ts on the lab vault, same loading as bench/v3-map.mjs.
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { buildVaultMap } from "../src/vault-map.ts";

const VAULT = "/Users/justus/obsidian_atomized_intermediary/lab/vault";
const MODEL = process.env.LAB_MODEL || "Xenova/paraphrase-multilingual-MiniLM-L12-v2";
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

const notes = [];
for (const abs of walk(VAULT)) {
  const rel = relative(VAULT, abs);
  if (rel.startsWith("real/") && !INDEXABLE.has(rel)) continue;
  const basename = rel.replace(/\.md$/, "").split("/").pop();
  if (/(^|\/)Attachments\//.test(rel) || /\.dup$/.test(basename)) continue;
  const body = stripFront(readFileSync(abs, "utf8"));
  const v = cache[noteText({ basename, body })];
  if (v) notes.push({ rel, basename, vec: v });
}
const D = notes[0].vec.length;
const mean = new Array(D).fill(0);
for (const n of notes) for (let i = 0; i < D; i++) mean[i] += n.vec[i] / notes.length;
const l2 = (v) => { let s = 0; for (const x of v) s += x * x; s = Math.sqrt(s) || 1; return v.map((x) => x / s); };
const vecs = notes.map((n) => Float32Array.from(l2(n.vec.map((x, i) => x - mean[i]))));
const input = notes.map((n, i) => ({ path: n.rel, title: n.basename, vec: vecs[i] }));

const map = buildVaultMap(input);
const N = map.points.length;
const cluster = map.points.map((p) => p.cluster);
const K = 10;

const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
function topK(scoreFn, i) {
  const s = [];
  for (let j = 0; j < N; j++) if (j !== i) s.push([scoreFn(i, j), j]);
  s.sort((a, b) => b[0] - a[0]);
  return s.slice(0, K).map((x) => x[1]);
}
const hiNN = [], loNN = [];
for (let i = 0; i < N; i++) {
  hiNN.push(topK((a, b) => dot(vecs[a], vecs[b]), i));
  loNN.push(topK((a, b) => -Math.hypot(map.points[a].x - map.points[b].x, map.points[a].y - map.points[b].y), i));
}

// 1. How does a note's top-10 split across regions?
const distinct = new Array(12).fill(0);
let ownShare = 0, usefulSplit = 0;
for (let i = 0; i < N; i++) {
  const cs = hiNN[i].map((j) => cluster[j]);
  const set = new Set(cs);
  distinct[set.size]++;
  ownShare += cs.filter((c) => c === cluster[i]).length / K;
  // "useful split": 2-4 groups and no single group holds 9+ of the 10 cards
  const counts = [...set].map((c) => cs.filter((x) => x === c).length);
  if (set.size >= 2 && set.size <= 4 && Math.max(...counts) <= 8) usefulSplit++;
}
console.log(`\n==== REGION PROBE  (${MODEL}) ====`);
console.log(`${N} notes, ${map.clusters.length} regions (shipped k-means)`);
console.log(`\n[1] regions represented in a note's content top-${K}:`);
for (let d = 1; d < distinct.length; d++) if (distinct[d]) console.log(`    ${d} region(s): ${(100 * distinct[d] / N).toFixed(1)}% of notes`);
console.log(`    mean share of top-${K} from the note's OWN region: ${(100 * ownShare / N).toFixed(1)}%`);
console.log(`    lists that split usefully (2-4 groups, none holding 9+): ${(100 * usefulSplit / N).toFixed(1)}%`);

// 2. Is the PCA scatter faithful? neighbour preservation + visual separation of colours.
let overlap = 0, sameColour = 0;
for (let i = 0; i < N; i++) {
  const hi = new Set(hiNN[i]);
  overlap += loNN[i].filter((j) => hi.has(j)).length / K;
  sameColour += loNN[i].filter((j) => cluster[j] === cluster[i]).length / K;
}
console.log(`\n[2] current map (global 2-D PCA):`);
console.log(`    true top-${K} neighbours that are also 2-D neighbours: ${(100 * overlap / N).toFixed(1)}%`);
console.log(`    2-D neighbours sharing the note's colour: ${(100 * sameColour / N).toFixed(1)}%  (100% = clean territories)`);

// 3. Alternative layout: place region CENTROIDS by classical MDS, members around them by local PCA.
//    Cheap, deterministic; measures whether a "territory" layout is more faithful than global PCA.
const ids = map.clusters.map((c) => c.id);
const cent = new Map();
for (const id of ids) {
  const sum = new Float32Array(D); let n = 0;
  for (let i = 0; i < N; i++) if (cluster[i] === id) { n++; for (let d = 0; d < D; d++) sum[d] += vecs[i][d]; }
  cent.set(id, Float32Array.from(l2(Array.from(sum))));
}
function pca2(rows) { // power iteration, 2 comps, rows = array of arrays (already centred by caller)
  const dims = rows[0].length; const comps = [];
  let seed = 7; const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff - 0.5; };
  for (let c = 0; c < 2; c++) {
    let w = Array.from({ length: dims }, rnd);
    for (let it = 0; it < 60; it++) {
      const nx = new Array(dims).fill(0);
      for (const x of rows) { let s = 0; for (let d = 0; d < dims; d++) s += x[d] * w[d]; for (let d = 0; d < dims; d++) nx[d] += s * x[d]; }
      for (const p of comps) { let s = 0; for (let d = 0; d < dims; d++) s += nx[d] * p[d]; for (let d = 0; d < dims; d++) nx[d] -= s * p[d]; }
      w = l2(nx);
    }
    comps.push(w);
  }
  return rows.map((x) => comps.map((w) => { let s = 0; for (let d = 0; d < dims; d++) s += x[d] * w[d]; return s; }));
}
const centre = (rows) => { const m = new Array(rows[0].length).fill(0); for (const r of rows) for (let d = 0; d < m.length; d++) m[d] += r[d] / rows.length; return rows.map((r) => r.map((x, d) => x - m[d])); };
const cRows = ids.map((id) => Array.from(cent.get(id)));
const cXY = pca2(centre(cRows)); // PCA of centroids == classical MDS on them up to scale
const cScale = Math.max(...cXY.flat().map(Math.abs)) || 1;
const pos = new Array(N);
ids.forEach((id, ci) => {
  const members = []; for (let i = 0; i < N; i++) if (cluster[i] === id) members.push(i);
  const local = members.length >= 3 ? pca2(centre(members.map((i) => Array.from(vecs[i])))) : members.map(() => [0, 0]);
  const lScale = Math.max(...local.flat().map(Math.abs)) || 1;
  const radius = 0.10 + 0.22 * Math.sqrt(members.length / N); // bigger region, bigger territory
  members.forEach((i, mi) => { pos[i] = [cXY[ci][0] / cScale + radius * local[mi][0] / lScale, cXY[ci][1] / cScale + radius * local[mi][1] / lScale]; });
});
let overlap2 = 0, sameColour2 = 0;
for (let i = 0; i < N; i++) {
  const s = [];
  for (let j = 0; j < N; j++) if (j !== i) s.push([Math.hypot(pos[i][0] - pos[j][0], pos[i][1] - pos[j][1]), j]);
  s.sort((a, b) => a[0] - b[0]);
  const nn = s.slice(0, K).map((x) => x[1]);
  const hi = new Set(hiNN[i]);
  overlap2 += nn.filter((j) => hi.has(j)).length / K;
  sameColour2 += nn.filter((j) => cluster[j] === cluster[i]).length / K;
}
console.log(`\n[3] territory layout (centroid MDS + local PCA per region):`);
console.log(`    true top-${K} neighbours that are also 2-D neighbours: ${(100 * overlap2 / N).toFixed(1)}%`);
console.log(`    2-D neighbours sharing the note's colour: ${(100 * sameColour2 / N).toFixed(1)}%`);

// 4. The labels a user would actually read.
console.log(`\n[4] region labels as shipped (c-TF-IDF over titles):`);
for (const c of [...map.clusters].sort((a, b) => b.size - a.size)) console.log(`    ${String(c.size).padStart(4)}  ${c.label}`);
