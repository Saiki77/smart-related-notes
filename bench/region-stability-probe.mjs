// Can regions behave like sections of a self-sorting notebook? Three questions, measured on
// the lab vault against the SHIPPED src/vault-map.ts clustering:
//   [1] STABILITY  when 10% new notes arrive, how many existing notes change region?
//                  cold re-cluster (what the map does today) vs warm start vs warm + hysteresis
//   [2] FILING     is a new note filed where its nearest neighbours already live, and how
//                  often is the call a clear one?
//   [3] SPACES     do the ~16 regions group into a handful of coherent larger spaces?
//   node --experimental-strip-types bench/region-stability-probe.mjs
//   LAB_MODEL="jinaai/jina-embeddings-v5-text-nano-text-matching" node --experimental-strip-types bench/region-stability-probe.mjs
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
  const v = cache[noteText({ basename, body: stripFront(readFileSync(abs, "utf8")) })];
  if (v) notes.push({ rel, basename, vec: v });
}
const N = notes.length, D = notes[0].vec.length;
const norm = (v) => { let s = 0; for (const x of v) s += x * x; s = Math.sqrt(s) || 1; const o = new Float32Array(v.length); for (let i = 0; i < v.length; i++) o[i] = v[i] / s; return o; };
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const mean = new Float32Array(D);
for (const n of notes) for (let i = 0; i < D; i++) mean[i] += n.vec[i] / N;
const V = notes.map((n) => norm(n.vec.map((x, i) => x - mean[i])));
const folderOf = (rel) => { const p = rel.split("/"); return p.length > 2 ? p[1] : p[0]; };
const folders = notes.map((n) => folderOf(n.rel));

let seed = 20260918;
const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };

const shipped = (idx) => { // shipped clustering on a subset; returns cluster id per position in idx
  const map = buildVaultMap(idx.map((i) => ({ path: notes[i].rel, title: notes[i].basename, vec: V[i] })));
  const byPath = new Map(map.points.map((p) => [p.path, p.cluster]));
  return { assign: idx.map((i) => byPath.get(notes[i].rel)), labels: new Map(map.clusters.map((c) => [c.id, c.label])) };
};
function centroidsOf(idx, assign) {
  const ids = [...new Set(assign)].sort((a, b) => a - b);
  const sums = new Map(ids.map((id) => [id, new Float32Array(D)]));
  idx.forEach((i, p) => { const s = sums.get(assign[p]); for (let d = 0; d < D; d++) s[d] += V[i][d]; });
  return { ids, C: ids.map((id) => norm(sums.get(id))) };
}
// Spherical Lloyd iterations from GIVEN centroids. `prev` + `margin` = hysteresis: a note keeps
// its previous region unless another centroid beats it by more than `margin`.
function lloyd(idx, C0, prev = null, margin = 0, iters = 30) {
  let C = C0.map((c) => Float32Array.from(c));
  const assign = new Array(idx.length).fill(-1);
  for (let it = 0; it < iters; it++) {
    let moved = 0;
    idx.forEach((i, p) => {
      let b = 0, bs = -Infinity;
      for (let c = 0; c < C.length; c++) { const s = dot(V[i], C[c]); if (s > bs) { bs = s; b = c; } }
      if (prev && prev[p] !== undefined && prev[p] >= 0 && b !== prev[p] && bs - dot(V[i], C[prev[p]]) <= margin) b = prev[p];
      if (assign[p] !== b) { assign[p] = b; moved++; }
    });
    const sums = C.map(() => new Float32Array(D)), cnt = C.map(() => 0);
    idx.forEach((i, p) => { cnt[assign[p]]++; const s = sums[assign[p]]; for (let d = 0; d < D; d++) s[d] += V[i][d]; });
    C = C.map((c, k) => (cnt[k] > 0 ? norm(sums[k]) : c));
    if (!moved) break;
  }
  return { assign, C };
}
function hungarian(cost) { // square matrix, minimise; returns row -> col
  const n = cost.length, u = new Array(n + 1).fill(0), v = new Array(n + 1).fill(0), p = new Array(n + 1).fill(0), way = new Array(n + 1).fill(0);
  for (let i = 1; i <= n; i++) {
    p[0] = i; let j0 = 0; const minv = new Array(n + 1).fill(Infinity), used = new Array(n + 1).fill(false);
    do {
      used[j0] = true; const i0 = p[j0]; let delta = Infinity, j1 = 0;
      for (let j = 1; j <= n; j++) if (!used[j]) {
        const cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
        if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
      }
      for (let j = 0; j <= n; j++) if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else minv[j] -= delta;
      j0 = j1;
    } while (p[j0] !== 0);
    do { const j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
  }
  const res = new Array(n).fill(-1);
  for (let j = 1; j <= n; j++) if (p[j] > 0) res[p[j] - 1] = j - 1;
  return res;
}
function purity(assign, labels) {
  const by = new Map();
  assign.forEach((c, i) => { if (!by.has(c)) by.set(c, new Map()); const m = by.get(c); m.set(labels[i], (m.get(labels[i]) ?? 0) + 1); });
  let hit = 0; for (const m of by.values()) hit += Math.max(...m.values());
  return hit / assign.length;
}
const avg = (a) => a.reduce((s, x) => s + x, 0) / a.length;

console.log(`\n==== REGION STABILITY PROBE  (${MODEL}) ====\n${N} notes`);

// ---------- [1] + [2]: five random 90/10 splits ----------
const R = { cold: [], warm: [], hyst: [], pCold: [], pWarm: [], pHyst: [], agree: [], clear: [], between: [] };
for (let rep = 0; rep < 5; rep++) {
  const order = notes.map((_, i) => i).sort(() => rnd() - 0.5);
  const cut = Math.round(N * 0.9);
  const oldIdx = order.slice(0, cut), newIdx = order.slice(cut), allIdx = [...oldIdx, ...newIdx];

  const base = shipped(oldIdx);
  const { ids, C } = centroidsOf(oldIdx, base.assign);
  const basePos = base.assign.map((id) => ids.indexOf(id)); // region index 0..k-1 for old notes

  // cold: what the map does today, re-run from scratch on everything, best-case relabelled
  const cold = shipped(allIdx);
  const coldIds = [...new Set(cold.assign)].sort((a, b) => a - b);
  const m = Math.max(coldIds.length, ids.length);
  const O = Array.from({ length: m }, () => new Array(m).fill(0));
  oldIdx.forEach((_, p) => { O[coldIds.indexOf(cold.assign[p])][basePos[p]]--; });
  const match = hungarian(O);
  let same = 0; oldIdx.forEach((_, p) => { if (match[coldIds.indexOf(cold.assign[p])] === basePos[p]) same++; });
  R.cold.push(1 - same / oldIdx.length);
  R.pCold.push(purity(cold.assign, allIdx.map((i) => folders[i])));

  // warm start from the saved centroids
  const warm = lloyd(allIdx, C);
  same = 0; oldIdx.forEach((_, p) => { if (warm.assign[p] === basePos[p]) same++; });
  R.warm.push(1 - same / oldIdx.length);
  R.pWarm.push(purity(warm.assign, allIdx.map((i) => folders[i])));

  // warm start + hysteresis (keep the old region unless another wins by > 0.03)
  const prev = [...basePos, ...newIdx.map(() => -1)];
  const hyst = lloyd(allIdx, C, prev, 0.03);
  same = 0; oldIdx.forEach((_, p) => { if (hyst.assign[p] === basePos[p]) same++; });
  R.hyst.push(1 - same / oldIdx.length);
  R.pHyst.push(purity(hyst.assign, allIdx.map((i) => folders[i])));

  // filing: a new note goes to its nearest saved centroid. Do its 5 nearest OLD notes live there?
  let agree = 0, clear = 0, between = 0;
  for (const i of newIdx) {
    const sims = C.map((c) => dot(V[i], c));
    const orderC = sims.map((s, k) => [s, k]).sort((a, b) => b[0] - a[0]);
    const filed = orderC[0][1], margin = orderC[0][0] - orderC[1][0];
    if (margin >= 0.05) clear++; else if (margin < 0.02) between++;
    const nn = oldIdx.map((j, p) => [dot(V[i], V[j]), basePos[p]]).sort((a, b) => b[0] - a[0]).slice(0, 5);
    const votes = new Map(); for (const [, r] of nn) votes.set(r, (votes.get(r) ?? 0) + 1);
    const top = [...votes.entries()].sort((a, b) => b[1] - a[1])[0][0];
    if (top === filed) agree++;
  }
  R.agree.push(agree / newIdx.length); R.clear.push(clear / newIdx.length); R.between.push(between / newIdx.length);
}
const pct = (a) => `${(100 * avg(a)).toFixed(1)}%`;
console.log(`\n[1] 10% new notes arrive. Existing notes that CHANGE region (mean of 5 splits):`);
console.log(`    cold re-cluster (today's behaviour, best-case relabelled): ${pct(R.cold)}   purity ${avg(R.pCold).toFixed(3)}`);
console.log(`    warm start from saved centroids:                           ${pct(R.warm)}   purity ${avg(R.pWarm).toFixed(3)}`);
console.log(`    warm start + hysteresis 0.03:                              ${pct(R.hyst)}   purity ${avg(R.pHyst).toFixed(3)}`);
console.log(`\n[2] filing a NEW note into the nearest saved region:`);
console.log(`    lands where most of its 5 nearest notes already live: ${pct(R.agree)}`);
console.log(`    clear call (margin >= 0.05): ${pct(R.clear)}    between two regions (margin < 0.02): ${pct(R.between)}`);

// ---------- [3]: do regions group into a few spaces? ----------
const allIdx = notes.map((_, i) => i);
const full = shipped(allIdx);
const { ids, C } = centroidsOf(allIdx, full.assign);
const pos = full.assign.map((id) => ids.indexOf(id));
function agglomerate(S) {
  let groups = C.map((_, k) => [k]);
  while (groups.length > S) {
    let best = [-Infinity, 0, 1];
    for (let a = 0; a < groups.length; a++) for (let b = a + 1; b < groups.length; b++) {
      let s = 0; for (const x of groups[a]) for (const y of groups[b]) s += dot(C[x], C[y]);
      s /= groups[a].length * groups[b].length;
      if (s > best[0]) best = [s, a, b];
    }
    groups[best[1]] = [...groups[best[1]], ...groups[best[2]]]; groups.splice(best[2], 1);
  }
  return groups;
}
console.log(`\n[3] regions -> spaces (average-linkage over region centroids). purity vs top folders:`);
console.log(`    regions alone (k=${ids.length}): ${purity(pos, folders).toFixed(3)}`);
for (const S of [4, 5, 6, 7]) {
  const groups = agglomerate(S);
  const spaceOf = new Map(); groups.forEach((g, s) => g.forEach((k) => spaceOf.set(k, s)));
  console.log(`    ${S} spaces: purity ${purity(pos.map((k) => spaceOf.get(k)), folders).toFixed(3)}   sizes ${groups.map((g) => g.reduce((s, k) => s + pos.filter((x) => x === k).length, 0)).join(" / ")}`);
}
const groups = agglomerate(5);
console.log(`\n    the 5-space grouping, each region with its size, shipped label and dominant folder:`);
groups.forEach((g, s) => {
  console.log(`    space ${s + 1}`);
  for (const k of g) {
    const members = allIdx.filter((_, p) => pos[p] === k);
    const f = new Map(); for (const i of members) f.set(folders[i], (f.get(folders[i]) ?? 0) + 1);
    const top = [...f.entries()].sort((a, b) => b[1] - a[1]).slice(0, 2).map(([n, c]) => `${n} ${c}`).join(", ");
    console.log(`      ${String(members.length).padStart(4)}  ${(full.labels.get(ids[k]) ?? "").padEnd(40)} ${top}`);
  }
});

// how tightly do notes sit in their region? (who would count as a loose page)
const own = allIdx.map((i, p) => dot(V[i], C[pos[p]])).sort((a, b) => a - b);
const q = (f) => own[Math.floor(f * (own.length - 1))].toFixed(3);
console.log(`\n[4] similarity of a note to its own region centre: p5 ${q(0.05)}  p25 ${q(0.25)}  median ${q(0.5)}  p75 ${q(0.75)}`);
