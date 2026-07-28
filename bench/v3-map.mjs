// Verify the SHIPPED vault-map code (src/vault-map.ts, imported directly) on the
// lab vault: are the clusters real, and are the auto-generated labels readable?
//
//   node --experimental-strip-types bench/v3-map.mjs
import { readFileSync, readdirSync, statSync, writeFileSync, mkdirSync } from "node:fs";
import { join, relative } from "node:path";
import { buildVaultMap } from "../src/vault-map.ts";
import { fisherYates } from "./shuffle.mjs";

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
// Center exactly as the plugin does before handing vectors to the map.
const D = notes[0].vec.length;
const mean = new Array(D).fill(0);
for (const n of notes) for (let i = 0; i < D; i++) mean[i] += n.vec[i] / notes.length;
const l2 = (v) => { let s = 0; for (const x of v) s += x * x; s = Math.sqrt(s) || 1; return v.map((x) => x / s); };

const input = notes.map((n) => ({
  path: n.rel, title: n.basename,
  vec: Float32Array.from(l2(n.vec.map((x, i) => x - mean[i]))),
}));

const t0 = Date.now();
const map = buildVaultMap(input);
const ms = Date.now() - t0;

// Purity against the vault's own folders, plus a shuffled baseline so the number
// is interpretable rather than merely large.
const folderOf = (rel) => { const parts = rel.split("/"); return parts.length > 2 ? parts[1] : parts[0]; };
const labels = notes.map((n) => folderOf(n.rel));
function purity(assign) {
  const byC = new Map();
  assign.forEach((c, i) => { if (!byC.has(c)) byC.set(c, []); byC.get(c).push(labels[i]); });
  let hit = 0;
  for (const members of byC.values()) {
    const cnt = new Map();
    for (const l of members) cnt.set(l, (cnt.get(l) ?? 0) + 1);
    hit += Math.max(...cnt.values());
  }
  return +(hit / assign.length).toFixed(3);
}
const assign = map.points.map((p) => p.cluster);
let seed = 12345;
const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
const shuffled = fisherYates(assign, rnd);

console.log(`\n==== VAULT MAP (shipped code) ====`);
console.log(`model ${MODEL}`);
console.log(`${map.points.length} notes, ${map.clusters.length} clusters, built in ${ms} ms`);
console.log(`purity vs folders: ${purity(assign)}   (shuffled baseline ${purity(shuffled)})\n`);
for (const c of [...map.clusters].sort((a, b) => b.size - a.size)) {
  const members = map.points.filter((p) => p.cluster === c.id);
  const folders = new Map();
  for (const m of members) { const f = folderOf(m.path); folders.set(f, (folders.get(f) ?? 0) + 1); }
  const top = [...folders.entries()].sort((a, b) => b[1] - a[1]).slice(0, 2)
    .map(([f, n]) => `${f} ${n}`).join(", ");
  console.log(`  ${String(c.size).padStart(4)}  ${c.label.padEnd(34)}  mostly: ${top}`);
}
// Coordinate sanity: a collapsed projection would put everything on one spot.
const xs = map.points.map((p) => p.x), ys = map.points.map((p) => p.y);
const spread = (a) => { const m = a.reduce((s, x) => s + x, 0) / a.length; return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / a.length); };
console.log(`\ncoordinate spread: x ${spread(xs).toFixed(3)}  y ${spread(ys).toFixed(3)} (both should be well above 0)`);

// Language-split baseline: if clustering by language alone scored as well, the
// map would be showing us German vs English rather than topic.
const isGerman = (t) => (t.match(/\b(der|die|das|und|nicht|ein|eine|mit|von|zu|ist|sind|f\u00fcr|auf|dem|den)\b/gi) || []).length >= 3;
const langAssign = notes.map((n) => (isGerman(n.basename + " " + (n.body || "")) ? 0 : 1));
const out = {
  model: MODEL, notes: map.points.length, clusters: map.clusters.length, buildMs: ms,
  k: map.clusters.length,
  purity: purity(assign), shuffledBaseline: purity(shuffled), languageBaseline: purity(langAssign),
  spreadX: +spread(xs).toFixed(3), spreadY: +spread(ys).toFixed(3),
  labels: [...map.clusters].sort((a, b) => b.size - a.size).map((c) => ({ label: c.label, size: c.size })),
};
console.log(`language-split baseline purity: ${out.languageBaseline}`);
mkdirSync("/Users/justus/obsidian_atomized_intermediary/lab/results", { recursive: true });
writeFileSync("/Users/justus/obsidian_atomized_intermediary/lab/results/v3-map.json", JSON.stringify(out, null, 1));
console.log("wrote lab/results/v3-map.json");
