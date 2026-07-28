// Template-crowding fix study. The measured bug: every daily note's top-10 is
// ~10/10 other daily notes. Verbatim-line dedup only moved it 9.92 -> 9.0, so the
// residual is NOT boilerplate lines — it is the key skeleton ("Fokus:", "Schlaf:",
// "- [x]") plus shared diary register, which is genuinely in the embedding.
//
// Four candidate fixes, measured on the same held-out-link protocol so a crowding
// fix that destroys recall is visible immediately:
//   A  raw                          (baseline)
//   B  verbatim-line df dedup       (what v3-levers tested)
//   C  B + key-skeleton strip       ("Fokus: x" -> "x", checkbox markers dropped)
//   D  C + template-group centroid subtraction   (post-processing, no re-embed)
//   E  D + per-group display cap (MMR-style)     (display layer)
//
//   node bench/v3-crowding.mjs
import { pipeline, env } from "@huggingface/transformers";
import { readFileSync, writeFileSync, readdirSync, statSync, mkdirSync } from "node:fs";
import { join, relative } from "node:path";
import { fisherYates } from "./shuffle.mjs";

env.allowLocalModels = false;
const VAULT = process.env.LAB_VAULT || "/Users/justus/obsidian_atomized_intermediary/lab/vault";
const OUT = "/Users/justus/obsidian_atomized_intermediary/lab/results/v3-crowding.json";
const MODEL = process.env.LAB_MODEL || "Xenova/paraphrase-multilingual-MiniLM-L12-v2";
const CACHE_DIR = process.env.HOME + "/.cache/srn-lab";
const CACHE = join(CACHE_DIR, `v3-${MODEL.replace(/[^a-z0-9]/gi, "_")}.json`);
const GROUP_CAP = Number(process.env.V3_GROUPCAP ?? 3); // max results per template group in top-10

const fold = (s) => s.toLowerCase().normalize("NFC")
  .replace(/ä/g, "ae").replace(/ö/g, "oe").replace(/ü/g, "ue").replace(/ß/g, "ss")
  .replace(/\s+/g, " ").trim();
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const l2 = (v) => { let s = 0; for (const x of v) s += x * x; s = Math.sqrt(s) || 1; return v.map((x) => x / s); };
function mulberry32(a) { return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function walk(dir) {
  const out = [];
  for (const name of readdirSync(dir)) {
    if (name.startsWith(".")) continue;
    const p = join(dir, name); const s = statSync(p);
    if (s.isDirectory()) out.push(...walk(p)); else if (name.endsWith(".md")) out.push(p);
  }
  return out;
}
const stripFront = (raw) => { const m = raw.match(/^---\n[\s\S]*?\n---\n?/); return m ? raw.slice(m[0].length) : raw; };

const manifest = JSON.parse(readFileSync("/Users/justus/obsidian_atomized_intermediary/lab/corpus-manifest.json", "utf8"));
const INDEXABLE = new Set(manifest.answer_paths.map((p) => "real/" + p));
const notes = [];
for (const abs of walk(VAULT)) {
  const rel = relative(VAULT, abs);
  if (rel.startsWith("real/") && !INDEXABLE.has(rel)) continue;
  const basename = rel.replace(/\.md$/, "").split("/").pop();
  if (/(^|\/)Attachments\//.test(rel) || /\.dup$/.test(basename)) continue;
  notes.push({ rel, basename, body: stripFront(readFileSync(abs, "utf8")), isDaily: /(^|\/)Daily\//.test(rel) });
}
const N = notes.length;

// ---------- df tables ----------
const LINE_DF_MAX = 3, KEY_DF_MAX = 3;
const lineDf = new Map(), keyDf = new Map();
const KEY_RE = /^\s*[-*+]?\s*([\p{L} ]{2,24}):\s*(.+)$/u;
for (const n of notes) {
  const seenL = new Set(), seenK = new Set();
  for (const line of n.body.split("\n")) {
    const f = fold(line);
    if (f.length >= 6) seenL.add(f);
    const m = line.match(KEY_RE);
    if (m) seenK.add(fold(m[1]));
  }
  for (const f of seenL) lineDf.set(f, (lineDf.get(f) ?? 0) + 1);
  for (const k of seenK) keyDf.set(k, (keyDf.get(k) ?? 0) + 1);
}

// ---------- text variants ----------
function textOf(n, mode) {
  let lines = n.body.split("\n");
  if (mode !== "raw") {
    lines = lines.filter((line) => {
      const f = fold(line);
      return f.length < 6 || (lineDf.get(f) ?? 0) < LINE_DF_MAX;
    });
  }
  if (mode === "skeleton") {
    lines = lines.map((line) => {
      // "- [x] foo" / "- [ ] foo" -> "foo"   (checkbox furniture)
      let out = line.replace(/^\s*[-*+]\s*\[[ xX]\]\s*/, "");
      // "Fokus: niedrig" -> "niedrig" when the KEY is template-common
      const m = out.match(KEY_RE);
      if (m && (keyDf.get(fold(m[1])) ?? 0) >= KEY_DF_MAX) out = m[2];
      return out;
    });
  }
  let body = lines.join("\n");
  if (!body.trim()) body = n.body;
  return (n.basename + "\n\n" + body).slice(0, 8000);
}

// ---------- embed ----------
let cache = {};
try { cache = JSON.parse(readFileSync(CACHE, "utf8")); if (cache.__model !== MODEL) cache = {}; } catch { }
const MODES = ["raw", "dedup", "skeleton"];
const wanted = notes.flatMap((n) => MODES.map((m) => textOf(n, m)));
const missing = [...new Set(wanted.filter((t) => !cache[t]))];
console.log(`${N} notes; ${missing.length} new texts to embed`);
if (missing.length) {
  const ex = await pipeline("feature-extraction", MODEL, { dtype: "fp32" });
  const isWhole = /jina-embeddings-v5/i.test(MODEL);
  for (let i = 0; i < missing.length; i += isWhole ? 1 : 16) {
    const batch = missing.slice(i, i + (isWhole ? 1 : 16));
    if (isWhole) {
      const o = await ex("Document: " + batch[0], { pooling: "none" });
      const d = o.dims, seq = d.length === 3 ? d[1] : d[0], dim = d.length === 3 ? d[2] : d[1];
      cache[batch[0]] = l2(Array.from(o.data.subarray((seq - 1) * dim, seq * dim))).map((x) => +x.toFixed(5));
      o.dispose?.();
    } else {
      const t = await ex(batch, { pooling: "mean", normalize: true });
      t.tolist().forEach((v, j) => { cache[batch[j]] = v.map((x) => +x.toFixed(5)); });
    }
  }
  cache.__model = MODEL;
  mkdirSync(CACHE_DIR, { recursive: true });
  writeFileSync(CACHE, JSON.stringify(cache));
}
function space(mode) {
  const V = notes.map((n) => cache[textOf(n, mode)]);
  const D = V[0].length, mean = new Array(D).fill(0);
  for (const v of V) for (let i = 0; i < D; i++) mean[i] += v[i];
  for (let i = 0; i < D; i++) mean[i] /= N;
  return V.map((v) => l2(v.map((x, i) => x - mean[i])));
}

// ---------- template groups ----------
// Two notes belong to the same template group when they share >= 2 boilerplate
// lines (df >= LINE_DF_MAX). Union-find over that relation. This is derived from
// the vault itself — no folder or tag assumptions.
const boiler = notes.map((n) => {
  const s = new Set();
  for (const line of n.body.split("\n")) {
    const f = fold(line);
    if (f.length >= 6 && (lineDf.get(f) ?? 0) >= LINE_DF_MAX) s.add(f);
  }
  return s;
});
const parent = [...Array(N).keys()];
const find = (x) => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
const union = (a, b) => { const ra = find(a), rb = find(b); if (ra !== rb) parent[ra] = rb; };
const byLine = new Map();
for (let i = 0; i < N; i++) for (const f of boiler[i]) { if (!byLine.has(f)) byLine.set(f, []); byLine.get(f).push(i); }
for (const [, members] of byLine) {
  for (let a = 0; a < members.length; a++) for (let b = a + 1; b < members.length; b++) {
    const shared = [...boiler[members[a]]].filter((x) => boiler[members[b]].has(x)).length;
    if (shared >= 2) union(members[a], members[b]);
  }
}
const group = notes.map((_, i) => find(i));
const groupSize = new Map();
for (const g of group) groupSize.set(g, (groupSize.get(g) ?? 0) + 1);
const templateGroups = [...groupSize.entries()].filter(([, c]) => c >= 3);
console.log(`template groups (>=3 members): ${templateGroups.length}; largest ${Math.max(0, ...templateGroups.map(([, c]) => c))}`);

// group-centroid subtraction: remove the shared direction inside each template group
function subtractGroupCentroid(C) {
  const out = C.map((v) => v.slice());
  for (const [g, count] of templateGroups) {
    const members = [];
    for (let i = 0; i < N; i++) if (group[i] === g) members.push(i);
    if (members.length < 3) continue;
    const D = C[0].length, cen = new Array(D).fill(0);
    for (const i of members) for (let d = 0; d < D; d++) cen[d] += C[i][d];
    const cn = l2(cen);
    for (const i of members) {
      const p = dot(C[i], cn);
      out[i] = l2(C[i].map((x, d) => x - p * cn[d]));
    }
  }
  return out;
}

// ---------- metrics ----------
const byFold = new Map(notes.map((n, i) => [fold(n.basename), i]));
const uniq = new Set();
for (let s = 0; s < N; s++) for (const m of notes[s].body.matchAll(/\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]/g)) {
  const name = m[1].trim(); if (!name) continue;
  const t = byFold.get(fold(name)); if (t === undefined || t === s) continue;
  uniq.add(`${Math.min(s, t)}-${Math.max(s, t)}`);
}
const edges = [...uniq].map((k) => k.split("-").map(Number));
const rnd = mulberry32(20260727);
const shuffled = fisherYates(edges, rnd);
const nHold = Math.floor(shuffled.length * 0.2);
const heldOut = shuffled.slice(0, nHold), train = shuffled.slice(nHold);
const adj = Array.from({ length: N }, () => new Set());
for (const [a, b] of train) { adj[a].add(b); adj[b].add(a); }

function rankFor(C, s, cap) {
  const cand = [];
  for (let t = 0; t < N; t++) { if (t === s || adj[s].has(t)) continue; cand.push([dot(C[s], C[t]), t]); }
  cand.sort((a, b) => b[0] - a[0]);
  if (!cap) return cand;
  const used = new Map(), out = [], spill = [];
  for (const e of cand) {
    const g = group[e[1]];
    const c = used.get(g) ?? 0;
    if (c < cap) { used.set(g, c + 1); out.push(e); } else spill.push(e);
    if (out.length >= 50) break;
  }
  return out.concat(spill);
}
function crowding(C, cap) {
  const daily = notes.map((n, i) => [n, i]).filter(([n]) => n.isDaily).map(([, i]) => i);
  if (!daily.length) return null;
  let acc = 0;
  for (const s of daily) acc += rankFor(C, s, cap).slice(0, 10).filter(([, t]) => notes[t].isDaily).length;
  return +(acc / daily.length).toFixed(2);
}
function recall(C, cap) {
  const bySrc = new Map();
  for (const [a, b] of heldOut) {
    if (!bySrc.has(a)) bySrc.set(a, []); bySrc.get(a).push(b);
    if (!bySrc.has(b)) bySrc.set(b, []); bySrc.get(b).push(a);
  }
  let hit = 0, tot = 0;
  for (const [s, targets] of bySrc) {
    const top = new Set(rankFor(C, s, cap).slice(0, 10).map((x) => x[1]));
    for (const t of new Set(targets)) { if (adj[s].has(t)) continue; tot++; if (top.has(t)) hit++; }
  }
  return +(hit / tot).toFixed(4);
}

const Sraw = space("raw"), Sded = space("dedup"), Sskel = space("skeleton");
const Sgrp = subtractGroupCentroid(Sskel);
// ALTERNATIVE grouping signal: a note's HEADING set. Obsidian already caches
// headings per file (metadataCache.getCache(path).headings), so grouping on them
// needs no stored signature, no body re-read and no index migration — where
// grouping on boilerplate lines needs all three. Only worth it if it fixes the
// crowding just as well, which is what this variant measures.
const headingsOf = (n) => {
  const out = new Set();
  for (const line of n.body.split("\n")) {
    const m = line.match(/^\s{0,3}(#{1,6})\s+(.+?)\s*$/);
    if (m) out.add(fold(m[2]));
  }
  return out;
};
const heads = notes.map(headingsOf);
const hParent = [...Array(N).keys()];
const hFind = (x) => { while (hParent[x] !== x) { hParent[x] = hParent[hParent[x]]; x = hParent[x]; } return x; };
const byHead = new Map();
for (let i = 0; i < N; i++) for (const h of heads[i]) { if (!byHead.has(h)) byHead.set(h, []); byHead.get(h).push(i); }
for (const [, members] of byHead) {
  for (let a = 0; a < members.length; a++) for (let b = a + 1; b < members.length; b++) {
    const shared = [...heads[members[a]]].filter((x) => heads[members[b]].has(x)).length;
    if (shared >= 2) { const ra = hFind(members[a]), rb = hFind(members[b]); if (ra !== rb) hParent[ra] = rb; }
  }
}
const hGroup = notes.map((_, i) => hFind(i));
const hSize = new Map();
for (const g of hGroup) hSize.set(g, (hSize.get(g) ?? 0) + 1);
function subtractByGroup(C, groupOf) {
  const out = C.map((v) => v.slice());
  const sizes = new Map();
  for (const g of groupOf) sizes.set(g, (sizes.get(g) ?? 0) + 1);
  for (const [g, count] of sizes) {
    if (count < 3) continue;
    const members = [];
    for (let i = 0; i < N; i++) if (groupOf[i] === g) members.push(i);
    const D = C[0].length, cen = new Array(D).fill(0);
    for (const i of members) for (let d = 0; d < D; d++) cen[d] += C[i][d];
    const cn = l2(cen);
    for (const i of members) { const p = dot(C[i], cn); out[i] = l2(C[i].map((x, d) => x - p * cn[d])); }
  }
  return out;
}
// Apply the SHIPPED size ceiling: single-linkage chains merge unrelated template
// families, so a component past this size is not evidence of a template.
const ceilingN = Math.max(30, Math.round(N * 0.05));
const hSizeMap = new Map();
for (const g of hGroup) hSizeMap.set(g, (hSizeMap.get(g) ?? 0) + 1);
const hGroupCapped = hGroup.map((g, i) => (hSizeMap.get(g) > ceilingN ? -1 - i : g));
const dailyGroup = hGroup[notes.findIndex((n) => n.isDaily)];
console.log(`ceiling ${ceilingN}; groups>=3: ${[...hSizeMap.values()].filter((c) => c >= 3).length}; ` +
  `rejected as too large: ${[...hSizeMap.values()].filter((c) => c > ceilingN).length}; ` +
  `daily group size ${hSizeMap.get(dailyGroup)} -> ${hSizeMap.get(dailyGroup) > ceilingN ? "REJECTED" : "kept"}`);
const ShGrp = subtractByGroup(Sraw, hGroupCapped);
const ShGrpDed = subtractByGroup(Sded, hGroupCapped);
console.log(`heading groups (>=3 members): ${[...hSize.values()].filter((c) => c >= 3).length}`);

const variants = [
  ["A raw", Sraw, 0],
  ["F heading-group centroid (raw)", ShGrp, 0],
  ["G heading-group centroid + dedup", ShGrpDed, 0],
  ["B verbatim-line dedup", Sded, 0],
  ["C + key-skeleton strip", Sskel, 0],
  ["D + group-centroid subtraction", Sgrp, 0],
  ["E + per-group display cap", Sgrp, GROUP_CAP],
  ["E' cap only (no centroid)", Sraw, GROUP_CAP],
];
const rows = variants.map(([label, C, cap]) => ({
  variant: label, dailyCrowding: crowding(C, cap), linkRecall10: recall(C, cap),
}));
console.log("\n==== TEMPLATE CROWDING ====");
console.log(`${N} notes | ${edges.length} edges | groupCap=${GROUP_CAP}`);
for (const r of rows) console.log(`  ${r.variant.padEnd(34)} crowding ${String(r.dailyCrowding).padStart(5)}/10   R@10 ${r.linkRecall10}`);
mkdirSync("/Users/justus/obsidian_atomized_intermediary/lab/results", { recursive: true });
writeFileSync(OUT, JSON.stringify({ model: MODEL, notes: N, groupCap: GROUP_CAP, templateGroups: templateGroups.length, rows }, null, 1));
console.log("\nwrote", OUT);
