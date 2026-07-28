// CHANNEL INDEPENDENCE STUDY.
//
// The v4-metric result said something useful by failing: massaging the content
// space bought +0.028 and nearly all of it vanished once the graph channel was
// present. The +0.08 win came from adding a channel that knows something the
// others do not. So the question for the architecture is not "how do we polish
// the embedding" but "which OTHER independent signals does a vault contain".
//
// This tests a third candidate: LEXICAL match (BM25). Dense embeddings blur exact
// tokens - identifiers, names, notation, numbers - which is precisely what a
// sparse term model is good at. Classic hybrid retrieval says the two compose;
// this measures whether that holds on a personal vault.
//
//   node bench/v4-channels.mjs
//
// Reported per channel and per combination: held-out link recall@10, plus the
// rank correlation between channels. Two channels that agree are one channel.
import { readFileSync, writeFileSync, readdirSync, statSync, mkdirSync } from "node:fs";
import { join, relative } from "node:path";
import { fisherYates } from "./shuffle.mjs";

const VAULT = process.env.LAB_VAULT || "/Users/justus/obsidian_atomized_intermediary/lab/vault";
const OUT = "/Users/justus/obsidian_atomized_intermediary/lab/results/v4-channels.json";
const MODEL = process.env.LAB_MODEL || "jinaai/jina-embeddings-v5-text-nano-text-matching";
const CACHE = join(process.env.HOME, ".cache/srn-lab", `v3-${MODEL.replace(/[^a-z0-9]/gi, "_")}.json`);

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

// ---------- channel 3: BM25 over note text ----------
// Wikilink targets are stripped first: a link IS the label, so leaving the target
// name in the text would let the lexical channel read the answer off the page.
const tokenise = (t) => fold(t.replace(/\[\[[^\]]*\]\]/g, " ")).split(/[^a-z0-9]+/).filter((w) => w.length >= 3);
const docs = notes.map((n) => tokenise(n.basename + " " + n.body));
const df = new Map();
for (const d of docs) for (const w of new Set(d)) df.set(w, (df.get(w) ?? 0) + 1);
const avgLen = docs.reduce((s, d) => s + d.length, 0) / N;
const K1 = 1.2, B = 0.75;
const tfMaps = docs.map((d) => { const m = new Map(); for (const w of d) m.set(w, (m.get(w) ?? 0) + 1); return m; });
const idf = (w) => Math.log(1 + (N - (df.get(w) ?? 0) + 0.5) / ((df.get(w) ?? 0) + 0.5));
function bm25(s, t) {
  // Symmetric: score doc t against the terms of doc s, using s's rarest terms so
  // a long note does not simply win by having more words.
  const terms = [...tfMaps[s].keys()].sort((a, b) => (df.get(a) ?? 0) - (df.get(b) ?? 0)).slice(0, 60);
  const dl = docs[t].length;
  let score = 0;
  for (const w of terms) {
    const f = tfMaps[t].get(w);
    if (!f) continue;
    score += idf(w) * (f * (K1 + 1)) / (f + K1 * (1 - B + B * dl / avgLen));
  }
  return score;
}

// ---------- channel 4: shared tags (the testable half of structureInfluence) ----------
// The plugin's existing structural boost is dominated by DIRECT links, which a
// link-prediction protocol excludes as candidates by construction, so that term
// cannot be evaluated here at all. Its other term, shared tags, can be: if tag
// overlap adds nothing on top of content+graph, then folding the boost into the
// fusion combiner is a refactor with no measurable payoff.
const tagsOf = (n) => {
  const out = new Set();
  const fm = n.body.match(/^---\n([\s\S]*?)\n---/);
  const src = (fm ? fm[1] : "") + " " + n.body;
  for (const m of src.matchAll(/(?:^|\s)#([\p{L}][\p{L}\p{N}_/-]*)/gu)) out.add(fold(m[1]));
  const tl = (fm ? fm[1] : "").match(/^tags:\s*\[([^\]]*)\]/m);
  if (tl) for (const t of tl[1].split(",")) { const x = fold(t.replace(/["']/g, "")); if (x) out.add(x); }
  const tb = (fm ? fm[1] : "").match(/^tags:\s*\n((?:\s*-\s*.+\n?)+)/m);
  if (tb) for (const line of tb[1].split("\n")) { const lm = line.match(/-\s*(.+)/); if (lm) out.add(fold(lm[1])); }
  return out;
};
const noteTags = notes.map(tagsOf);
const tagJaccard = (s, t) => {
  const a = noteTags[s], b = noteTags[t];
  if (!a.size || !b.size) return 0;
  let inter = 0;
  for (const x of a) if (b.has(x)) inter++;
  return inter / (a.size + b.size - inter);
};

// ---------- channel 2: Resource Allocation ----------
const RA = (s, t) => { let a = 0; for (const x of adj[s]) if (adj[t].has(x)) a += 1 / Math.max(1, adj[x].size); return a; };

// ---------- evaluation over arbitrary channel combinations ----------
const zf = (arr) => { const m = arr.reduce((a, b) => a + b, 0) / arr.length; const sd = Math.sqrt(arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length) || 1; return (x) => (x - m) / sd; };
const bySrc = new Map();
for (const [a, b] of heldOut) {
  if (!bySrc.has(a)) bySrc.set(a, []); bySrc.get(a).push(b);
  if (!bySrc.has(b)) bySrc.set(b, []); bySrc.get(b).push(a);
}
function recall(channels) {
  let hit = 0, tot = 0;
  for (const [s, targets] of bySrc) {
    const cand = [];
    for (let t = 0; t < N; t++) { if (t === s || adj[s].has(t)) continue; cand.push(t); }
    const parts = channels.map((ch) => {
      const raw = cand.map((t) => ch(s, t));
      const z = zf(raw);
      return raw.map(z);
    });
    const ranked = cand.map((t, i) => [parts.reduce((sum, p) => sum + p[i], 0), t]).sort((x, y) => y[0] - x[0]);
    const top = new Set(ranked.slice(0, 10).map((x) => x[1]));
    for (const t of new Set(targets)) { if (adj[s].has(t)) continue; tot++; if (top.has(t)) hit++; }
  }
  return +(hit / tot).toFixed(4);
}
const content = (s, t) => dot(C[s], C[t]);

// ---------- how independent are the channels really? ----------
// Spearman correlation of the three channels over a sample of candidate pairs.
// Highly correlated channels are one channel wearing two hats.
function spearman(f, g) {
  const sample = [];
  const r2 = mulberry32(11);
  for (let i = 0; i < 4000; i++) {
    const s = Math.floor(r2() * N), t = Math.floor(r2() * N);
    if (s !== t) sample.push([s, t]);
  }
  const rank = (fn) => {
    const vals = sample.map(([s, t]) => fn(s, t));
    const idx = vals.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]);
    const r = new Array(vals.length);
    idx.forEach(([, i], pos) => { r[i] = pos; });
    return r;
  };
  const a = rank(f), b = rank(g);
  const m = (arr) => arr.reduce((x, y) => x + y, 0) / arr.length;
  const ma = m(a), mb = m(b);
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < a.length; i++) { num += (a[i] - ma) * (b[i] - mb); da += (a[i] - ma) ** 2; db += (b[i] - mb) ** 2; }
  return +(num / Math.sqrt(da * db)).toFixed(3);
}

const rows = [
  { config: "content only", recall: recall([content]) },
  { config: "lexical only (BM25)", recall: recall([bm25]) },
  { config: "graph only (RA)", recall: recall([RA]) },
  { config: "content + graph", recall: recall([content, RA]) },
  { config: "content + lexical", recall: recall([content, bm25]) },
  { config: "content + graph + lexical", recall: recall([content, RA, bm25]) },
  { config: "content + graph + tags", recall: recall([content, RA, tagJaccard]) },
  { config: "content + tags", recall: recall([content, tagJaccard]) },
];
const corr = {
  content_vs_lexical: spearman(content, bm25),
  content_vs_graph: spearman(content, RA),
  lexical_vs_graph: spearman(bm25, RA),
  content_vs_tags: spearman(content, tagJaccard),
  tags_vs_graph: spearman(tagJaccard, RA),
};

console.log(`\n==== CHANNEL STUDY (held-out link recall@10) ====`);
console.log(`model ${MODEL} | ${N} notes | ${heldOut.length} held-out links\n`);
for (const r of rows) console.log(`  ${r.config.padEnd(28)} ${r.recall.toFixed(4)}`);
console.log(`\n  channel rank correlation (lower = more independent = more worth fusing)`);
for (const [k, v] of Object.entries(corr)) console.log(`    ${k.padEnd(22)} ${v}`);
mkdirSync("/Users/justus/obsidian_atomized_intermediary/lab/results", { recursive: true });
writeFileSync(OUT, JSON.stringify({ model: MODEL, notes: N, heldOut: heldOut.length, rows, corr }, null, 1));
console.log("\nwrote", OUT);
