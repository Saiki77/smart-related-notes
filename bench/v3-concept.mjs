// Concept search redesign. ARCHITECTURE section 5 proposed cluster-seeded
// prototype ranking; it was measured and FAILED (member-precision@10 an erratic
// 0.3-0.7 across five category queries, and 0.0 on the cold-start case). This
// harness tests replacements against the same ground truth.
//
//   node bench/v3-concept.mjs
//
// Ground truth
//   MOC members  : the user's five map-of-content notes; their outgoing links are
//                  a hand-curated member list. The MOC itself is never a result.
//   Physik       : a folder with NO map-of-content, so folder membership is the
//                  only signal. This is the cold-start case the feature exists for.
//
// Variants
//   base      plain cosine of the query against centered note vectors
//   gen       base minus a GENERALITY penalty (the note's own mean similarity to
//             the vault). Hypothesis: hubness correction failed as a global
//             ranking fix, but the concept-search failure mode IS a general note
//             outranking specific ones, so the same correction should pay here.
//   rocchio   query re-formed as q + beta * centroid(top-k), then re-ranked
//   roc+gen   both
//   graph     rocchio, then expanded along the link graph from the seed notes
//             (members of a category tend to link to each other)
import { pipeline, env } from "@huggingface/transformers";
import { readFileSync, writeFileSync, readdirSync, statSync, mkdirSync } from "node:fs";
import { join, relative } from "node:path";

env.allowLocalModels = false;
const VAULT = process.env.LAB_VAULT || "/Users/justus/obsidian_atomized_intermediary/lab/vault";
const OUT = "/Users/justus/obsidian_atomized_intermediary/lab/results/v3-concept.json";
const MODEL = process.env.LAB_MODEL || "Xenova/paraphrase-multilingual-MiniLM-L12-v2";
const CACHE_DIR = process.env.HOME + "/.cache/srn-lab";
const CACHE = join(CACHE_DIR, `v3-${MODEL.replace(/[^a-z0-9]/gi, "_")}.json`);
const ROC_K = Number(process.env.CS_ROCK ?? 5);      // pseudo-relevance depth
const ROC_BETA = Number(process.env.CS_BETA ?? 0.7); // weight of the feedback centroid
const GEN_LAMBDA = Number(process.env.CS_GEN ?? 1.0);

const fold = (s) => s.toLowerCase().normalize("NFC").replace(/ä/g, "ae").replace(/ö/g, "oe").replace(/ü/g, "ue").replace(/ß/g, "ss").replace(/\s+/g, " ").trim();
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const l2 = (v) => { let s = 0; for (const x of v) s += x * x; s = Math.sqrt(s) || 1; return v.map((x) => x / s); };
function walk(dir) {
  const out = [];
  for (const name of readdirSync(dir)) {
    if (name.startsWith(".")) continue;
    const p = join(dir, name), st = statSync(p);
    if (st.isDirectory()) out.push(...walk(p)); else if (name.endsWith(".md")) out.push(p);
  }
  return out;
}
const stripFront = (raw) => { const m = raw.match(/^---\n[\s\S]*?\n---\n?/); return m ? raw.slice(m[0].length) : raw; };
const noteText = (n) => (n.basename + "\n\n" + n.body).slice(0, 8000);

const manifest = JSON.parse(readFileSync("/Users/justus/obsidian_atomized_intermediary/lab/corpus-manifest.json", "utf8"));
const INDEXABLE = new Set(manifest.answer_paths.map((p) => "real/" + p));
const notes = [], mocs = new Map();
for (const abs of walk(VAULT)) {
  const rel = relative(VAULT, abs);
  if (rel.startsWith("real/") && !INDEXABLE.has(rel)) continue;
  const basename = rel.replace(/\.md$/, "").split("/").pop();
  if (/(^|\/)Attachments\//.test(rel) || /\.dup$/.test(basename)) continue;
  const body = stripFront(readFileSync(abs, "utf8"));
  if (/\bMOC\b/i.test(basename)) { mocs.set(basename, body); continue; } // GT source, never a result
  notes.push({ rel, basename, body });
}
const N = notes.length;

// ---------- embed ----------
let cache = {};
try { cache = JSON.parse(readFileSync(CACHE, "utf8")); if (cache.__model !== MODEL) cache = {}; } catch { }
const QUERIES = [
  ["Machine Learning MOC", "Machine Learning"],
  ["Statistik MOC", "Statistik"],
  ["Programmierung MOC", "Programmierung"],
  ["Datenbanken MOC", "Datenbanken"],
  ["Theoretische Informatik MOC", "Theoretische Informatik"],
];
// Title-only vectors. A category query is one or two words; a paraphrase model
// trained on sentence pairs is out of distribution comparing that to a whole
// note, but a TITLE is the same shape as the query. If this rescues the default
// model, concept search can ship without forcing the heavy model on anyone.
const titleText = (n) => n.basename;
const wanted = notes.map(noteText).concat(notes.map(titleText), QUERIES.map((q) => q[1]), ["Physik"]);
const missing = [...new Set(wanted.filter((t) => !cache[t]))];
console.log(`${N} notes; ${missing.length} to embed`);
if (missing.length) {
  const ex = await pipeline("feature-extraction", MODEL, { dtype: "fp32" });
  const whole = /jina-embeddings-v5/i.test(MODEL);
  for (let i = 0; i < missing.length; i += whole ? 1 : 16) {
    const batch = missing.slice(i, i + (whole ? 1 : 16));
    if (whole) {
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

// ---------- centered space ----------
const V = notes.map((n) => cache[noteText(n)]);
const D = V[0].length;
const mean = new Array(D).fill(0);
for (const v of V) for (let i = 0; i < D; i++) mean[i] += v[i];
for (let i = 0; i < D; i++) mean[i] /= N;
const center = (v) => l2(v.map((x, i) => x - mean[i]));
const C = V.map(center);
// Title space gets its OWN centroid: title vectors live in a different region of
// the space than whole-note vectors, so centering them with the note centroid
// would leave a systematic offset.
const TV = notes.map((n) => cache[titleText(n)]);
const tmean = new Array(D).fill(0);
for (const v of TV) for (let i = 0; i < D; i++) tmean[i] += v[i];
for (let i = 0; i < D; i++) tmean[i] /= N;
const centerT = (v) => l2(v.map((x, i) => x - tmean[i]));
const T = TV.map(centerT);
const byFold = new Map(notes.map((n, i) => [fold(n.basename), i]));

// ---------- generality: each note's mean similarity to the vault ----------
// A note that is close to EVERYTHING is a general note. That is exactly what
// outranks the specific members on a category query.
const generality = new Array(N).fill(0);
for (let i = 0; i < N; i++) {
  let acc = 0;
  for (let j = 0; j < N; j++) if (j !== i) acc += dot(C[i], C[j]);
  generality[i] = acc / (N - 1);
}

// ---------- link graph (for the expansion variant) ----------
const adj = Array.from({ length: N }, () => new Set());
for (let i = 0; i < N; i++) {
  for (const m of notes[i].body.matchAll(/\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]/g)) {
    const name = m[1].trim(); if (!name) continue;
    const j = byFold.get(fold(name));
    if (j === undefined || j === i) continue;
    adj[i].add(j); adj[j].add(i);
  }
}

// ---------- GT ----------
function mocMembers(mocBase) {
  const raw = mocs.get(mocBase) ?? "";
  const out = new Set();
  for (const m of raw.matchAll(/\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]/g)) {
    const j = byFold.get(fold(m[1].trim()));
    if (j !== undefined) out.add(j);
  }
  return out;
}
const tasks = QUERIES.map(([moc, q]) => ({ label: q, query: q, gt: mocMembers(moc) }));
tasks.push({
  label: "Physik (cold start, no MOC)", query: "Physik",
  gt: new Set(notes.map((n, i) => [n, i]).filter(([n]) => n.rel.startsWith("real/Physik/")).map(([, i]) => i)),
});

// ---------- rankers ----------
const rank = (score) => {
  const idx = [...Array(N).keys()];
  idx.sort((a, b) => score(b) - score(a));
  return idx;
};
function variants(qv, rawQuery) {
  const base = (i) => dot(qv, C[i]);
  const gen = (i) => base(i) - GEN_LAMBDA * generality[i];

  const rocchioVec = (scoreFn) => {
    const top = rank(scoreFn).slice(0, ROC_K);
    const acc = new Array(D).fill(0);
    for (const i of top) for (let d = 0; d < D; d++) acc[d] += C[i][d];
    const fb = l2(acc);
    return l2(qv.map((x, d) => x + ROC_BETA * fb[d]));
  };
  const qRoc = rocchioVec(base);
  const roc = (i) => dot(qRoc, C[i]);
  const rocGen = (i) => roc(i) - GEN_LAMBDA * generality[i];

  // graph expansion: notes linked to the pseudo-relevant seeds get a bonus,
  // normalised by degree so a hub does not win every category.
  const seeds = rank(rocGen).slice(0, ROC_K);
  const linkBonus = new Array(N).fill(0);
  for (const s of seeds) for (const nb of adj[s]) linkBonus[nb] += 1 / Math.sqrt(Math.max(1, adj[nb].size));
  const maxB = Math.max(1e-9, ...linkBonus);
  const graph = (i) => rocGen(i) + 0.5 * (linkBonus[i] / maxB);

  // Title channel: compare the short query to the short title, in title space.
  const qt = centerT(cache[rawQuery]);
  const title = (i) => dot(qt, T[i]);
  // Best of both: a note qualifies on either its title or its body, then the
  // same pseudo-relevance + graph expansion runs on the combined score.
  const both = (i) => Math.max(base(i), title(i));
  const qBoth = (() => {
    const top = rank(both).slice(0, ROC_K);
    const acc = new Array(D).fill(0);
    for (const i of top) for (let d = 0; d < D; d++) acc[d] += C[i][d];
    return l2(acc);
  })();
  const bothRoc = (i) => Math.max(base(i), title(i)) + ROC_BETA * dot(qBoth, C[i]);
  const seeds2 = rank(bothRoc).slice(0, ROC_K);
  const lb2 = new Array(N).fill(0);
  for (const s of seeds2) for (const nb of adj[s]) lb2[nb] += 1 / Math.sqrt(Math.max(1, adj[nb].size));
  const maxB2 = Math.max(1e-9, ...lb2);
  const bothAll = (i) => bothRoc(i) + 0.5 * (lb2[i] / maxB2);

  return { base, gen, roc, "roc+gen": rocGen, graph, title, "title+body": both, "all": bothAll };
}

const P_AT = 10;
const rows = [];
for (const t of tasks) {
  const qv = center(cache[t.query]);
  const vs = variants(qv, t.query);
  const row = { query: t.label, gtN: t.gt.size };
  for (const [name, fn] of Object.entries(vs)) {
    const ranked = rank(fn);
    const n = Math.min(P_AT, t.gt.size);
    const top = ranked.slice(0, n);
    row[name] = +(top.filter((i) => t.gt.has(i)).length / n).toFixed(3);
  }
  rows.push(row);
}
const names = ["base", "gen", "roc", "roc+gen", "graph", "title", "title+body", "all"];
const avg = {};
for (const nm of names) avg[nm] = +(rows.reduce((s, r) => s + r[nm], 0) / rows.length).toFixed(3);
const avgMoc = {};
for (const nm of names) avgMoc[nm] = +(rows.slice(0, 5).reduce((s, r) => s + r[nm], 0) / 5).toFixed(3);

console.log("\n==== CONCEPT SEARCH (member-precision@10) ====");
console.log(`model ${MODEL} | rocchio k=${ROC_K} beta=${ROC_BETA} | generality lambda=${GEN_LAMBDA}\n`);
console.log("  " + "query".padEnd(32) + names.map((n) => n.padStart(11)).join(""));
for (const r of rows) console.log("  " + r.query.slice(0, 31).padEnd(32) + names.map((n) => String(r[n]).padStart(11)).join(""));
console.log("  " + "AVERAGE (all 6)".padEnd(32) + names.map((n) => String(avg[n]).padStart(11)).join(""));
console.log("  " + "AVERAGE (5 MOC only)".padEnd(32) + names.map((n) => String(avgMoc[n]).padStart(11)).join(""));
mkdirSync("/Users/justus/obsidian_atomized_intermediary/lab/results", { recursive: true });
writeFileSync(OUT, JSON.stringify({ model: MODEL, rocK: ROC_K, beta: ROC_BETA, genLambda: GEN_LAMBDA, rows, avg, avgMoc }, null, 1));
console.log("\nwrote", OUT);
