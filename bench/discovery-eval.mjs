// Discovery-mechanisms harness: bridges & residual parallels vs plain cosine.
//
// Empirically measures two discovery mechanisms previously dismissed on theory:
//   (a) bridge-sem / bridge-link — A-B-C idea triangles: A,C unlinked and
//       semantically distant, both close to a mediator B (cosine legs / link legs).
//   (b) residual — cross-domain analogy via centroid-subtracted residuals
//       (leave-one-out domain centroid), vs the plain centered-cosine baseline.
//
// Ground truth: lab/discovery-gt.json — endpoint pairs from the real/Ideen
// connection notes. ALL real/Ideen/* notes are removed from the pool; the test
// is whether a mechanism can rediscover the endpoint pairs without them.
// NOTE (methodology review): bridge-link has structurally ZERO achievable
// recall on this GT — no non-Ideen note co-links any GT pair. It is evaluated
// on planted bridges (ground-truth.json br1-4) and judged precision only.
//
// Pre-registered PRIMARY readout: the per-pair unconditional rank table
// (a GT pair ineligible for a method = miss) with median rank per method and a
// paired sign test on per-pair ranks. recall@K tables are descriptive only.
//
//   node bench/discovery-eval.mjs
import { pipeline, env } from "@huggingface/transformers";
import { readFileSync, writeFileSync, readdirSync, statSync, existsSync } from "node:fs";
import { join, relative } from "node:path";

env.allowLocalModels = false;

const VAULT = process.env.LAB_VAULT || "/Users/justus/obsidian_atomized_intermediary/lab/vault";
const DGT_PATH = process.env.LAB_DGT || "/Users/justus/obsidian_atomized_intermediary/lab/discovery-gt.json";
const GT_PATH = process.env.LAB_GT || "/Users/justus/obsidian_atomized_intermediary/lab/ground-truth.json";
const RESULTS = "/Users/justus/obsidian_atomized_intermediary/lab/results";
const MODEL = process.env.LAB_MODEL || "Xenova/paraphrase-multilingual-MiniLM-L12-v2";
const DIST_SWEEP = [0.25, 0.35, 0.45];           // bridge "semantically distant" thresholds
const DIST_MAX = Number(process.env.BR_DISTMAX ?? 0.35); // threshold used for exports/judging
const MIN_DOM = Number(process.env.RS_MINDOM ?? 8);      // residual needs >= this many domain siblings
const BUDGET = [10, 25, 50, 100];                // recall@K (descriptive)
const JUDGE_TOP = Number(process.env.DV_JTOP ?? 12);     // per-method judged export
const JUDGE_TECH = Number(process.env.DV_JTECH ?? 8);    // per-method technical-only judged export

// ---------- helpers ----------
function walk(dir) {
  const out = [];
  for (const name of readdirSync(dir)) {
    if (name.startsWith(".")) continue;
    const p = join(dir, name);
    const s = statSync(p);
    if (s.isDirectory()) out.push(...walk(p));
    else if (name.endsWith(".md")) out.push(p);
  }
  return out;
}
const fold = (s) => s.toLowerCase().normalize("NFC")
  .replace(/ä/g, "ae").replace(/ö/g, "oe").replace(/ü/g, "ue").replace(/ß/g, "ss")
  .replace(/\s+/g, " ").trim();
// Frontmatter: tags AND aliases, both inline-array and block-list forms.
// (409 vault notes use block-style aliases; rn-eval's inline-only parser misses
// them and corrupts the "unlinked" filter — implementation-review fix.)
function parseFront(raw) {
  const m = raw.match(/^---\n([\s\S]*?)\n---\n?/);
  const tags = new Set(), aliases = [];
  if (m) {
    const fm = m[1];
    const grab = (key, into) => {
      const inline = fm.match(new RegExp(`^${key}:\\s*\\[([^\\]]*)\\]`, "m"));
      if (inline) { for (const t of inline[1].split(",")) { const x = t.trim().replace(/^["'#]+|["']$/g, ""); if (x) into(x); } return; }
      const block = fm.match(new RegExp(`^${key}:\\s*\\n((?:\\s*-\\s*.+\\n?)+)`, "m"));
      if (block) for (const line of block[1].split("\n")) { const lm = line.match(/-\s*(.+)/); if (lm) { const x = lm[1].trim().replace(/^["'#]+|["']$/g, ""); if (x) into(x); } }
    };
    grab("tags", (x) => tags.add(x.toLowerCase().split("/")[0]));
    grab("aliases", (x) => aliases.push(x));
  }
  return { tags, aliases, body: m ? raw.slice(m[0].length) : raw };
}
function wikilinks(body) {
  const out = new Set();
  const re = /\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]/g;
  let m;
  while ((m = re.exec(body)) !== null) out.add(fold(m[1]));
  return out;
}
const l2 = (v) => { let s = 0; for (const x of v) s += x * x; s = Math.sqrt(s) || 1; return v.map((x) => x / s); };
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const seg = new Intl.Segmenter(undefined, { granularity: "sentence" });
function sentencesOf(body) {
  const out = [];
  const clean = body.replace(/```[\s\S]*?```/g, " ").replace(/%%[\s\S]*?%%/g, " ")
    .replace(/<[^>]+>/g, " ").replace(/\[\[([^\]|]+)\|([^\]]+)\]\]/g, "$2").replace(/\[\[([^\]]+)\]\]/g, "$1")
    .replace(/^\s{0,3}(#{1,6}|>|[-*+]|\d+\.)\s+/gm, "").replace(/[*_~`]/g, "");
  for (const line of clean.split("\n")) {
    if (/^\s*\|/.test(line)) continue;
    for (const { segment } of seg.segment(line)) {
      const s = segment.replace(/\s+/g, " ").trim();
      if (s.length >= 8) out.push(s);
    }
  }
  return out;
}
// crude language probe with abstain (implementation-review fix)
const DE_STOP = new Set("der die das und ist nicht mit ein eine von zu den dem des im auf fuer als auch wird sind werden bei oder wenn dass sich nur noch aber wie nach man kann muss".split(" "));
const EN_STOP = new Set("the and of to in is that it for on with as are was this be by from or an at not but have has which can will would when what how".split(" "));
function langOf(body) {
  let de = 0, en = 0;
  for (const w of fold(body).split(/[^a-z]+/)) { if (DE_STOP.has(w)) de++; if (EN_STOP.has(w)) en++; }
  if (de >= 5 && de >= 2 * en) return "de";
  if (en >= 5 && en >= 2 * de) return "en";
  return "mixed";
}
function firstPara(body) {
  for (const block of body.split(/\n\s*\n/)) {
    const t = block.replace(/^#+.*$/gm, "").replace(/\[\[([^\]|]+)\|([^\]]+)\]\]/g, "$2").replace(/\[\[([^\]]+)\]\]/g, "$1")
      .replace(/[>*_~`]/g, "").replace(/\s+/g, " ").trim();
    if (t.length >= 60) return t.slice(0, 320);
  }
  return "";
}

// ---------- load vault ----------
const dgt = JSON.parse(readFileSync(DGT_PATH, "utf8"));
const gt2 = JSON.parse(readFileSync(GT_PATH, "utf8"));
const manifest = JSON.parse(readFileSync("/Users/justus/obsidian_atomized_intermediary/lab/corpus-manifest.json", "utf8"));
const INDEXABLE_REAL = new Set(manifest.answer_paths.map((p) => "real/" + p));
const files = walk(VAULT).filter((abs) => {
  const rel = relative(VAULT, abs);
  return (!rel.startsWith("real/") || INDEXABLE_REAL.has(rel)) && !/(^|\/)Attachments\//.test(rel);
});

const HUB_NAMES = new Set(["zettelkasten index", "mathematik uebersicht", "vault insights (smart related notes)", "untitled"].map(fold));
// real/Personal split: these are project/study notes, not novel material (intent-review fix)
const PERSONAL_TECH = new Set(["Smart Related Notes", "SurfacerAI", "LectureSyncAI", "Proofwork", "Proofwork Security On The Self Writing Web", "Projects", "T2000 Studienarbeit", "UNI Themen", "Test Note", "GOA"].map(fold));
// GOA stays novel — listed here only so the explicit override below is auditable:
PERSONAL_TECH.delete(fold("GOA"));
const isPersonalTech = (base) => PERSONAL_TECH.has(fold(base));

const notes = [];
for (const abs of files) {
  const rel = relative(VAULT, abs);
  const basename = rel.replace(/\.md$/, "").split("/").pop();
  const raw = readFileSync(abs, "utf8");
  const { tags, aliases, body } = parseFront(raw);
  const isDup = /\.dup\.md$/.test(rel);
  const isDaily = /^\d{4}-\d{2}-\d{2}$/.test(basename);
  const isIdeen = rel.startsWith("real/Ideen/");
  const isHub = /\bMOC\b/i.test(basename) || HUB_NAMES.has(fold(basename)) || body.trim().length < 40;
  let domain;
  const top = rel.split("/")[0];
  if (top !== "real") domain = top;
  else {
    const sub = rel.split("/")[1];
    if (!rel.includes("/", 5)) domain = "real/misc";
    else if (sub === "Personal") domain = isPersonalTech(basename) ? "real/Personal-tech" : "real/Personal-novel";
    else domain = "real/" + sub;
  }
  notes.push({ rel, basename, tags, aliases, body, domain, isDup, isDaily, isIdeen, isHub, links: wikilinks(body), lang: langOf(body) });
}
// pool = embedded universe. Ideen/dups/dailies are OUT of everything, incl. the
// global centroid and domain centroids (implementation-review fix). Hubs stay in
// the pool (mediator role, flagged) but are never pair endpoints.
const pool = notes.filter((n) => !n.isIdeen && !n.isDup && !n.isDaily);
console.log(`vault notes ${notes.length}; pool ${pool.length} (excluded: ${notes.length - pool.length} = ideen ${notes.filter(n => n.isIdeen).length}, dup ${notes.filter(n => n.isDup).length}, daily ${notes.filter(n => n.isDaily && !n.isIdeen).length})`);

// name resolution: basename AND aliases -> [notes]; collisions kept as lists
// (Skalarprodukt exists twice; last-wins maps shadow notes — impl-review fix)
const byName = new Map();
for (const n of pool) {
  for (const key of [fold(n.basename), ...n.aliases.map(fold)]) {
    if (!key) continue;
    if (!byName.has(key)) byName.set(key, []);
    byName.get(key).push(n);
  }
}
const linkTargets = (n) => { // notes n links to, alias-aware, all collision matches
  const out = new Set();
  for (const t of n.links) for (const m of byName.get(t) || []) out.add(m);
  return out;
};
const outLinks = new Map(pool.map((n) => [n, linkTargets(n)]));
const linked = (a, b) => outLinks.get(a).has(b) || outLinks.get(b).has(a);

// ---------- embeddings ----------
const cachePath = process.env.AQ_CACHE || "/private/tmp/claude-501/-Users-justus-obsidian-atomized-intermediary/fdd9de71-217a-43e1-9ca7-fb239cfd5cb4/scratchpad/aq-embed-cache.json";
let cache = {};
try { cache = JSON.parse(readFileSync(cachePath, "utf8")); if (cache.__model !== MODEL) cache = {}; } catch { }
for (const n of pool) n.sents = [n.basename, ...sentencesOf(n.body).slice(0, 60)];
const allTexts = [...new Set(pool.flatMap((n) => n.sents))];
const missing = allTexts.filter((t) => !cache[t]);
const extractor = await pipeline("feature-extraction", MODEL, { dtype: "fp32" });
if (missing.length) {
  console.log(`embedding ${missing.length} new texts (${allTexts.length - missing.length} cached) ...`);
  for (let i = 0; i < missing.length; i += 32) {
    const batch = missing.slice(i, i + 32);
    const t = await extractor(batch, { pooling: "mean", normalize: true });
    const vs = t.tolist();
    batch.forEach((txt, j) => { cache[txt] = vs[j]; });
  }
  cache.__model = MODEL;
  try { writeFileSync(cachePath, JSON.stringify(cache)); } catch { }
}
for (const n of pool) {
  const vs = n.sents.map((s) => cache[s]).filter(Boolean);
  if (!vs.length) { n.mean = null; continue; }
  const d = vs[0].length, m = new Array(d).fill(0);
  for (const v of vs) for (let i = 0; i < d; i++) m[i] += v[i];
  n.mean = l2(m);
}
const emb = pool.filter((n) => n.mean);
const D = emb[0].mean.length;
{ // corpus centering (dailies/dups/ideen already excluded from this centroid)
  const c = new Array(D).fill(0);
  for (const n of emb) for (let i = 0; i < D; i++) c[i] += n.mean[i];
  const cN = l2(c);
  for (const n of emb) { const p = dot(n.mean, cN); n.cmean = l2(n.mean.map((x, i) => x - p * cN[i])); }
}

// ---------- residual vectors (leave-one-out domain centroid) ----------
const domSum = new Map(), domN = new Map();
for (const n of emb) {
  if (!domSum.has(n.domain)) { domSum.set(n.domain, new Array(D).fill(0)); domN.set(n.domain, 0); }
  const s = domSum.get(n.domain);
  for (let i = 0; i < D; i++) s[i] += n.cmean[i];
  domN.set(n.domain, domN.get(n.domain) + 1);
}
for (const n of emb) {
  const k = domN.get(n.domain);
  if (k < MIN_DOM) { n.resid = null; continue; }        // degenerate domain -> N/A, never a fake rank
  const s = domSum.get(n.domain);
  const loo = n.cmean.map((x, i) => (s[i] - x) / (k - 1)); // leave-one-out centroid
  n.resid = l2(n.cmean.map((x, i) => x - loo[i]));
}
const domTable = [...domN.entries()].map(([d, k]) => ({ domain: d, n: k, residual_ok: k >= MIN_DOM }));
console.log("domains:", domTable.map((r) => `${r.domain}:${r.n}${r.residual_ok ? "" : "(res-N/A)"}`).join("  "));

// ---------- candidate universe U ----------
const endpoints = emb.filter((n) => !n.isHub);
const N = endpoints.length;
endpoints.forEach((n, i) => { n.idx = i; });
const S = new Float32Array(N * N); // cosine of centered means
for (let i = 0; i < N; i++) { S[i * N + i] = 1; for (let j = i + 1; j < N; j++) { const s = dot(endpoints[i].cmean, endpoints[j].cmean); S[i * N + j] = s; S[j * N + i] = s; } }
const pairs = []; // {i,j,cos}
for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
  const a = endpoints[i], b = endpoints[j];
  if (a.domain === b.domain) continue;
  if (linked(a, b)) continue;
  pairs.push({ i, j, cos: S[i * N + j] });
}
console.log(`endpoints ${N}; cross-domain unlinked pairs |U| = ${pairs.length}`);

// ---------- method scores ----------
// plain
const plainRanked = pairs.map((p, k) => [k, p.cos]).sort((x, y) => y[1] - x[1]);
// residual
const residRanked = pairs.map((p, k) => {
  const a = endpoints[p.i], b = endpoints[p.j];
  return (a.resid && b.resid) ? [k, dot(a.resid, b.resid)] : null;
}).filter(Boolean).sort((x, y) => y[1] - x[1]);
// bridge-sem: mediator search once for all pairs under the widest threshold.
// Mediator pool: everything embedded except the endpoints themselves; hubs
// allowed (flagged), plants flagged, sensitivity pass excludes plants.
const PLANT_MEDIATORS = new Set([
  "Projects/Exploration vs Exploitation.md", "Projects/Zufall im Plot.md",
  "Projects/Caching Essay.md", "Projects/Programmiersprachen Essay.md",
  "Aetherfall/Schreibprozess/Revision Strategie.md", "Aetherfall/Schreibprozess/Plot Struktur.md",
  "Aetherfall/Charaktere/Charakterkern.md",
]);
const TRAP = gt2.generic_bridge_trap?.file;
const mediators = emb; // includes hubs; excludes ideen/dups/dailies by pool construction
const medCos = new Map(); // mediator -> Float32Array over endpoints
for (const m of mediators) {
  const row = new Float32Array(N);
  for (let i = 0; i < N; i++) row[i] = m === endpoints[i] ? -1 : dot(m.cmean, endpoints[i].cmean);
  medCos.set(m, row);
}
const WIDE = Math.max(...DIST_SWEEP);
const bridgeInfo = new Map(); // pairIdx -> {best, m, bestNP, mNP}  (NP = no-plant sensitivity)
for (let k = 0; k < pairs.length; k++) {
  const p = pairs[k];
  if (p.cos > WIDE) continue;
  let best = -1, bm = null, bestNP = -1, bmNP = null;
  for (const m of mediators) {
    if (m === endpoints[p.i] || m === endpoints[p.j]) continue;
    const row = medCos.get(m);
    const leg = Math.min(row[p.i], row[p.j]);
    if (leg > best) { best = leg; bm = m; }
    if (leg > bestNP && !PLANT_MEDIATORS.has(m.rel)) { bestNP = leg; bmNP = m; }
  }
  bridgeInfo.set(k, { best, m: bm, bestNP, mNP: bmNP });
}
const bridgeRankedAt = (thr, noPlant = false) => [...bridgeInfo.entries()]
  .filter(([k]) => pairs[k].cos <= thr)
  .map(([k, v]) => [k, noPlant ? v.bestNP : v.best])
  .sort((x, y) => y[1] - x[1]);
// bridge-link: mediator must be link-adjacent (alias-aware, either direction) to both.
const adj = new Map(); // note -> Set of link-adjacent pool notes
for (const n of pool) {
  const s = new Set(outLinks.get(n));
  adj.set(n, s);
}
for (const n of pool) for (const t of outLinks.get(n)) adj.get(t)?.add(n);
const linkRanked = [];
for (let k = 0; k < pairs.length; k++) {
  const p = pairs[k];
  if (p.cos > DIST_MAX) continue;
  const a = endpoints[p.i], b = endpoints[p.j];
  const small = adj.get(a).size < adj.get(b).size ? a : b, other = small === a ? b : a;
  let best = -1, bm = null;
  for (const m of adj.get(small)) {
    if (m === a || m === b || m.isIdeen || m.isDup || m.isDaily || !m.mean) continue;
    if (!adj.get(other).has(m)) continue;
    const leg = Math.min(dot(m.cmean, a.cmean), dot(m.cmean, b.cmean));
    if (leg > best) { best = leg; bm = m; }
  }
  if (bm) linkRanked.push([k, best, bm]);
}
linkRanked.sort((x, y) => y[1] - x[1]);

// ---------- GT evaluation ----------
const pairIdxByKey = new Map(pairs.map((p, k) => [[endpoints[p.i].rel, endpoints[p.j].rel].sort().join(">>"), k]));
const relByFold = new Map(emb.map((n) => [n.rel, n]));
const keyOf = (x, y) => [x, y].sort().join(">>");
const rankIn = (rankedList, k) => { const r = rankedList.findIndex((e) => e[0] === k); return r === -1 ? null : r + 1; };
const OVERLAPS = { dg01: "br1/an1", dg02: "br1/an1", dg17: "br1/an1", dg09: "an3", dg10: "an3", dg11: "br2" };

function evalPair(id, xf, yf, tier, relation) {
  const a = relByFold.get(xf), b = relByFold.get(yf);
  const row = { id, tier, x: xf, y: yf, relation, overlaps: OVERLAPS[id] || null };
  if (!a || !b) { row.status = "missing-from-pool"; return row; }
  if (a.isHub || b.isHub) { row.status = "hub-endpoint"; return row; }
  if (a.domain === b.domain) { row.status = "excluded-same-domain"; return row; }
  if (linked(a, b)) { row.status = "already-linked"; return row; }
  const k = pairIdxByKey.get(keyOf(xf, yf));
  if (k === undefined) { row.status = "not-in-U"; return row; }
  row.status = "in-U";
  row.cos = +pairs[k].cos.toFixed(3);
  row.plain = rankIn(plainRanked, k);
  row.residual = (endpoints[pairs[k].i].resid && endpoints[pairs[k].j].resid) ? rankIn(residRanked, k) : null;
  row.residual_na = row.residual === null && !(endpoints[pairs[k].i].resid && endpoints[pairs[k].j].resid) ? "domain<MIN_DOM" : undefined;
  row.bridge_sem = {}; row.bridge_sem_noplant = {};
  for (const thr of DIST_SWEEP) {
    row.bridge_sem[thr] = pairs[k].cos <= thr ? rankIn(bridgeRankedAt(thr), k) : "ineligible(cos)";
    row.bridge_sem_noplant[thr] = pairs[k].cos <= thr ? rankIn(bridgeRankedAt(thr, true), k) : "ineligible(cos)";
  }
  const bi = bridgeInfo.get(k);
  if (bi && pairs[k].cos <= WIDE) row.mediator = { rel: bi.m.rel, leg: +bi.best.toFixed(3), plant: PLANT_MEDIATORS.has(bi.m.rel), hub: !!bi.m.isHub, noplant_mediator: bi.mNP?.rel, noplant_leg: bi.bestNP >= 0 ? +bi.bestNP.toFixed(3) : null };
  row.bridge_link = rankIn(linkRanked, k);
  return row;
}
const gtRows = dgt.pairs.map((p) => evalPair(p.id, p.x_file, p.y_file, p.tier, p.relation));
const plantedRows = [
  ...(gt2.bridges || []).map((br) => evalPair(br.id, br.a_file, br.c_file, "planted-bridge", br.id)),
  ...(gt2.analogies || []).map((an) => evalPair(an.id, an.x_file, an.y_file, "planted-analogy", an.relation)),
];

// primary readout: medians + sign tests over in-U dg pairs (unconditional: miss = Infinity)
const inU = gtRows.filter((r) => r.status === "in-U");
const uncond = (r, method) => {
  if (method === "plain") return r.plain ?? Infinity;
  if (method === "residual") return r.residual ?? Infinity;
  if (method === "bridge_sem") { const v = r.bridge_sem?.[DIST_MAX]; return typeof v === "number" ? v : Infinity; }
  if (method === "bridge_link") return r.bridge_link ?? Infinity;
};
const median = (xs) => { const s = [...xs].sort((a, b) => a - b); return s.length ? (s[s.length >> 1] === Infinity ? "miss" : s[s.length >> 1]) : null; };
const signTest = (m1, m2) => {
  let w1 = 0, w2 = 0;
  for (const r of inU) { const a = uncond(r, m1), b = uncond(r, m2); if (a < b) w1++; else if (b < a) w2++; }
  const n = w1 + w2;
  let p = 0; // two-sided exact binomial
  for (let k = Math.max(w1, w2); k <= n; k++) { let c = 1; for (let i = 0; i < k; i++) c = c * (n - i) / (i + 1); p += c / Math.pow(2, n); }
  return { [m1]: w1, [m2]: w2, ties: inU.length - n, p_two_sided: +Math.min(1, 2 * p).toFixed(4) };
};
const recallAt = (method) => Object.fromEntries(BUDGET.map((K) => {
  const hit = inU.filter((r) => uncond(r, method) <= K).length;
  return [K, `${hit}/${inU.length}`];
}));

// ---------- language probe on every method's top-50 (control incl. plain) ----------
const langProbe = {};
for (const [name, list] of [["plain", plainRanked], ["residual", residRanked], ["bridge_sem", bridgeRankedAt(DIST_MAX)], ["bridge_link", linkRanked]]) {
  const top = list.slice(0, 50).map(([k]) => pairs[k]);
  let cross = 0, abstain = 0;
  for (const p of top) {
    const la = endpoints[p.i].lang, lb = endpoints[p.j].lang;
    if (la === "mixed" || lb === "mixed") abstain++;
    else if (la !== lb) cross++;
  }
  langProbe[name] = { top: top.length, cross_language: cross, abstain };
}

// ---------- judged-precision export ----------
function bestSentencePair(a, b) {
  const va = a.sents.map((s) => cache[s]).filter(Boolean).map(l2);
  const vb = b.sents.map((s) => cache[s]).filter(Boolean).map(l2);
  let best = -1, bi = 0, bj = 0;
  for (let i = 0; i < va.length; i++) for (let j = 0; j < vb.length; j++) {
    const s = dot(va[i], vb[j]);
    if (s > best) { best = s; bi = i; bj = j; }
  }
  return { asent: a.sents[bi], bsent: b.sents[bj], cos: +best.toFixed(3) };
}
const isTech = (n) => !["Aetherfall", "real/Personal-novel"].includes(n.domain);
const judgeItems = new Map(); // pairKey -> item
function addJudge(list, method, count, techOnly = false) {
  let added = 0;
  for (const entry of list) {
    if (added >= count) break;
    const k = entry[0], p = pairs[k];
    const a = endpoints[p.i], b = endpoints[p.j];
    if (techOnly && !(isTech(a) && isTech(b))) continue;
    const key = keyOf(a.rel, b.rel);
    if (!judgeItems.has(key)) {
      const ev = bestSentencePair(a, b);
      judgeItems.set(key, {
        key, afile: a.rel, bfile: b.rel, tech_pair: isTech(a) && isTech(b),
        apara: firstPara(a.body), bpara: firstPara(b.body), evidence: ev,
        methods: [], scores: {}, mediator: null,
        gt: dgt.pairs.some((g) => keyOf(g.x_file, g.y_file) === key) || null,
      });
    }
    const it = judgeItems.get(key);
    if (!it.methods.includes(method + (techOnly ? ":tech" : ""))) it.methods.push(method + (techOnly ? ":tech" : ""));
    it.scores[method] = +entry[1].toFixed(3);
    if (method.startsWith("bridge_sem")) { const bi = bridgeInfo.get(k); it.mediator = { rel: bi.m.rel, plant: PLANT_MEDIATORS.has(bi.m.rel), hub: !!bi.m.isHub }; }
    if (method === "bridge_link") it.mediator = { rel: entry[2].rel, link: true };
    added++;
  }
}
addJudge(plainRanked, "plain", JUDGE_TOP);
addJudge(residRanked, "residual", JUDGE_TOP);
addJudge(bridgeRankedAt(DIST_MAX), "bridge_sem", JUDGE_TOP);
addJudge(linkRanked, "bridge_link", JUDGE_TOP);
addJudge(plainRanked, "plain", JUDGE_TECH, true);
addJudge(residRanked, "residual", JUDGE_TECH, true);
addJudge(bridgeRankedAt(DIST_MAX), "bridge_sem", JUDGE_TECH, true);
addJudge(linkRanked, "bridge_link", JUDGE_TECH, true);
// deterministic shuffle (mulberry32, fixed seed) so judge order carries no method signal
let seed = 0x5eed;
const rand = () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
const judgeList = [...judgeItems.values()];
for (let i = judgeList.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [judgeList[i], judgeList[j]] = [judgeList[j], judgeList[i]]; }
judgeList.forEach((it, i) => { it.jid = `J${String(i + 1).padStart(2, "0")}`; });

// ---------- outputs ----------
const summary = {
  model: MODEL,
  params: { DIST_SWEEP, DIST_MAX, MIN_DOM, BUDGET, JUDGE_TOP, JUDGE_TECH },
  pool: { vault: notes.length, pool: pool.length, embedded: emb.length, endpoints: N, pairs_U: pairs.length },
  domains: domTable,
  note_bridge_link: "structurally 0 recall on Ideen GT (no non-Ideen co-linker exists); evaluate on planted bridges + judged precision only",
  gt_partition: Object.fromEntries(["in-U", "excluded-same-domain", "already-linked", "missing-from-pool", "hub-endpoint"].map((s) => [s, gtRows.filter((r) => r.status === s).map((r) => r.id)])),
  primary_readout: {
    n_in_U: inU.length,
    median_unconditional_rank: Object.fromEntries(["plain", "residual", "bridge_sem", "bridge_link"].map((m) => [m, median(inU.map((r) => uncond(r, m)))])),
    sign_tests: {
      plain_vs_residual: signTest("plain", "residual"),
      plain_vs_bridge_sem: signTest("plain", "bridge_sem"),
      residual_vs_bridge_sem: signTest("residual", "bridge_sem"),
    },
  },
  recall_descriptive: Object.fromEntries(["plain", "residual", "bridge_sem", "bridge_link"].map((m) => [m, recallAt(m)])),
  recall_splits: {
    primary_tier: inU.filter((r) => r.tier === "primary").length,
    secondary_tier: inU.filter((r) => r.tier === "secondary").length,
    tech_tech_pairs: inU.filter((r) => { const a = relByFold.get(r.x), b = relByFold.get(r.y); return isTech(a) && isTech(b); }).map((r) => r.id),
    caveat: "tech<->tech Ideen recall rests on the pairs listed above (tiny n) — precision-only evidence for the technical use case; see judged lists",
  },
  gt_rows: gtRows,
  planted_rows: plantedRows,
  language_probe_top50: langProbe,
  trap_mediator_share_top100: bridgeRankedAt(DIST_MAX).slice(0, 100).filter(([k]) => bridgeInfo.get(k)?.m?.rel === TRAP).length,
  judged_items: judgeList.length,
};
writeFileSync(join(RESULTS, "discovery-eval.json"), JSON.stringify(summary, null, 2));
writeFileSync(join(RESULTS, "discovery-judging.json"), JSON.stringify({ comment: "Blind-panel input. Judge-facing view must strip methods/scores/mediator/gt fields.", items: judgeList }, null, 2));
const md = ["# Discovery mechanisms — top pairs per method", ""];
for (const [name, list] of [["plain cosine", plainRanked], ["residual (LOO domain centroid)", residRanked], [`bridge-sem (cos<=${DIST_MAX})`, bridgeRankedAt(DIST_MAX)], ["bridge-link", linkRanked]]) {
  md.push(`## ${name}`);
  for (const e of list.slice(0, 15)) {
    const p = pairs[e[0]], a = endpoints[p.i], b = endpoints[p.j];
    const bi = name.startsWith("bridge-sem") ? bridgeInfo.get(e[0]) : null;
    md.push(`- ${e[1].toFixed(3)}  \`${a.rel}\` <> \`${b.rel}\` (cos ${p.cos.toFixed(2)})${bi ? ` via \`${bi.m.rel}\`${PLANT_MEDIATORS.has(bi.m.rel) ? " [PLANT]" : ""}${bi.m.isHub ? " [HUB]" : ""}` : ""}${name === "bridge-link" ? ` via \`${e[2].rel}\`` : ""}`);
  }
  md.push("");
}
writeFileSync(join(RESULTS, "discovery-review.md"), md.join("\n"));

console.log("\n==== PRIMARY: per-pair unconditional ranks (in-U dg pairs) ====");
for (const r of inU) console.log(`${r.id} ${r.tier[0]} cos=${r.cos}  plain#${r.plain}  resid#${r.residual ?? "N/A"}  brSem#${JSON.stringify(r.bridge_sem[DIST_MAX])}  brLink#${r.bridge_link ?? "-"}  ${r.overlaps ? "[overlaps " + r.overlaps + "]" : ""}`);
console.log("medians:", JSON.stringify(summary.primary_readout.median_unconditional_rank));
console.log("sign tests:", JSON.stringify(summary.primary_readout.sign_tests));
console.log("recall (descriptive):", JSON.stringify(summary.recall_descriptive));
console.log("GT partition:", JSON.stringify(summary.gt_partition));
console.log("\n==== planted bridges/analogies ====");
for (const r of plantedRows) console.log(`${r.id} [${r.status}] cos=${r.cos ?? "-"}  plain#${r.plain ?? "-"}  resid#${r.residual ?? "-"}  brSem#${r.bridge_sem ? JSON.stringify(r.bridge_sem[DIST_MAX]) : "-"}  brLink#${r.bridge_link ?? "-"}`);
console.log("\nlanguage probe:", JSON.stringify(langProbe));
console.log(`judged items exported: ${judgeList.length}`);
console.log("wrote discovery-eval.json, discovery-judging.json, discovery-review.md");
