// C (tag-free concept search), D (cross-language twins), E (vault cartography)
// under jina-v5-nano. See lab/EXPERIMENT-ideas-validation.md + AMENDMENTS.
//   node bench/ideas-eval.mjs
import { readFileSync, writeFileSync } from "node:fs";
import { relative } from "node:path";
import { walk, stripFront, paragraphsOf, noteText } from "./jina-cache.mjs";
import { fisherYates } from "./shuffle.mjs";

const VAULT = "/Users/justus/obsidian_atomized_intermediary/lab/vault";
const CACHE_PATH = process.env.JINA_CACHE || (process.env.HOME + "/.cache/srn-lab/jina-cache.json");
const OUT = "/Users/justus/obsidian_atomized_intermediary/lab/results/ideas-eval.json";
const PREFIX = "Document: ";
const MODEL = "jinaai/jina-embeddings-v5-text-nano-text-matching";

const fold = (s) => s.toLowerCase().normalize("NFC").replace(/ä/g, "ae").replace(/ö/g, "oe").replace(/ü/g, "ue").replace(/ß/g, "ss").replace(/\s+/g, " ").trim();
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const l2 = (v) => { let s = 0; for (const x of v) s += x * x; s = Math.sqrt(s) || 1; return v.map((x) => x / s); };

let cache = JSON.parse(readFileSync(CACHE_PATH, "utf8"));
let pipe = null;
async function embed(texts) {
  const missing = [...new Set(texts.filter((t) => !cache[PREFIX + t]))];
  if (!missing.length) return;
  if (!pipe) {
    const { pipeline, env } = await import("@huggingface/transformers");
    env.allowLocalModels = false;
    pipe = await pipeline("feature-extraction", MODEL, { dtype: "fp32" });
  }
  for (const t of missing) {
    const o = await pipe(PREFIX + t, { pooling: "none" });
    const d = o.dims, data = o.data;
    const seq = d.length === 3 ? d[1] : d[0], dim = d.length === 3 ? d[2] : d[1];
    let v = Array.from(data.subarray((seq - 1) * dim, seq * dim));
    const n = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
    cache[PREFIX + t] = v.map((x) => +(x / n).toFixed(5));
    o.dispose?.();
  }
  writeFileSync(CACHE_PATH, JSON.stringify(cache));
}
const vec = (t) => cache[PREFIX + t] || null;

// deterministic RNG (fixed seeds, pre-registered)
function mulberry32(a) { return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

// ---------- load ----------
const manifest = JSON.parse(readFileSync("/Users/justus/obsidian_atomized_intermediary/lab/corpus-manifest.json", "utf8"));
const INDEXABLE = new Set(manifest.answer_paths.map((p) => "real/" + p));
const notes = [];
for (const abs of walk(VAULT)) {
  const rel = relative(VAULT, abs);
  if (rel.startsWith("real/") && !INDEXABLE.has(rel)) continue;
  const basename = rel.replace(/\.md$/, "").split("/").pop();
  if (/\bMOC\b/i.test(basename) || /(^|\/)Attachments\//.test(rel) || /\.dup$/.test(basename)) continue;
  const body = stripFront(readFileSync(abs, "utf8"));
  notes.push({ rel, basename, body });
}
await embed(notes.map((n) => noteText(n.basename, n.body)));
for (const n of notes) n.v = vec(noteText(n.basename, n.body));
const withV = notes.filter((n) => n.v);
const D = withV[0].v.length;
const mean = new Array(D).fill(0);
for (const n of withV) for (let i = 0; i < D; i++) mean[i] += n.v[i];
for (let i = 0; i < D; i++) mean[i] /= withV.length;
const center = (v) => l2(v.map((x, i) => x - mean[i]));
for (const n of withV) n.c = center(n.v);
const byFold = new Map(withV.map((n) => [fold(n.basename), n]));
console.log(`loaded ${withV.length} notes (dups excluded)`);

// ---------- spherical k-means (k-means++, fixed seed, 5 restarts) ----------
function kmeans(vectors, k, seed) {
  let best = null;
  for (let r = 0; r < 5; r++) {
    const rnd = mulberry32(seed + r * 1000);
    const cents = [vectors[Math.floor(rnd() * vectors.length)]];
    while (cents.length < k) {
      const dists = vectors.map((v) => Math.min(...cents.map((c) => 1 - dot(v, c))));
      const sum = dists.reduce((s, x) => s + x, 0);
      let pick = rnd() * sum;
      let idx = 0;
      for (; idx < dists.length - 1 && pick > dists[idx]; idx++) pick -= dists[idx];
      cents.push(vectors[idx]);
    }
    let assign = new Array(vectors.length).fill(0);
    for (let it = 0; it < 40; it++) {
      let changed = 0;
      for (let i = 0; i < vectors.length; i++) {
        let bi = 0, bs = -2;
        for (let j = 0; j < k; j++) { const s = dot(vectors[i], cents[j]); if (s > bs) { bs = s; bi = j; } }
        if (assign[i] !== bi) { assign[i] = bi; changed++; }
      }
      for (let j = 0; j < k; j++) {
        const members = vectors.filter((_, i) => assign[i] === j);
        if (!members.length) continue;
        const m = new Array(D).fill(0);
        for (const v of members) for (let i = 0; i < D; i++) m[i] += v[i];
        cents[j] = l2(m);
      }
      if (!changed) break;
    }
    const inertia = vectors.reduce((s, v, i) => s + (1 - dot(v, cents[assign[i]])), 0);
    if (!best || inertia < best.inertia) best = { cents, assign, inertia };
  }
  return best;
}

// ---------- C: concept search ----------
const MOC_QUERIES = [
  ["Machine Learning MOC", "Machine Learning"],
  ["Statistik MOC", "Statistik"],
  ["Programmierung MOC", "Programmierung"],
  ["Datenbanken MOC", "Datenbanken"],
  ["Theoretische Informatik MOC", "Theoretische Informatik"],
];
function mocMembers(mocBase) {
  const raw = readFileSync(`${VAULT}/real/${mocBase}.md`, "utf8");
  const out = new Set();
  for (const m of raw.matchAll(/\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]/g)) {
    const t = byFold.get(fold(m[1]));
    if (t && !/\bMOC\b/i.test(t.basename)) out.add(t.rel);
  }
  return [...out];
}
await embed(MOC_QUERIES.map(([, q]) => q).concat(["Physik"]));
const km12 = kmeans(withV.map((n) => n.c), 12, 42);
const prec = (rankedRels, gtSet, n) => {
  const top = rankedRels.slice(0, n);
  return top.length ? +(top.filter((r) => gtSet.has(r)).length / top.length).toFixed(3) : null;
};
const conceptRows = [];
for (const [mocBase, query] of MOC_QUERIES) {
  const gtAll = mocMembers(mocBase);
  const gtSet = new Set(gtAll);
  const qv = center(vec(query));
  // baseline: plain cosine
  const baseRanked = [...withV].sort((a, b) => dot(qv, b.c) - dot(qv, a.c)).map((n) => n.rel);
  // cluster-seeded: top-2 centroids by query cosine, prototype = normalized sum
  const centScores = km12.cents.map((c, j) => [dot(qv, c), j]).sort((a, b) => b[0] - a[0]);
  const proto = l2(km12.cents[centScores[0][1]].map((x, i) => x + km12.cents[centScores[1][1]][i]));
  const clusterRanked = [...withV].sort((a, b) => dot(proto, b.c) - dot(proto, a.c)).map((n) => n.rel);
  // hub-seeded held-out, 5 fixed splits
  const hubPrecs = [];
  for (let s = 1; s <= 5; s++) {
    const rnd = mulberry32(s);
    const shuffled = fisherYates(gtAll, rnd);
    const seed = shuffled.slice(0, Math.floor(gtAll.length / 2));
    const held = new Set(shuffled.slice(Math.floor(gtAll.length / 2)));
    const sv = seed.map((r) => withV.find((n) => n.rel === r).c);
    const m = new Array(D).fill(0);
    for (const v of sv) for (let i = 0; i < D; i++) m[i] += v[i];
    const hproto = l2(m);
    const seedSet = new Set(seed);
    const ranked = withV.filter((n) => !seedSet.has(n.rel)).sort((a, b) => dot(hproto, b.c) - dot(hproto, a.c)).map((n) => n.rel);
    hubPrecs.push(prec(ranked, held, Math.min(10, held.size)));
  }
  conceptRows.push({ query, gtN: gtAll.length,
    baselineP10: prec(baseRanked, gtSet, 10), baselineR25: +(baseRanked.slice(0, 25).filter((r) => gtSet.has(r)).length / gtAll.length).toFixed(3),
    clusterP10: prec(clusterRanked, gtSet, 10), clusterR25: +(clusterRanked.slice(0, 25).filter((r) => gtSet.has(r)).length / gtAll.length).toFixed(3),
    hubHeldP10avg: +(hubPrecs.reduce((s, x) => s + x, 0) / hubPrecs.length).toFixed(3), hubSplits: hubPrecs });
}
// Physik cold-start probe (no MOC exists): folder GT, cluster-seeded + baseline
{
  const gtSet = new Set(withV.filter((n) => n.rel.startsWith("real/Physik/")).map((n) => n.rel));
  const qv = center(vec("Physik"));
  const baseRanked = [...withV].sort((a, b) => dot(qv, b.c) - dot(qv, a.c)).map((n) => n.rel);
  const centScores = km12.cents.map((c, j) => [dot(qv, c), j]).sort((a, b) => b[0] - a[0]);
  const proto = l2(km12.cents[centScores[0][1]].map((x, i) => x + km12.cents[centScores[1][1]][i]));
  const clusterRanked = [...withV].sort((a, b) => dot(proto, b.c) - dot(proto, a.c)).map((n) => n.rel);
  conceptRows.push({ query: "Physik (cold start, folder GT)", gtN: gtSet.size,
    baselineP10: prec(baseRanked, gtSet, 10), clusterP10: prec(clusterRanked, gtSet, 10),
    baselineR25: +(baseRanked.slice(0, 25).filter((r) => gtSet.has(r)).length / gtSet.size).toFixed(3),
    clusterR25: +(clusterRanked.slice(0, 25).filter((r) => gtSet.has(r)).length / gtSet.size).toFixed(3) });
}

// MOC gap-filling receipts (non-gating): top non-member full-prototype hits
const gapReceipts = [];
for (const [mocBase, query] of MOC_QUERIES.slice(0, 2)) {
  const gtAll = mocMembers(mocBase);
  const sv = gtAll.map((r) => withV.find((n) => n.rel === r).c);
  const m = new Array(D).fill(0);
  for (const v of sv) for (let i = 0; i < D; i++) m[i] += v[i];
  const proto = l2(m);
  const gtSet = new Set(gtAll);
  const hits = withV.filter((n) => !gtSet.has(n.rel)).sort((a, b) => dot(proto, b.c) - dot(proto, a.c)).slice(0, 5);
  gapReceipts.push({ moc: mocBase, missing: hits.map((n) => n.rel) });
}

// ---------- D: cross-language twins (RAW cosine space, pre-registered) ----------
const langStops = {
  de: new Set("der die das und oder nicht ein eine mit von zu im ist sind war fuer auf als auch des dem den bei aus nach wird wenn dann man kann sich nur noch aber ueber damit durch beim zum zur".split(" ")),
  en: new Set("the and or not a an with of to in is are was for on as also from by at when then can that this it be its which each has have".split(" ")),
};
function langOf(text) {
  const words = fold(text).split(/[^a-z]+/).filter(Boolean);
  let de = 0, en = 0;
  for (const w of words) { if (langStops.de.has(w)) de++; if (langStops.en.has(w)) en++; }
  if (de >= 2 * Math.max(1, en)) return "de";
  if (en >= 2 * Math.max(1, de)) return "en";
  return "mixed";
}
const mathFrac = (t) => (t.match(/[$\\^_{}=+]/g) || []).length / Math.max(1, t.length);
const paraDf = new Map();
for (const n of notes) for (const f of new Set(paragraphsOf(n.body).map(fold))) paraDf.set(f, (paraDf.get(f) ?? 0) + 1);
const twinParas = [];
for (const n of notes) {
  for (const p of paragraphsOf(n.body)) {
    if ((paraDf.get(fold(p)) ?? 0) >= 3 || mathFrac(p) > 0.15) continue;
    const lang = langOf(p);
    if (lang === "mixed") continue;
    const v = vec(p);
    if (v) twinParas.push({ rel: n.rel, lang, text: p, v }); // RAW vector
  }
}
const de = twinParas.filter((p) => p.lang === "de"), en = twinParas.filter((p) => p.lang === "en");
const twins = [];
for (const a of de) for (const b of en) {
  if (a.rel === b.rel) continue;
  const s = dot(a.v, b.v);
  if (s >= 0.85) twins.push({ sim: +s.toFixed(3), aRel: a.rel, bRel: b.rel, aText: a.text.slice(0, 220), bText: b.text.slice(0, 220) });
}
twins.sort((a, b) => b.sim - a.sim);
const cpdRank = twins.findIndex((t) =>
  (t.aRel.includes("Abkuehlung und Akzeptanz") && t.bRel.includes("Acceptance Rule Notes")) ||
  (t.bRel.includes("Abkuehlung und Akzeptanz") && t.aRel.includes("Acceptance Rule Notes"))) + 1;

// ---------- E: cartography ----------
const cartoNotes = withV.filter((n) => n.rel.startsWith("real/") && n.rel.includes("/") &&
  !/^real\/(Daily)\//.test(n.rel) && n.rel.split("/").length > 2);
const folders = cartoNotes.map((n) => n.rel.split("/")[1]);
const folderSet = [...new Set(folders)];
function purityNmi(assign, labels, k) {
  const N = labels.length;
  const clusters = new Map();
  for (let i = 0; i < N; i++) { if (!clusters.has(assign[i])) clusters.set(assign[i], []); clusters.get(assign[i]).push(labels[i]); }
  let pur = 0, mi = 0;
  const labelCount = new Map();
  for (const l of labels) labelCount.set(l, (labelCount.get(l) ?? 0) + 1);
  let hC = 0, hL = 0;
  for (const [, members] of clusters) {
    const cnt = new Map();
    for (const l of members) cnt.set(l, (cnt.get(l) ?? 0) + 1);
    pur += Math.max(...cnt.values());
    const pc = members.length / N;
    hC -= pc * Math.log(pc);
    for (const [l, c] of cnt) {
      const pxy = c / N;
      mi += pxy * Math.log(pxy / (pc * (labelCount.get(l) / N)));
    }
  }
  for (const [, c] of labelCount) { const p = c / N; hL -= p * Math.log(p); }
  return { purity: +(pur / N).toFixed(3), nmi: +(mi / Math.sqrt(hC * hL)).toFixed(3) };
}
const cartoResults = {};
for (const k of [8, 12, 16]) {
  const kmr = kmeans(cartoNotes.map((n) => n.c), k, 42);
  cartoResults["k" + k] = purityNmi(kmr.assign, folders, k);
  if (k === 12) cartoResults.km12 = kmr;
}
// calibration baselines
{
  const rnd = mulberry32(7);
  const perms = [];
  for (let t = 0; t < 20; t++) {
    const shuffled = fisherYates(folders, rnd);
    perms.push(purityNmi(cartoResults.km12.assign, shuffled, 12).purity);
  }
  cartoResults.permutationPurity = +(perms.reduce((s, x) => s + x, 0) / perms.length).toFixed(3);
  const langAssign = cartoNotes.map((n) => langOf(n.body) === "de" ? 0 : 1);
  cartoResults.languageSplit = purityNmi(langAssign, folders, 2);
}
// PCA 2-D (power iteration on covariance of centered note vecs)
function pca2(vectors) {
  const N = vectors.length;
  const mv = new Array(D).fill(0);
  for (const v of vectors) for (let i = 0; i < D; i++) mv[i] += v[i] / N;
  const X = vectors.map((v) => v.map((x, i) => x - mv[i]));
  const matvec = (w) => {
    const out = new Array(D).fill(0);
    for (const x of X) { const s = dot(x, w); for (let i = 0; i < D; i++) out[i] += s * x[i]; }
    return out;
  };
  const rnd = mulberry32(3);
  const comps = [];
  for (let c = 0; c < 2; c++) {
    let w = l2(new Array(D).fill(0).map(() => rnd() - 0.5));
    for (let it = 0; it < 60; it++) {
      let nw = matvec(w);
      for (const p of comps) { const s = dot(nw, p); nw = nw.map((x, i) => x - s * p[i]); }
      w = l2(nw);
    }
    comps.push(w);
  }
  return X.map((x) => [dot(x, comps[0]), dot(x, comps[1])]);
}
const xy = pca2(cartoNotes.map((n) => n.c));
// cluster keyphrase labels: top df-idf title words per cluster
const STOPW = new Set("the a of and und der die das for in im to von mit".split(" "));
function clusterLabel(j) {
  const cnt = new Map();
  cartoNotes.forEach((n, i) => {
    if (cartoResults.km12.assign[i] !== j) return;
    for (const w of new Set(fold(n.basename).split(/[^a-z0-9]+/).filter((w) => w.length >= 4 && !STOPW.has(w))))
      cnt.set(w, (cnt.get(w) ?? 0) + 1);
  });
  return [...cnt.entries()].sort((a, b) => b[1] - a[1]).slice(0, 3).map(([w]) => w).join(" · ");
}
// SVG map
{
  const xs = xy.map((p) => p[0]), ys = xy.map((p) => p[1]);
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)], [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  const sx = (x) => 60 + (x - x0) / (x1 - x0) * 780, sy = (y) => 70 + (y - y0) / (y1 - y0) * 560;
  const colors = ["#e05252", "#e0a552", "#c8d052", "#6fd052", "#52d0a0", "#52c8d0", "#5288d0", "#7a52d0", "#c052d0", "#d05285", "#8a8f9e", "#d0d0d0"];
  const parts = [`<svg viewBox="0 0 900 700" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system,Helvetica,sans-serif"><rect width="900" height="700" fill="#0f1117"/>`,
    `<text x="30" y="34" fill="#e8eaf0" font-size="17" font-weight="700">Vault cartography — ${cartoNotes.length} notes, jina-v5-nano, k=12 spherical k-means</text>`,
    `<text x="30" y="54" fill="#8a90a2" font-size="11">purity ${cartoResults.k12.purity} vs folders (permutation baseline ${cartoResults.permutationPurity}, language-split ${cartoResults.languageSplit.purity}) · NMI ${cartoResults.k12.nmi}</text>`];
  cartoNotes.forEach((n, i) => {
    parts.push(`<circle cx="${sx(xy[i][0]).toFixed(1)}" cy="${sy(xy[i][1]).toFixed(1)}" r="3.2" fill="${colors[cartoResults.km12.assign[i] % 12]}" opacity="0.75"><title>${n.basename.replace(/&/g, "&amp;").replace(/</g, "&lt;")}</title></circle>`);
  });
  for (let j = 0; j < 12; j++) {
    const members = xy.filter((_, i) => cartoResults.km12.assign[i] === j);
    if (members.length < 4) continue;
    const cx = members.reduce((s, p) => s + sx(p[0]), 0) / members.length;
    const cy = members.reduce((s, p) => s + sy(p[1]), 0) / members.length;
    parts.push(`<text x="${cx.toFixed(0)}" y="${cy.toFixed(0)}" fill="#ffffff" font-size="11.5" font-weight="600" text-anchor="middle" stroke="#0f1117" stroke-width="3" paint-order="stroke">${clusterLabel(j)}</text>`);
  }
  parts.push("</svg>");
  writeFileSync("/Users/justus/obsidian_atomized_intermediary/lab/results/vault-map.svg", parts.join("\n"));
}

const out = {
  C: { rows: conceptRows, gapReceipts, k: 12 },
  D: { deParas: de.length, enParas: en.length, mined: twins.length, cpdRank: cpdRank || null, top: twins.slice(0, 12) },
  E: { k8: cartoResults.k8, k12: cartoResults.k12, k16: cartoResults.k16,
    permutationPurity: cartoResults.permutationPurity, languageSplit: cartoResults.languageSplit, notes: cartoNotes.length },
};
console.log(JSON.stringify(out, (k, v) => k === "km12" ? undefined : v, 1));
writeFileSync(OUT, JSON.stringify(out, (k, v) => k === "km12" ? undefined : v, 1));
console.log("wrote", OUT, "and lab/results/vault-map.svg");
