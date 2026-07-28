// Core related-notes ranking suite for the lab vault.
//
// Measures the plugin's ranking pipeline (stage-1 cosine shortlist over
// centered note means, stage-2 bidirectional MaxSim over sentence chunks with
// title weighting) against every ground-truth section that has never had a
// harness:
//   link_recall@10   — do a note's actual wikilink targets rank in its top-10?
//   bridges          — related-but-unlinked discoveries (a should surface c)
//   analogies        — cross-domain structural analogies (x should surface y)
//   style_confounds  — same writing register, unrelated topics: must NOT rank top-10
//   near_dupes       — must rank mutual top-1
//   hub_traps        — MOCs and templated dailies must not dominate top-10 lists
//
//   node bench/rn-eval.mjs
import { pipeline, env } from "@huggingface/transformers";
import { readFileSync, writeFileSync, readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";

env.allowLocalModels = false;

const VAULT = process.env.LAB_VAULT || "/Users/justus/obsidian_atomized_intermediary/lab/vault";
const GT_PATH = process.env.LAB_GT || "/Users/justus/obsidian_atomized_intermediary/lab/ground-truth.json";
const OUT = process.env.LAB_OUT || "/Users/justus/obsidian_atomized_intermediary/lab/results/rn-eval.json";
const MODEL = process.env.LAB_MODEL || "Xenova/paraphrase-multilingual-MiniLM-L12-v2";
const SHORTLIST = Number(process.env.RN_SHORTLIST ?? 50);
const TITLE_W = Number(process.env.RN_TITLEW ?? 2);
const BIMAX = process.env.RN_BIMAX !== "0";

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
function parseFront(raw) {
  const m = raw.match(/^---\n([\s\S]*?)\n---\n?/);
  const aliases = [];
  if (m) {
    const am = m[1].match(/aliases:\s*\[([^\]]*)\]/);
    if (am) for (const a of am[1].split(",")) { const t = a.trim().replace(/^["']|["']$/g, ""); if (t) aliases.push(t); }
  }
  return { aliases, body: m ? raw.slice(m[0].length) : raw };
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

// ---------- load vault ----------
const gt = JSON.parse(readFileSync(GT_PATH, "utf8"));
const manifest = JSON.parse(readFileSync("/Users/justus/obsidian_atomized_intermediary/lab/corpus-manifest.json", "utf8"));
const INDEXABLE_REAL = new Set(manifest.answer_paths.map((p) => "real/" + p));
const files = walk(VAULT).filter((abs) => {
  const rel = relative(VAULT, abs);
  return (!rel.startsWith("real/") || INDEXABLE_REAL.has(rel)) && !/(^|\/)Attachments\//.test(rel);
});
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
const notes = [];
for (const abs of files) {
  const rel = relative(VAULT, abs);
  const basename = rel.replace(/\.md$/, "").split("/").pop();
  const raw = readFileSync(abs, "utf8");
  const { aliases, body } = parseFront(raw);
  const title = aliases.length ? `${basename} (${aliases.join(", ")})` : basename;
  notes.push({ rel, basename, links: wikilinks(body), sents: [title, ...sentencesOf(body).slice(0, 60)] });
}
console.log(`indexed ${notes.length} notes; embedding with ${MODEL} ...`);

const extractor = await pipeline("feature-extraction", MODEL, { dtype: "fp32" });
const cachePath = process.env.AQ_CACHE || "/private/tmp/claude-501/-Users-justus-obsidian-atomized-intermediary/fdd9de71-217a-43e1-9ca7-fb239cfd5cb4/scratchpad/aq-embed-cache.json";
let cache = {};
try { cache = JSON.parse(readFileSync(cachePath, "utf8")); if (cache.__model !== MODEL) cache = {}; } catch { }
const allTexts = [...new Set(notes.flatMap((n) => n.sents))];
const missing = allTexts.filter((t) => !cache[t]);
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
for (const n of notes) {
  n.chunks = n.sents.map((s) => cache[s]).filter(Boolean).map(l2);
  const d0 = n.chunks[0]?.length;
  if (!d0) { n.mean = null; continue; }
  const m = new Array(d0).fill(0);
  for (const v of n.chunks) for (let i = 0; i < d0; i++) m[i] += v[i];
  n.mean = l2(m);
}
const pool = notes.filter((n) => n.mean);
const d = pool[0].mean.length, c = new Array(d).fill(0);
for (const n of pool) for (let i = 0; i < d; i++) c[i] += n.mean[i];
const cN = l2(c);
const center = (v) => { const p = dot(v, cN); return l2(v.map((x, i) => x - p * cN[i])); };
for (const n of pool) { n.cmean = center(n.mean); n.cchunks = n.chunks.map(center); }

// ---------- ranking: stage-1 cosine, stage-2 biMax ----------
function biMax(a, b) {
  let sab = 0, wab = 0;
  for (let i = 0; i < a.cchunks.length; i++) {
    let best = -1;
    for (const cb of b.cchunks) { const s = dot(a.cchunks[i], cb); if (s > best) best = s; }
    const w = i === 0 ? TITLE_W : 1;
    sab += w * best; wab += w;
  }
  let sba = 0, wba = 0;
  for (let j = 0; j < b.cchunks.length; j++) {
    let best = -1;
    for (const ca of a.cchunks) { const s = dot(b.cchunks[j], ca); if (s > best) best = s; }
    const w = j === 0 ? TITLE_W : 1;
    sba += w * best; wba += w;
  }
  return (sab / wab + sba / wba) / 2;
}
const rankCache = new Map();
function ranking(n) {
  if (rankCache.has(n)) return rankCache.get(n);
  const s1 = pool.filter((m) => m !== n).map((m) => [m, dot(n.cmean, m.cmean)])
    .sort((x, y) => y[1] - x[1]);
  let out;
  if (BIMAX) {
    const short = s1.slice(0, SHORTLIST);
    out = short.map(([m]) => [m, biMax(n, m)]).sort((x, y) => y[1] - x[1])
      .concat(s1.slice(SHORTLIST));
  } else out = s1;
  const r = out.map(([m, s]) => ({ m, s }));
  rankCache.set(n, r);
  return r;
}
const byRel = new Map(pool.map((n) => [n.rel, n]));
const byBase = new Map(pool.map((n) => [fold(n.basename), n]));
const rankOf = (n, target) => {
  const r = ranking(n).findIndex((e) => e.m === target);
  return r === -1 ? null : r + 1;
};

// ---------- link recall@10 ----------
let hit = 0, total = 0;
const sampled = pool.filter((n) => n.links.size >= 1 && n.links.size <= 30);
for (const n of sampled) {
  const top10 = new Set(ranking(n).slice(0, 10).map((e) => fold(e.m.basename)));
  for (const l of n.links) {
    if (!byBase.has(l)) continue;
    total++;
    if (top10.has(l)) hit++;
  }
}
const linkRecall = total ? hit / total : null;

// ---------- GT sections ----------
const get = (f) => byRel.get(f);
const bridgeRows = (gt.bridges || []).map((br) => {
  const a = get(br.a_file), b0 = get(br.b_file), cc = get(br.c_file);
  if (!a || !cc) return { id: br.id, missing: true };
  return { id: br.id, a_to_c: rankOf(a, cc), c_to_a: rankOf(cc, a), a_to_b: b0 ? rankOf(a, b0) : null };
});
const analogyRows = (gt.analogies || []).map((an) => {
  const x = get(an.x_file), y = get(an.y_file);
  if (!x || !y) return { id: an.id, missing: true };
  return { id: an.id, x_to_y: rankOf(x, y), y_to_x: rankOf(y, x), relation: an.relation };
});
const confoundRows = (gt.style_confounds || []).map((sc) => {
  const x = get(sc.x_file), y = get(sc.y_file);
  if (!x || !y) return { id: sc.id, missing: true };
  const r = rankOf(x, y);
  return { id: sc.id, mutualRank: r, leaksTop10: r !== null && r <= 10 };
});
const dupeRows = (gt.near_dupes || []).map((nd) => {
  const x = get(nd.x_file), y = get(nd.y_file);
  if (!x || !y) return { id: nd.id, missing: true };
  return { id: nd.id, x_to_y: rankOf(x, y), y_to_x: rankOf(y, x) };
});
// hub traps: MOC appearance rate across all top-10s; daily-daily crowding
const mocs = (gt.hub_traps?.mocs || []).map(get).filter(Boolean);
let mocAppearances = 0, listCount = 0;
for (const n of pool.slice(0, 200)) {
  const top10 = ranking(n).slice(0, 10).map((e) => e.m);
  listCount++;
  if (top10.some((m) => mocs.includes(m))) mocAppearances++;
}
const dailies = pool.filter((n) => /^\d{4}-\d{2}-\d{2}$/.test(n.basename));
let dailyPairsTop10 = 0;
for (const n of dailies) {
  const top10 = ranking(n).slice(0, 10).map((e) => e.m);
  dailyPairsTop10 += top10.filter((m) => dailies.includes(m)).length;
}

const summary = {
  model: MODEL, params: { SHORTLIST, TITLE_W, BIMAX },
  notes: pool.length,
  link_recall_at_10: linkRecall !== null ? +linkRecall.toFixed(3) : null,
  link_pairs_evaluated: total,
  bridges: bridgeRows,
  analogies: analogyRows,
  style_confound_leaks: confoundRows.filter((r) => r.leaksTop10).map((r) => r.id),
  style_confounds: confoundRows,
  near_dupes: dupeRows,
  moc_in_top10_rate: +(mocAppearances / (listCount || 1)).toFixed(3),
  daily_daily_top10_avg: +(dailyPairsTop10 / (dailies.length || 1)).toFixed(2),
};
console.log("\n==== SUMMARY ====");
console.log(JSON.stringify(summary, null, 2));
writeFileSync(OUT, JSON.stringify(summary, null, 2));
console.log("wrote", OUT);
