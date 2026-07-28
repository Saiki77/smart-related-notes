// Resurfacing-engine + dupe-alarm validation harness (jina-v5-nano). v2 after
// 3-critic panel — see lab/EXPERIMENT-ideas-validation.md AMENDMENTS.
//
// A (resurfacing): natural GT = paragraphs containing wikilinks; three query
// variants per (paragraph, target):
//   typed   — links replaced by display text (link-autocomplete product)
//   context — link display text removed (magic product)
//   before  — only the text preceding the link (realism slice)
// B (dupe alarm): 8 paraphrased .dup notes (GT paraphrase_dups) must retrieve
// their original top-1 at a high alarm threshold; false alarms measured on all
// normal paragraphs with G6 copy-rule and cross-language exemptions.
//
//   node bench/resurf-eval.mjs
// Env: JINA_CACHE, LAB_VAULT, RS_G1 (default 1), RS_MINW (25), RS_OUT.
// Missing query embeddings are computed inline (lazy model load) and cached.
import { readFileSync, writeFileSync } from "node:fs";
import { relative } from "node:path";
import { walk, stripFront, paragraphsOf, noteText } from "./jina-cache.mjs";

const VAULT = process.env.LAB_VAULT || "/Users/justus/obsidian_atomized_intermediary/lab/vault";
const CACHE_PATH = process.env.JINA_CACHE || (process.env.HOME + "/.cache/srn-lab/jina-cache.json");
const OUT = process.env.RS_OUT || "/Users/justus/obsidian_atomized_intermediary/lab/results/resurf-eval.json";
const PREFIX = "Document: ";
const G1 = process.env.RS_G1 !== "0";
const MINW = Number(process.env.RS_MINW ?? 25);
const MODEL = "jinaai/jina-embeddings-v5-text-nano-text-matching";

const fold = (s) => s.toLowerCase().normalize("NFC")
  .replace(/ä/g, "ae").replace(/ö/g, "oe").replace(/ü/g, "ue").replace(/ß/g, "ss")
  .replace(/\s+/g, " ").trim();
const SUFFIXES = ["en", "er", "es", "em", "e", "n", "s"];
const stem = (w) => { for (const suf of SUFFIXES) if (w.length - suf.length >= 4 && w.endsWith(suf)) return w.slice(0, w.length - suf.length); return w; };
const stemWords = (t) => fold(t).split(/[^a-z0-9]+/).filter(Boolean).map(stem);
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const l2 = (v) => { let s = 0; for (const x of v) s += x * x; s = Math.sqrt(s) || 1; return v.map((x) => x / s); };
const wordCount = (t) => t.split(/\s+/).filter(Boolean).length;

const manifest = JSON.parse(readFileSync("/Users/justus/obsidian_atomized_intermediary/lab/corpus-manifest.json", "utf8"));
const INDEXABLE = new Set(manifest.answer_paths.map((p) => "real/" + p));
const AUTHENTIC = new Set(manifest.paths.map((p) => "real/" + p));
const gt = JSON.parse(readFileSync("/Users/justus/obsidian_atomized_intermediary/lab/ground-truth.json", "utf8"));
const PARA_DUPS = new Set((gt.paraphrase_dups || []).map((e) => e.dup_file));

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
  console.log(`embedding ${missing.length} query texts ...`);
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

// ---------- load notes ----------
const HUB_RE = /\bMOC\b|Uebersicht|Zettelkasten Index|Vault Insights|^Untitled$/i;
function parseAliases(raw) {
  const m = raw.match(/^---\n([\s\S]*?)\n---/);
  if (!m) return [];
  const out = [];
  const am = m[1].match(/^aliases:\s*\n((?:\s*-\s*.+\n?)+)/m);
  if (am) for (const line of am[1].split("\n")) { const lm = line.match(/-\s*(.+)/); if (lm) out.push(lm[1].trim().replace(/^["']+|["']+$/g, "")); }
  return out;
}
const notes = [];
for (const abs of walk(VAULT)) {
  const rel = relative(VAULT, abs);
  if (rel.startsWith("real/") && !INDEXABLE.has(rel)) continue;
  const basename = rel.replace(/\.md$/, "").split("/").pop();
  if (/\bMOC\b/i.test(basename) || /(^|\/)Attachments\//.test(rel)) continue;
  const raw = readFileSync(abs, "utf8");
  const body = stripFront(raw);
  notes.push({ rel, basename, body, aliases: parseAliases(raw),
    isDup: /\.dup$/.test(basename), isParaDup: PARA_DUPS.has(rel),
    isHub: HUB_RE.test(basename), authentic: AUTHENTIC.has(rel) });
}
const byFold = new Map(notes.map((n) => [fold(n.basename), n]));

// paragraph df (boilerplate)
const paraDf = new Map();
for (const n of notes) {
  for (const f of new Set(paragraphsOf(n.body).map(fold))) paraDf.set(f, (paraDf.get(f) ?? 0) + 1);
}
const isBoiler = (t) => (paraDf.get(fold(t)) ?? 0) >= 3;

// index paragraphs (A/B share it; ALL .dup notes excluded per amendment 2)
for (const n of notes) {
  n.paras = n.isDup ? [] : paragraphsOf(n.body).filter((p) => !G1 || !isBoiler(p));
}
await embed(notes.flatMap((n) => n.paras));
const indexNotes = notes.filter((n) => !n.isDup && n.paras.length);
// centering: raw paragraph centroid, subtract, renormalize (amendment 5)
{
  const all = indexNotes.flatMap((n) => n.paras.map(vec).filter(Boolean));
  const d = all[0].length, mean = new Array(d).fill(0);
  for (const v of all) for (let i = 0; i < d; i++) mean[i] += v[i];
  for (let i = 0; i < d; i++) mean[i] /= all.length;
  globalThis.__pMean = mean;
}
const centerP = (v) => l2(v.map((x, i) => x - globalThis.__pMean[i]));
for (const n of indexNotes) n.paraVecsC = n.paras.map((p) => vec(p)).filter(Boolean).map(centerP);
const unreachable = notes.filter((n) => !n.isDup && !n.isHub && !n.paras.length).length;
console.log(`index: ${indexNotes.length} notes, ${indexNotes.reduce((s, n) => s + n.paraVecsC.length, 0)} paragraphs (G1=${G1}); unreachable notes (no >=25w paragraph): ${unreachable}`);

function retrieve(qv, excludeRel) {
  const scored = [];
  for (const n of indexNotes) {
    if (n.rel === excludeRel) continue;
    let best = -1, bi = -1;
    for (let i = 0; i < n.paraVecsC.length; i++) { const s = dot(qv, n.paraVecsC[i]); if (s > best) { best = s; bi = i; } }
    if (bi >= 0) scored.push([best, n, n.paras[bi]]);
  }
  scored.sort((a, b) => b[0] - a[0]);
  return scored;
}

// ---------- A: stripped-link prefetch ----------
const LINK_RE = /\[\[([^\]|#]+)(?:\|([^\]]+))?(?:#[^\]]*)?\]\]/g;
function rawBlocks(body) {
  return body.replace(/```[\s\S]*?```/g, "\n").replace(/%%[\s\S]*?%%/g, "\n").split(/\n\s*\n/);
}
const cleanBlock = (b) => {
  const lines = b.split("\n").filter((l) => !/^\s*#{1,6}\s/.test(l) && !/^\s*\|/.test(l))
    .map((l) => l.replace(/^\s*>+\s*/, "").replace(/\[![A-Za-z]+\][+-]?\s*/g, ""));
  return lines.join(" ")
    .replace(/\[\[([^\]|]+)\|([^\]]+)\]\]/g, "$2").replace(/\[\[([^\]]+)\]\]/g, "$1")
    .replace(/^[\s>*+-]+/, "").replace(/[*_~`]/g, "").replace(/\s+/g, " ").trim();
};
// contiguous stemmed-word-sequence containment
function containsSeq(haystack, needle) {
  if (!needle.length) return false;
  for (let i = 0; i + needle.length <= haystack.length; i++)
    if (needle.every((w, j) => haystack[i + j] === w)) return true;
  return false;
}
function titlePresent(queryText, target) {
  const hay = stemWords(queryText);
  const names = [target.basename, ...target.aliases];
  return names.some((nm) => containsSeq(hay, stemWords(nm)));
}

const cases = []; // per (paragraph, target)
const paraCases = new Map(); // paragraph id -> {queries, targets:[rels]}
for (const n of notes) {
  if (n.isDup || n.isHub) continue;
  const blocks = rawBlocks(n.body);
  for (let b = 0; b < blocks.length; b++) {
    const raw = blocks[b];
    const clean = cleanBlock(raw);
    if (wordCount(clean) < MINW) continue;
    if (G1 && isBoiler(clean)) continue;
    const targets = [];
    LINK_RE.lastIndex = 0;
    let m;
    while ((m = LINK_RE.exec(raw)) !== null) {
      const t = byFold.get(fold(m[1]));
      if (t && t.rel !== n.rel && !t.isDup && !t.isHub && t.paras.length) targets.push({ note: t, matchIndex: m.index });
    }
    if (!targets.length) continue;
    const context = cleanBlock(raw.replace(LINK_RE, " "));
    const pid = `${n.rel}#${b}`;
    paraCases.set(pid, { note: n, typed: clean, targets: [...new Set(targets.map((t) => t.note.rel))] });
    for (const t of targets) {
      const before = cleanBlock(raw.slice(0, t.matchIndex));
      cases.push({ pid, note: n, target: t.note,
        typed: clean,
        context: wordCount(context) >= 15 ? context : null,
        before: wordCount(before) >= 15 ? before : null,
        titlePresent: titlePresent(clean, t.note),
        authentic: n.authentic });
    }
  }
}
console.log(`A cases: ${cases.length} (paragraphs: ${paraCases.size}; authentic: ${cases.filter(c => c.authentic).length}; title-absent(typed): ${cases.filter(c => !c.titlePresent).length})`);
await embed(cases.flatMap((c) => [c.typed, c.context, c.before].filter(Boolean)));

const rankCache = new Map(); // text||exclude -> ranked
function rankedFor(text, excludeRel) {
  const k = excludeRel + "||" + text;
  if (!rankCache.has(k)) {
    const v = vec(text);
    rankCache.set(k, v ? retrieve(centerP(v), excludeRel) : null);
  }
  return rankCache.get(k);
}
function evalVariant(variant) {
  const rows = [];
  for (const c of cases) {
    const text = c[variant];
    if (!text) continue;
    const ranked = rankedFor(text, c.note.rel);
    if (!ranked) continue;
    const idx = ranked.findIndex(([, n]) => n.rel === c.target.rel);
    rows.push({ pid: c.pid, rank: idx + 1, top1Score: ranked[0][0], top1: ranked[0][1].rel,
      titlePresent: c.titlePresent, authentic: c.authentic, note: c.note.rel, target: c.target.rel });
  }
  const agg = (rs) => rs.length ? {
    n: rs.length,
    top1: +(rs.filter((r) => r.rank === 1).length / rs.length).toFixed(3),
    top3: +(rs.filter((r) => r.rank >= 1 && r.rank <= 3).length / rs.length).toFixed(3),
    mrr: +(rs.reduce((s, r) => s + (r.rank >= 1 ? 1 / r.rank : 0), 0) / rs.length).toFixed(3),
  } : { n: 0 };
  // macro per distinct target (title-absent slice concern)
  const macro = (rs) => {
    const byT = new Map();
    for (const r of rs) { if (!byT.has(r.target)) byT.set(r.target, []); byT.get(r.target).push(r); }
    const per = [...byT.values()].map((g) => g.filter((r) => r.rank >= 1 && r.rank <= 3).length / g.length);
    return per.length ? +(per.reduce((s, x) => s + x, 0) / per.length).toFixed(3) : null;
  };
  const rowsTA = rows.filter((r) => !r.titlePresent);
  // per-paragraph any-target-in-top-3 (typed only meaningful, computed generally)
  const byPid = new Map();
  for (const r of rows) { if (!byPid.has(r.pid)) byPid.set(r.pid, []); byPid.get(r.pid).push(r); }
  const anyTop3 = [...byPid.values()].filter((g) => g.some((r) => r.rank >= 1 && r.rank <= 3)).length / Math.max(1, byPid.size);
  return { rows, all: agg(rows), titleAbsent: { ...agg(rowsTA), macroTop3: macro(rowsTA) },
    titlePresentAgg: agg(rows.filter((r) => r.titlePresent)),
    authentic: agg(rows.filter((r) => r.authentic)),
    anyTargetTop3PerParagraph: +anyTop3.toFixed(3) };
}

// ---------- silence set ----------
const boilerQueries = [];
{
  const seen = new Set();
  for (const n of notes) {
    if (n.isDup) continue;
    for (const p of paragraphsOf(n.body)) {
      if (!isBoiler(p) || seen.has(fold(p))) continue;
      seen.add(fold(p));
      boilerQueries.push({ text: p, from: n.rel });
    }
  }
}
await embed(boilerQueries.map((q) => q.text));
const boilerTops = boilerQueries.map((q) => rankedFor(q.text, q.from)).filter(Boolean).map((r) => r[0][0]);

// ---------- B: dupe alarm (paraphrase tier) ----------
const langStops = {
  de: new Set("der die das und oder nicht ein eine mit von zu im ist sind war fuer auf als auch des dem den bei aus nach wird wenn dann man kann".split(" ")),
  en: new Set("the and or not a an with of to in is are was for on as also from by at when then can that this it be".split(" ")),
};
function langOf(text) {
  const words = fold(text).split(/[^a-z]+/).filter(Boolean);
  let de = 0, en = 0;
  for (const w of words) { if (langStops.de.has(w)) de++; if (langStops.en.has(w)) en++; }
  if (de >= 2 * Math.max(1, en)) return "de";
  if (en >= 2 * Math.max(1, de)) return "en";
  return "mixed";
}
const foldedLines = (body) => new Set(body.split("\n").map(fold).filter((l) => l.length >= 30));
const copyRule = (a, b) => { // G6: identical folded line >=30 chars in both notes
  const la = foldedLines(a.body);
  for (const l of foldedLines(b.body)) if (la.has(l)) return true;
  return false;
};
const dupRows = [];
for (const n of notes.filter((n) => n.isParaDup)) {
  const orig = byFold.get(fold(n.basename.replace(/\.dup$/, "")));
  if (!orig) continue;
  for (const p of paragraphsOf(n.body)) {
    if (G1 && isBoiler(p)) continue;
    await embed([p]);
    const ranked = rankedFor(p, n.rel);
    if (!ranked) continue;
    dupRows.push({ dup: n.rel, orig: orig.rel, top: ranked[0][1].rel, topScore: +ranked[0][0].toFixed(3), hit: ranked[0][1].rel === orig.rel });
  }
}
// normal-paragraph alarm distribution (dup queries and dup index already excluded)
const normalTops = [];
for (const n of indexNotes) {
  for (const p of n.paras) {
    const ranked = rankedFor(p, n.rel);
    if (!ranked) continue;
    const [score, tn, tp] = ranked[0];
    normalTops.push({ score: +score.toFixed(3), from: n.rel, to: tn.rel,
      copy: score >= 0.75 ? copyRule(n, tn) : false,
      crossLang: langOf(p) !== "mixed" && langOf(tp) !== "mixed" && langOf(p) !== langOf(tp),
      text: p.slice(0, 100), toText: tp.slice(0, 100) });
  }
}

// ---------- report ----------
const typed = evalVariant("typed");
const context = evalVariant("context");
const before = evalVariant("before");
const q = (arr, p) => arr.length ? [...arr].sort((a, b) => a - b)[Math.min(arr.length - 1, Math.floor(p * arr.length))] : null;
const dupScores = dupRows.map((r) => r.topScore);
const alarmEligible = normalTops.filter((r) => !r.copy && !r.crossLang);
const normScores = alarmEligible.map((r) => r.score);
const thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9];
const curve = thresholds.map((t) => ({
  t,
  boilerFire: boilerTops.length ? +(boilerTops.filter((s) => s >= t).length / boilerTops.length).toFixed(3) : null,
  normalFire: +(normScores.filter((s) => s >= t).length / normScores.length).toFixed(4),
  typedTop1Covered: +(typed.rows.filter((r) => r.rank === 1 && r.top1Score >= t).length / Math.max(1, typed.rows.length)).toFixed(3),
  dupRecall: dupScores.length ? +(dupRows.filter((r) => r.hit && r.topScore >= t).length / dupRows.length).toFixed(3) : null,
}));
const summary = {
  model: "jina-v5-nano", G1, MINW,
  A: {
    cases: cases.length, paragraphs: paraCases.size, unreachableNotes: unreachable,
    typed: { ...typed, rows: undefined }, context: { ...context, rows: undefined }, before: { ...before, rows: undefined },
  },
  silence: { boilerQueries: boilerQueries.length, boilerTopP50: q(boilerTops, 0.5), boilerTopP90: q(boilerTops, 0.9) },
  B: {
    paraDupParagraphs: dupRows.length,
    top1IsOriginal: +(dupRows.filter((r) => r.hit).length / Math.max(1, dupRows.length)).toFixed(3),
    dupScoreP10: q(dupScores, 0.1), dupScoreP50: q(dupScores, 0.5), dupScoreMin: Math.min(...dupScores),
    normalEligible: alarmEligible.length,
    normalTopP99: q(normScores, 0.99), normalTopP999: q(normScores, 0.999),
    exemptCopy: normalTops.filter((r) => r.copy).length, exemptCrossLang: normalTops.filter((r) => r.crossLang && !r.copy).length,
  },
  curve,
};
console.log(JSON.stringify(summary, null, 1));
writeFileSync(OUT, JSON.stringify({ summary,
  typedRows: typed.rows, contextRows: context.rows, beforeRows: before.rows, dupRows,
  normalHigh: alarmEligible.filter((r) => r.score >= 0.75).sort((a, b) => b.score - a.score).slice(0, 30),
  authenticItemized: typed.rows.filter((r) => r.authentic),
}, null, 1));
console.log("wrote", OUT);
