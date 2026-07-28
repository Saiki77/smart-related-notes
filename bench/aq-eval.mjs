// Answered-Questions validation harness for the lab vault.
//
// Measures the "Sparks / Answered Questions" mechanism on planted ground truth:
//   - can the embedding rank the TRUE answer above distractors for each question?
//   - does the paraphrase trap (a restatement of the question in another note)
//     outrank the true answer in raw cosine (the problem), and does the
//     question-veto + similarity-ceiling gate remove it (the fix)?
//   - do fiction-dialogue questions leak a spark (they must not)?
//
// Runs from related-notes/ so it resolves @huggingface/transformers. Reads the
// lab vault (markdown on disk, NOT the iCloud vault). Model = plugin default.
//
//   node bench/aq-eval.mjs
//
import { pipeline, env } from "@huggingface/transformers";
import { readFileSync, writeFileSync, readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";

env.allowLocalModels = false;

const VAULT = process.env.LAB_VAULT || "/Users/justus/obsidian_atomized_intermediary/lab/vault";
const GT_PATH = process.env.LAB_GT || "/Users/justus/obsidian_atomized_intermediary/lab/ground-truth.json";
const OUT = process.env.LAB_OUT || "/Users/justus/obsidian_atomized_intermediary/lab/results/aq-eval.json";
const MODEL = process.env.LAB_MODEL || "Xenova/paraphrase-multilingual-MiniLM-L12-v2";

// Gates (the design's proposed defaults; tweak to see the effect).
const FLOOR = Number(process.env.AQ_FLOOR ?? 0.15);   // centered-cosine floor
const CEIL = Number(process.env.AQ_CEIL ?? 0.90);     // paraphrase ceiling
const LEN_RATIO = Number(process.env.AQ_LEN ?? 1.3);  // answer must be >= this * question length
const EXCLUDE_FICTION = process.env.AQ_FICTION !== "keep"; // exclude fiction areas from harvesting
// Precision gates under test (R1-R8), each independently toggleable for the grid.
const R1_QUOTEFIX = process.env.AQ_R1 === "1";      // quote-aware terminal-? detection
const R2_BATTERY = process.env.AQ_R2 === "1";       // consecutive-question brainstorm-battery harvest veto
const R3_SECONDPERSON = process.env.AQ_R3 === "1";  // second-person (dialogue/system) harvest veto
const R4_QUOTEDANS = process.env.AQ_R4 === "1";     // quoted-speech answer veto
const R5_TAGBREADTH = process.env.AQ_R5 === "1";    // organizational tags (too broad) don't count as areas
const R5_BREADTH_MAX = Number(process.env.AQ_R5_MAX ?? 0.25); // tag on > this fraction of tagged notes = organizational
const R6_ANCHOR = process.env.AQ_R6 === "1";        // >=1 question content word must appear in answer NOTE text
const R6_STRONG = Number(process.env.AQ_R6_STRONG ?? 0); // >0: strong-similarity fallback rescues anchor-less pairs (cross-language)
const R7_FRAGMIN = process.env.AQ_R7 === "1";       // uncertainty-harvested questions need >=8 words, >=2 content words
const R8_SOURCECAP = Number(process.env.AQ_R8 ?? 0); // >0: max sparks per question-source note in mined list
const R9_TABLES = process.env.AQ_R9 === "1";        // markdown table rows are neither questions nor answers

// ---------- fs walk ----------
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

// ---------- umlaut-folding normalize for anchor matching ----------
function fold(s) {
  return s
    .toLowerCase()
    .normalize("NFC")
    .replace(/ä/g, "ae").replace(/ö/g, "oe").replace(/ü/g, "ue").replace(/ß/g, "ss")
    .replace(/\s+/g, " ")
    .trim();
}

// ---------- frontmatter aliases + wikilinks ----------
function parseFront(raw) {
  const m = raw.match(/^---\n([\s\S]*?)\n---\n?/);
  const aliases = [];
  const tags = new Set();
  if (m) {
    const fm = m[1];
    const am = fm.match(/aliases:\s*\[([^\]]*)\]/);
    if (am) for (const a of am[1].split(",")) { const t = a.trim().replace(/^["']|["']$/g, ""); if (t) aliases.push(t); }
    const addTag = (t) => { const x = t.trim().replace(/^["'#]+|["']$/g, "").toLowerCase(); if (x) tags.add(x.split("/")[0]); };
    const tm = fm.match(/^tags:\s*\[([^\]]*)\]/m);
    if (tm) for (const t of tm[1].split(",")) addTag(t);
    else {
      const block = fm.match(/^tags:\s*\n((?:\s*-\s*.+\n?)+)/m);
      if (block) for (const line of block[1].split("\n")) { const lm = line.match(/-\s*(.+)/); if (lm) addTag(lm[1]); }
    }
  }
  return { aliases, tags, body: m ? raw.slice(m[0].length) : raw };
}
function wikilinks(body) {
  const out = new Set();
  const re = /\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]/g;
  let m;
  while ((m = re.exec(body)) !== null) out.add(m[1].trim());
  return out;
}

// ---------- sentence split (matches the plugin's Intl.Segmenter usage) ----------
// Segments per source LINE and reports the line index, so adjacency rules
// (brainstorm batteries live on one line) can see structure.
const seg = new Intl.Segmenter(undefined, { granularity: "sentence" });
function sentences(text) {
  const out = [];
  const clean = text
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/%%[\s\S]*?%%/g, " ")            // author comments are not content
    .replace(/<[^>]+>/g, " ")                  // html (font tags etc.)
    .replace(/==/g, "")                        // highlight markers
    .replace(/\[\[([^\]|]+)\|([^\]]+)\]\]/g, "$2")
    .replace(/\[\[([^\]]+)\]\]/g, "$1")
    .replace(/^\s{0,3}(#{1,6}|>|[-*+]|\d+\.)\s+/gm, "")
    .replace(/[*_~`]/g, "");
  const lines = clean.split("\n");
  for (let li = 0; li < lines.length; li++) {
    // R9: table rows ("| a | b |") are layout, not prose — neither questions nor answers.
    if (R9_TABLES && (/^\s*\|/.test(lines[li]) || (lines[li].match(/ \| /g) || []).length >= 2)) continue;
    let si = 0;
    for (const { segment } of seg.segment(lines[li])) {
      const s = segment.replace(/\s+/g, " ").trim();
      if (s.length >= 8) out.push({ text: s, line: li, sent: si++ });
    }
  }
  return out;
}
// Question-ish: terminal "?" OR an open-question restatement (uncertainty marker
// + interrogative word). A restatement of an open question is itself an open
// question however it is phrased; marking it question-ish both vetoes it as an
// "answer" and harvests it as a question source.
// Returns "qmark" | "uncert" | false so downstream rules can tell how it matched.
const UNCERT = /(ungekl(ae|ä)rt|unklar|bleibt offen|noch offen|offene frage|frage an mich|wei(ss|ß) (noch )?nicht|keine ahnung|herausfinden|muss ich .*(kl(ae|ä)ren|pr(ue|ü)fen)|unclear|still open|open question|not sure|don'?t know|need to figure|figure out|wonder(ing)?\s+(whether|if|how|why|what))/i;
const INTERROG = /\b(warum|wieso|weshalb|wie|ob|wann|wof(ue|ü)r|welche[rsnm]?|why|how|whether|when|what|which|where)\b/i;
const TRAIL_QUOTES = /["'"”„“«»‹›\)\]\s]+$/;
const isQuestion = (s) => {
  let t = s.trim();
  if (R1_QUOTEFIX) t = t.replace(TRAIL_QUOTES, "");
  if (/\?$/.test(t) || /\?\s*$/.test(s.trim())) return "qmark";
  return UNCERT.test(t) && INTERROG.test(t) ? "uncert" : false;
};
// R3: second-person address (dialogue or in-world system text) with no
// first-person marker. German "ihr" is skipped: too ambiguous (her/their).
const SECOND_PERSON = /\b(you|your|yours|du|dich|dir|dein(e[srnm]?)?|euch|euer|eure[srnm]?)\b/i;
const FIRST_PERSON = /\b(i|i'?m|i'?ve|my|mine|we|us|our|let'?s|ich|mir|mich|mein(e[srnm]?)?|wir|uns|unser(e[srnm]?)?)\b/i;
const isSecondPersonAddress = (s) => SECOND_PERSON.test(s) && !FIRST_PERSON.test(s);
// R4: quoted speech (the cleaning pass keeps quote characters).
const isQuotedSpeech = (s) => {
  const t = s.trim();
  return /^["'"„“«»‹]/.test(t) || /["'"”“«»›]$/.test(t);
};

// ---------- vector math ----------
const l2 = (v) => { let s = 0; for (const x of v) s += x * x; s = Math.sqrt(s) || 1; return v.map((x) => x / s); };
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const centroidOf = (vs) => { const d = vs[0].length, c = new Array(d).fill(0); for (const v of vs) for (let i = 0; i < d; i++) c[i] += v[i]; for (let i = 0; i < d; i++) c[i] /= vs.length; return l2(c); };
const center = (v, c) => { const p = dot(v, c); return l2(v.map((x, i) => x - p * c[i])); };

async function embedAll(extractor, texts) {
  // Disk cache so grid runs (same sentences, different gates) skip re-embedding.
  const cachePath = process.env.AQ_CACHE || "/private/tmp/claude-501/-Users-justus-obsidian-atomized-intermediary/fdd9de71-217a-43e1-9ca7-fb239cfd5cb4/scratchpad/aq-embed-cache.json";
  let cache = {};
  try { cache = JSON.parse(readFileSync(cachePath, "utf8")); if (cache.__model !== MODEL) cache = {}; } catch { }
  const missing = [...new Set(texts.filter((t) => !cache[t]))];
  if (missing.length) {
    console.log(`embedding ${missing.length} new sentences (${texts.length - missing.length} cached) ...`);
    for (let i = 0; i < missing.length; i += 32) {
      const batch = missing.slice(i, i + 32);
      const t = await extractor(batch, { pooling: "mean", normalize: true });
      const vs = t.tolist();
      batch.forEach((txt, j) => { cache[txt] = vs[j]; });
    }
    cache.__model = MODEL;
    try { writeFileSync(cachePath, JSON.stringify(cache)); } catch { }
  }
  return texts.map((t) => cache[t]);
}

// Fiction/prose detection. Synthetic lab: Szenen/ folders. Real vault: the novel
// lives as numbered chapter files ("Personal/12. Breakthrough.md"). Planning notes
// (GoA Plot, Storyline Ideas...) are NOT fiction; their questions are genuine.
const isFiction = (rel) =>
  /(^|\/)Szenen\//.test(rel) ||
  /(^|\/)\d+\.\s[^/]+\.md$/.test(rel);

// ---------- main ----------
const gt = JSON.parse(readFileSync(GT_PATH, "utf8"));
// Authentic-corpus manifest: a parallel session keeps generating synthetic notes
// under real/. Only manifest question paths count as the user's own writing, and
// only snapshotted answer paths enter the index at all — contamination is inert.
const manifest = JSON.parse(readFileSync("/Users/justus/obsidian_atomized_intermediary/lab/corpus-manifest.json", "utf8"));
const AUTHENTIC_Q = new Set(manifest.paths.map((p) => "real/" + p));
const INDEXABLE_REAL = new Set(manifest.answer_paths.map((p) => "real/" + p));
const files = walk(VAULT).filter((abs) => {
  const rel = relative(VAULT, abs);
  return !rel.startsWith("real/") || INDEXABLE_REAL.has(rel);
});
if (files.length === 0) { console.error("No notes found in", VAULT); process.exit(1); }

// Build a flat sentence table across the vault.
const units = []; // { rel, basename, isQ, qKind, battery, quoted, text, line, sent }
const linkOut = new Map(); // basename -> Set(basenames)
const noteTags = new Map(); // basename -> Set(top-level lowercase tags)
const noteText = new Map(); // basename -> folded full text (for R6 content-anchor)
for (const abs of files) {
  const rel = relative(VAULT, abs);
  const basename = rel.replace(/\.md$/, "").split("/").pop();
  const raw = readFileSync(abs, "utf8");
  const { aliases, tags, body } = parseFront(raw);
  linkOut.set(basename, wikilinks(body));
  noteTags.set(basename, tags);
  noteText.set(basename, fold(basename + " " + aliases.join(" ") + " " + body));
  const hub = /\bMOC\b/i.test(basename);
  const noteUnits = [];
  for (const s of sentences(body)) {
    const kind = isQuestion(s.text);
    noteUnits.push({ rel, basename, isQ: !!kind, qKind: kind, battery: false,
      quoted: isQuotedSpeech(s.text), isHub: hub, text: s.text, line: s.line, sent: s.sent });
  }
  // R2: mark brainstorm batteries — >=2 question sentences immediately adjacent
  // on the SAME line. Genuine tracked questions start a paragraph and are
  // followed by declarative elaboration; batteries are strung-together prompts.
  for (let i = 0; i < noteUnits.length - 1; i++) {
    const a = noteUnits[i], b = noteUnits[i + 1];
    if (a.isQ && b.isQ && a.line === b.line && b.sent === a.sent + 1) {
      a.battery = true; b.battery = true;
    }
  }
  units.push(...noteUnits);
}
// R5: tag breadth — a tag carried by more than R5_BREADTH_MAX of tagged notes is
// organizational (personal, concepts, ...), not topical, and defines no area.
const tagCount = new Map();
let taggedNotes = 0;
for (const [, tags] of noteTags) {
  if (!tags.size) continue;
  taggedNotes++;
  for (const t of tags) tagCount.set(t, (tagCount.get(t) ?? 0) + 1);
}
const isTopicalTag = (t) => (tagCount.get(t) ?? 0) / (taggedNotes || 1) <= R5_BREADTH_MAX;
const isChapterNote = (bn) => noteTags.get(bn)?.has("chapter") ?? false;
const sharesArea = (a, b) => {
  const ta = noteTags.get(a), tb = noteTags.get(b);
  if (!ta?.size || !tb?.size) return true; // untagged notes are not gated
  if (!R5_TAGBREADTH) {
    for (const t of ta) if (t !== "personal" && tb.has(t)) return true; // legacy: hardcoded 'personal' exclusion
    return false;
  }
  const topA = [...ta].filter(isTopicalTag), topB = new Set([...tb].filter(isTopicalTag));
  if (!topA.length || !topB.size) return true; // only organizational tags = effectively untagged
  for (const t of topA) if (topB.has(t)) return true;
  return false;
};
console.log(`vault: ${files.length} notes, ${units.length} sentences. embedding with ${MODEL} ...`);

const extractor = await pipeline("feature-extraction", MODEL, { dtype: "fp32" });
const rawVecs = await embedAll(extractor, units.map((u) => u.text));
const centroid = centroidOf(rawVecs.map(l2));
const vecs = rawVecs.map((v) => center(l2(v), centroid));
units.forEach((u, i) => { u.v = vecs[i]; });

// Per-note mean of centered sentence vectors (the plugin's note-level signal),
// used to shortlist notes before best-passage answer search.
const byNote = new Map();
for (const u of units) { if (!byNote.has(u.basename)) byNote.set(u.basename, []); byNote.get(u.basename).push(u); }
const noteMean = new Map();
for (const [bn, us] of byNote) { const d = us[0].v.length, m = new Array(d).fill(0); for (const u of us) for (let i = 0; i < d; i++) m[i] += u.v[i]; noteMean.set(bn, l2(m)); }

// Content-word overlap, to veto DECLARATIVE restatements of the question (the
// question-veto only catches question-shaped ones; pt10 proved that gap).
const STOP = new Set("der die das und oder in im auf mit ein eine einer den dem des zu zur zum von fuer ist sind war wird wie was warum wann wo wer wenn dass nicht auch nur the a an of to in on for is are was how why what when where who and or with that this it as be".split(" "));
function contentSet(s) { const out = new Set(); for (const w of fold(s).split(/[^a-z0-9]+/)) if (w.length >= 4 && !STOP.has(w)) out.add(w); return out; }
// R6 matches on a prefix stem so German inflection still anchors
// (Sattelpunkte -> sattelpun matches Sattelpunkt; Eigenvektoren -> Eigenvektor).
const anchorStem = (w) => w.slice(0, Math.max(4, Math.ceil(w.length * 0.75)));
function jaccard(a, b) { if (!a.size || !b.size) return 0; let i = 0; for (const x of a) if (b.has(x)) i++; return i / (a.size + b.size - i); }
const NSHORT = Number(process.env.AQ_NSHORT ?? 15);
const LEX_VETO = Number(process.env.AQ_LEX ?? 0.55);
const MIN_ANS_CHARS = 40;

// locate a planted anchor -> unit index
function findUnit(file, anchor) {
  const relWant = fold(file);
  const aWant = fold(anchor);
  let idx = -1;
  for (let i = 0; i < units.length; i++) {
    if (fold(units[i].rel) !== relWant) continue;
    if (fold(units[i].text).includes(aWant)) return i;
    if (idx === -1 && fold(units[i].text).includes(aWant.slice(0, 40))) idx = i;
  }
  return idx;
}

const linked = (aBase, bBase) => (linkOut.get(aBase)?.has(bBase) ?? false) || (linkOut.get(bBase)?.has(aBase) ?? false);

// Harvest-side vetoes (R2/R3/R7): would this sentence be mined as a question at all?
// Returns the veto reason or null. Applied to mining AND reported for planted QAs,
// so a rule that silently kills a genuine planted question shows up in recall.
function harvestVeto(u) {
  if (!u.isQ) return "not-question";
  if (R1_QUOTEFIX && u.quoted) return "quoted-dialogue"; // a quoted question is speech, not a tracked question
  if (R2_BATTERY && u.battery) return "battery";
  if (R3_SECONDPERSON && isSecondPersonAddress(u.text)) return "second-person";
  if (R7_FRAGMIN && u.qKind === "uncert") {
    if (u.text.split(/\s+/).length < 8 || contentSet(u.text).size < 2) return "fragment";
  }
  return null;
}
// Answer-side veto check for planted answer-traps: is this sentence eliminated as
// an answer candidate by the classifier gates (question-veto / quote-veto)?
const answerVetoed = (u) => u.isQ || (R4_QUOTEDANS && u.quoted);

// Score one question against candidate answers; return ranked list.
// gated=false: global sentence cosine (raw baseline).
// gated=true: shortlist notes by note-mean, then best declarative passage,
//   with floor+ceiling, min-length, question-veto, and lexical-restatement veto.
function rankAnswers(qUnit, { gated, poolFilter }) {
  const qWords = contentSet(qUnit.text);
  let pool = poolFilter ? units.filter(poolFilter) : units;
  if (gated) {
    const shortlistable = new Set(pool.map((u) => u.basename));
    const top = [...noteMean].filter(([bn]) => bn !== qUnit.basename && !/\bMOC\b/i.test(bn) && shortlistable.has(bn))
      .map(([bn, m]) => [bn, dot(qUnit.v, m)]).sort((a, b) => b[1] - a[1]).slice(0, NSHORT);
    const keep = new Set(top.map(([bn]) => bn));
    pool = pool.filter((u) => keep.has(u.basename));
  }
  const scored = [];
  for (const c of pool) {
    if (c.basename === qUnit.basename) continue;
    const s = dot(qUnit.v, c.v);
    if (gated) {
      if (c.isQ) continue;                                   // question veto
      if (c.text.length < MIN_ANS_CHARS) continue;           // answers are substantive
      if (s < FLOOR || s > CEIL) continue;                   // floor + paraphrase ceiling
      if (jaccard(qWords, contentSet(c.text)) > LEX_VETO) continue; // declarative-restatement veto
      if (R4_QUOTEDANS && c.quoted) continue;                // R4: quoted speech is not an answer
      if (R6_ANCHOR && !(R6_STRONG > 0 && s >= R6_STRONG)
        && ![...qWords].some((w) => noteText.get(c.basename)?.includes(anchorStem(w)))) continue; // R6: content anchor OR strong similarity
    }
    scored.push({ u: c, s });
  }
  scored.sort((a, b) => b.s - a.s);
  return scored;
}

const rankOf = (ranked, unitIdx) => {
  const target = units[unitIdx];
  const i = ranked.findIndex((r) => r.u === target);
  return i === -1 ? Infinity : i + 1;
};

// ---- QA evaluation ----
const qaRows = [];
for (const qa of gt.qa_pairs) {
  const qi = findUnit(qa.q_file, qa.q_anchor);
  const ai = findUnit(qa.a_file, qa.a_anchor);
  if (qi < 0 || ai < 0) { qaRows.push({ id: qa.id, missing: { q: qi < 0, a: ai < 0 } }); continue; }
  const qUnit = units[qi];
  const rawRanked = rankAnswers(qUnit, { gated: false });
  const gatedRanked = rankAnswers(qUnit, { gated: true });
  // note-level rank: position of the answer's NOTE among distinct gated notes
  const noteOrder = []; const seen = new Set();
  for (const r of gatedRanked) if (!seen.has(r.u.basename)) { seen.add(r.u.basename); noteOrder.push(r.u.basename); }
  const nr = noteOrder.indexOf(units[ai].basename);
  qaRows.push({
    id: qa.id,
    harvestVeto: harvestVeto(qUnit), // non-null = this rule set would never mine the question
    rawRank: rankOf(rawRanked, ai),
    gatedRank: rankOf(gatedRanked, ai),
    noteRank: nr === -1 ? Infinity : nr + 1,
    linkedAlready: linked(units[qi].basename, units[ai].basename),
    topGated: gatedRanked[0] ? `${gatedRanked[0].u.basename}: ${gatedRanked[0].u.text.slice(0, 55)}` : "(none)",
    answerScore: dot(qUnit.v, units[ai].v).toFixed(3),
  });
}

// ---- paraphrase trap: does a restatement outrank the true answer raw, and is it gated out? ----
const ptRows = [];
for (const pt of gt.paraphrase_traps || []) {
  const qa = gt.qa_pairs.find((q) => q.id === pt.of);
  if (!qa) continue;
  const qi = findUnit(qa.q_file, qa.q_anchor);
  const ai = findUnit(qa.a_file, qa.a_anchor);
  const ti = findUnit(pt.file, pt.anchor);
  if (qi < 0 || ai < 0 || ti < 0) { ptRows.push({ id: pt.id, missing: true }); continue; }
  const q = units[qi];
  const rawRanked = rankAnswers(q, { gated: false });
  const gatedRanked = rankAnswers(q, { gated: true });
  ptRows.push({
    id: pt.id,
    trapOutranksAnswerRaw: rankOf(rawRanked, ti) < rankOf(rawRanked, ai), // the PROBLEM
    trapSurvivesGate: rankOf(gatedRanked, ti) !== Infinity,               // the FIX (should be false)
    trapIsQuestion: units[ti].isQ,
  });
}

// ---- fiction leak: do dialogue questions surface any gated answer? ----
const fictionRows = [];
for (const fd of gt.fiction_dialogue_traps || []) {
  const fi = findUnit(fd.file, fd.anchor);
  if (fi < 0) { fictionRows.push({ id: fd.id, missing: true }); continue; }
  const q = units[fi];
  const excludedByArea = EXCLUDE_FICTION && isFiction(q.rel);
  const veto = harvestVeto(q);
  const gatedRanked = excludedByArea || veto ? [] : rankAnswers(q, { gated: true });
  fictionRows.push({
    id: fd.id,
    excludedByArea,
    harvestVeto: veto,
    leakedSparkCount: gatedRanked.length,
    topLeak: gatedRanked[0] ? `${gatedRanked[0].u.basename}: ${gatedRanked[0].u.text.slice(0, 50)}` : "(none)",
  });
}

// ---- new precision traps: harvest-side (battery / in-world / fragment) must be
// vetoed at harvest; answer-side (quoted dialogue) must be vetoed as answers. ----
const trapRows = [];
for (const [kind, list] of [["battery", gt.battery_traps], ["inworld", gt.inworld_traps], ["fragment", gt.fragment_traps]]) {
  for (const tr of list || []) {
    const ti = findUnit(tr.file, tr.anchor);
    if (ti < 0) { trapRows.push({ id: tr.id, kind, missing: true }); continue; }
    const u = units[ti];
    const veto = harvestVeto(u);
    const excludedByArea = EXCLUDE_FICTION && isFiction(u.rel);
    trapRows.push({ id: tr.id, kind, harvested: !veto && !excludedByArea, veto: veto ?? (excludedByArea ? "fiction-area" : null) });
  }
}
for (const tr of gt.quoted_answer_traps || []) {
  const ti = findUnit(tr.file, tr.anchor);
  if (ti < 0) { trapRows.push({ id: tr.id, kind: "quoted-answer", missing: true }); continue; }
  const u = units[ti];
  trapRows.push({ id: tr.id, kind: "quoted-answer", offeredAsAnswer: !answerVetoed(u), isQ: u.isQ, quoted: u.quoted });
}

// ---- mine the REAL vault for sparks: your questions, your answers ----
const TAG = process.env.AQ_TAG || "";
const minedRows = [];
let minedPrecision = null;
if (process.env.AQ_MINE !== "0") {
  // Question sources: all sanctioned real/ notes (AQ_QSRC=authentic restricts to the
  // user's own export); provenance travels with each mined pair either way.
  const qsrcOk = (u) => process.env.AQ_QSRC === "authentic" ? AUTHENTIC_Q.has(u.rel) : u.rel.startsWith("real/");
  const realQ = units.filter((u) => qsrcOk(u) && !harvestVeto(u) && u.text.split(/\s+/).length >= 6 && !isFiction(u.rel) && !isChapterNote(u.basename));
  // answers may come from anywhere in the real vault INCLUDING prose chapters
  // (a chapter can answer a planning question), but never from author questions,
  // and only from the SAME tag area (a plot question is never answered by a physics note).
  const isReal = (u) => u.rel.startsWith("real/");
  const sparks = [];
  for (const q of realQ) {
    const ranked = rankAnswers(q, { gated: true, poolFilter: isReal })
      .filter((r) => !linked(q.basename, r.u.basename) && sharesArea(q.basename, r.u.basename));
    if (ranked[0]) sparks.push({ q, a: ranked[0].u, s: ranked[0].s });
  }
  sparks.sort((a, b) => b.s - a.s);
  const seenPair = new Set();
  const perSource = new Map();
  for (const sp of sparks) {
    const k = `${sp.q.basename}>${sp.a.basename}`;
    if (seenPair.has(k)) continue;
    if (R8_SOURCECAP > 0 && (perSource.get(sp.q.basename) ?? 0) >= R8_SOURCECAP) continue;
    seenPair.add(k);
    perSource.set(sp.q.basename, (perSource.get(sp.q.basename) ?? 0) + 1);
    minedRows.push(sp);
    if (minedRows.length >= 20) break;
  }
  const md = [
    "# Sparks review: questions in your vault that may already have answers",
    "",
    "Mined from your real notes only (synthetic lab notes excluded from both sides).",
    "Judge each: does this feel like a genuine \"I never noticed\" moment? The bet",
    "needs roughly 5 of 20 to feel real.",
    "",
    ...minedRows.map((sp, i) => [
      `## ${i + 1}. score ${sp.s.toFixed(3)}`,
      `**Your question** (\`${sp.q.rel.replace(/^real\//, "")}\`):`,
      `> ${sp.q.text}`,
      `**Possible answer** (\`${sp.a.rel.replace(/^real\//, "")}\`):`,
      `> ${sp.a.text}`,
      "",
    ].join("\n")),
  ].join("\n");
  writeFileSync(`/Users/justus/obsidian_atomized_intermediary/lab/results/sparks-review${TAG ? "-" + TAG : ""}.md`, md);
  // Stable-keyed JSON so judge labels survive re-mining and config changes.
  writeFileSync(`/Users/justus/obsidian_atomized_intermediary/lab/results/sparks-mined${TAG ? "-" + TAG : ""}.json`,
    JSON.stringify(minedRows.map((sp) => ({ key: `${fold(sp.q.text).slice(0, 60)}>>${sp.a.basename}`,
      provenance: AUTHENTIC_Q.has(sp.q.rel) ? "authentic" : "generated",
      qfile: sp.q.rel, qtext: sp.q.text, afile: sp.a.rel, atext: sp.a.text, score: +sp.s.toFixed(4) })), null, 2));
  // Precision against judged labels (lab/results/spark-labels.json), if present.
  try {
    const labels = JSON.parse(readFileSync("/Users/justus/obsidian_atomized_intermediary/lab/results/spark-labels.json", "utf8"));
    const lmap = new Map(labels.map((l) => [l.key, l.label]));
    let real = 0, fal = 0, unknown = 0;
    for (const sp of minedRows) {
      const v = lmap.get(`${fold(sp.q.text).slice(0, 60)}>>${sp.a.basename}`);
      if (v === "REAL") real++; else if (v === "FALSE") fal++; else unknown++;
    }
    minedPrecision = { labeled: real + fal, real, false: fal, unlabeled: unknown,
      precision: real + fal ? +(real / (real + fal)).toFixed(3) : null };
  } catch { /* no labels yet */ }
  console.log(`\n==== mined sparks (real vault): ${realQ.length} questions harvested, top ${minedRows.length} kept ====`);
  for (const sp of minedRows.slice(0, 8)) console.log(`  ${sp.s.toFixed(3)}  Q[${sp.q.basename}] ${sp.q.text.slice(0, 56)}  ->  A[${sp.a.basename}] ${sp.a.text.slice(0, 56)}`);
}

// ---- summary ----
const ok = qaRows.filter((r) => !r.missing);
const recallAt = (k, key) => ok.filter((r) => r[key] <= k && !r.harvestVeto).length / (ok.length || 1);
const summary = {
  model: MODEL,
  gates: { FLOOR, CEIL, LEN_RATIO, EXCLUDE_FICTION,
    R1_QUOTEFIX, R2_BATTERY, R3_SECONDPERSON, R4_QUOTEDANS,
    R5_TAGBREADTH, R5_BREADTH_MAX, R6_ANCHOR, R7_FRAGMIN, R8_SOURCECAP },
  qa_found: `${ok.length}/${qaRows.length}`,
  qa_missing_anchors: qaRows.filter((r) => r.missing).map((r) => r.id),
  qa_harvest_killed: ok.filter((r) => r.harvestVeto).map((r) => `${r.id}:${r.harvestVeto}`),
  raw_recall_at_1: recallAt(1, "rawRank").toFixed(2),
  gated_sentence_recall_at_1: recallAt(1, "gatedRank").toFixed(2),
  gated_sentence_recall_at_3: recallAt(3, "gatedRank").toFixed(2),
  gated_NOTE_recall_at_1: recallAt(1, "noteRank").toFixed(2),
  gated_NOTE_recall_at_3: recallAt(3, "noteRank").toFixed(2),
  paraphrase_problem_rate: (ptRows.filter((r) => r.trapOutranksAnswerRaw).length / (ptRows.filter(r=>!r.missing).length || 1)).toFixed(2),
  paraphrase_still_leaks_after_gate: (ptRows.filter((r) => r.trapSurvivesGate).length / (ptRows.filter(r=>!r.missing).length || 1)).toFixed(2),
  fiction_leaks: fictionRows.filter((r) => !r.excludedByArea && !r.harvestVeto && r.leakedSparkCount > 0).length,
  trap_leaks: {
    battery: trapRows.filter((r) => r.kind === "battery" && r.harvested).length + "/" + trapRows.filter((r) => r.kind === "battery" && !r.missing).length,
    inworld: trapRows.filter((r) => r.kind === "inworld" && r.harvested).length + "/" + trapRows.filter((r) => r.kind === "inworld" && !r.missing).length,
    fragment: trapRows.filter((r) => r.kind === "fragment" && r.harvested).length + "/" + trapRows.filter((r) => r.kind === "fragment" && !r.missing).length,
    quoted_answer: trapRows.filter((r) => r.kind === "quoted-answer" && r.offeredAsAnswer).length + "/" + trapRows.filter((r) => r.kind === "quoted-answer" && !r.missing).length,
  },
  trap_missing: trapRows.filter((r) => r.missing).map((r) => `${r.kind}:${r.id}`),
  mined_count: minedRows.length,
  mined_precision: minedPrecision,
};

console.log("\n==== QA (question -> true answer) ====");
for (const r of qaRows) console.log(r.missing ? `${r.id}  MISSING q=${r.missing.q} a=${r.missing.a}` :
  `${r.id}  raw#${r.rawRank}  gated#${r.gatedRank}  note#${r.noteRank}  score ${r.answerScore}  harvestVeto=${r.harvestVeto}  top: ${r.topGated}`);
console.log("\n==== paraphrase traps ====");
for (const r of ptRows) console.log(r.missing ? `${r.id} MISSING` :
  `${r.id}  trapBeatsAnswerRaw=${r.trapOutranksAnswerRaw}  survivesGate=${r.trapSurvivesGate}  isQuestion=${r.trapIsQuestion}`);
console.log("\n==== fiction dialogue traps ====");
for (const r of fictionRows) console.log(r.missing ? `${r.id} MISSING` :
  `${r.id}  excludedByArea=${r.excludedByArea}  harvestVeto=${r.harvestVeto}  leaks=${r.leakedSparkCount}  ${r.topLeak}`);
console.log("\n==== precision traps ====");
for (const r of trapRows) console.log(r.missing ? `${r.kind}:${r.id} MISSING` :
  r.kind === "quoted-answer" ? `${r.kind}:${r.id}  offeredAsAnswer=${r.offeredAsAnswer} (isQ=${r.isQ} quoted=${r.quoted})`
  : `${r.kind}:${r.id}  harvested=${r.harvested}  veto=${r.veto}`);
console.log("\n==== SUMMARY ====");
console.log(JSON.stringify(summary, null, 2));

writeFileSync(OUT, JSON.stringify({ summary, qaRows, ptRows, fictionRows, trapRows }, null, 2));
console.log("\nwrote", OUT);
