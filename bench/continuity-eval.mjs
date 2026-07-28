// Continuity-pairs validation harness for the lab vault. v2 (2026-07-02).
//
// Feature idea: surface two notes that touch the SAME specific detail (a price,
// a physical trait, a named thing, a rare term) but never link to each other.
// Precision comes from rarity: a detail counts only if almost no other note has
// it. Semantic similarity is a secondary signal, gated so hubs and mundane
// phrase collisions do not surface.
//
//   node bench/continuity-eval.mjs
//
// v2 adds env-toggled precision gates (ALL default-off; defaults = v1 exactly):
//   CP_SENTDF=3      G1  drop folded sentences/lines appearing in >= N notes
//                        (boilerplate dedup) from means, sem channel, and
//                        detail extraction
//   CP_LANG=1        G2  language-frequency gate: kill a token iff ALL its words
//                        are language-common (bench/data/{de,en}_50k.txt,
//                        hermitdave FrequencyWords 2018, folded+suffix-stripped
//                        lookup, thresholds CP_DE / CP_EN) AND the token has no
//                        corpus term evidence (title/alias/heading/bold/link
//                        display span, tokenized stem-equality, stem >= 5)
//   CP_TRIGRAM=1     G3  number-bigram referent consistency: next content word
//                        after the bigram must not differ between the notes
//   CP_REFSINGLE=x   G4  stricter refCos bar for single-word tokens (phrases
//                        keep CP_REFMIN); CP_REFSCOPE=1 extends the referent
//                        gate to all in-scope pairs, not just the global head
//   CP_COCITE=1      G5  kill tokens that are a link-target's name linked in
//                        BOTH notes (co-citation is visible in backlinks
//                        already); CP_STOP2=1 adds discourse words to STOP
//   CP_EXACTLINE=1   G6  kill tokens whose source line is an identical folded
//                        copy-paste line (>= 30 chars) in both notes
//   CP_SEMMINLEN=25      sem channel: min sentence length
//   CP_SEMEXACT=1        sem channel: skip identical folded sentence matches
//   CP_NOSINGLE=1        drop single-word tokens as pair formers entirely
//
// Metrics against lab/ground-truth.json:
//   continuity_pairs     -> planted pairs must rank (channel "lex" or "sem";
//                           stretch entries reported but non-gating)
//   continuity_negatives -> traps must NOT surface top-K (stretch = non-gating)
//   near_dupes           -> must be excluded by the dupe gate, not surfaced
//   generic_bridge_trap  -> hubby essay must not dominate the pair list
// Startup checks: every anchor must resolve (else hard exit) and plants must
// not shift the df of tokens behind the previously judged 15 pairs.
// Plus: top real-vault pairs written for judging with token-centered receipts.
import { pipeline, env } from "@huggingface/transformers";
import { readFileSync, writeFileSync, readdirSync, statSync, existsSync } from "node:fs";
import { join, relative, dirname } from "node:path";
import { fileURLToPath } from "node:url";

env.allowLocalModels = false;

const BENCH = dirname(fileURLToPath(import.meta.url));
const VAULT = process.env.LAB_VAULT || "/Users/justus/obsidian_atomized_intermediary/lab/vault";
const GT_PATH = process.env.LAB_GT || "/Users/justus/obsidian_atomized_intermediary/lab/ground-truth.json";
const OUT = process.env.LAB_OUT || "/Users/justus/obsidian_atomized_intermediary/lab/results/continuity-eval.json";
const MODEL = process.env.LAB_MODEL || "Xenova/paraphrase-multilingual-MiniLM-L12-v2";
const TOPK = Number(process.env.CP_TOPK ?? 20);
const MAX_DF = Number(process.env.CP_MAXDF ?? 3);      // a detail token may appear in at most this many notes
const MIN_COS = Number(process.env.CP_MINCOS ?? 0.20); // pair must also be loosely related semantically
const DUPE_COS = Number(process.env.CP_DUPE ?? 0.80);  // above this, it's a near-dupe, not a continuity find
const SEM_CHANNEL = process.env.CP_SEM !== "0";        // cross-language channel: high-sim unlinked same-area pairs

// ---- v2 gate switches (defaults preserve v1) ----
const G1_SENTDF = Number(process.env.CP_SENTDF ?? 0);
const G2_LANG = process.env.CP_LANG === "1";
const DE_T = Number(process.env.CP_DE ?? 16000);
const EN_T = Number(process.env.CP_EN ?? 8000);
const G3_TRIGRAM = process.env.CP_TRIGRAM === "1";
const REF_SINGLE = Number(process.env.CP_REFSINGLE ?? 0);
const REF_SCOPE = process.env.CP_REFSCOPE === "1";
const G5_COCITE = process.env.CP_COCITE === "1";
const STOP2 = process.env.CP_STOP2 === "1";
const G6_EXACTLINE = process.env.CP_EXACTLINE === "1";
const SEM_MINLEN = Number(process.env.CP_SEMMINLEN ?? 0);
const SEM_EXACT_SKIP = process.env.CP_SEMEXACT === "1";
const NOSINGLE = process.env.CP_NOSINGLE === "1";
const SEM_CAND = Number(process.env.CP_SEMCAND ?? 300);
const REF_RARE = Number(process.env.CP_REFRARE ?? 0);  // >0: skip referent gate for self-certifying rare tokens
const REF_CLEAN = process.env.CP_REFCLEAN === "1";     // refCos over cleaned lines, not raw lines
const G7_HEADKILL = process.env.CP_HEADKILL === "1";   // G7: a bare list-header/heading line is not a detail-bearing receipt

// ---------- shared helpers (mirrors aq-eval.mjs) ----------
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
  const tags = new Set(), aliases = [];
  if (m) {
    const fm = m[1];
    const addTag = (t) => { const x = t.trim().replace(/^["'#]+|["']$/g, "").toLowerCase(); if (x) tags.add(x.split("/")[0]); };
    const tm = fm.match(/^tags:\s*\[([^\]]*)\]/m);
    if (tm) for (const t of tm[1].split(",")) addTag(t);
    else {
      const block = fm.match(/^tags:\s*\n((?:\s*-\s*.+\n?)+)/m);
      if (block) for (const line of block[1].split("\n")) { const lm = line.match(/-\s*(.+)/); if (lm) addTag(lm[1]); }
    }
    const am = fm.match(/^aliases:\s*\[([^\]]*)\]/m);
    if (am) for (const a of am[1].split(",")) { const x = a.trim().replace(/^["']+|["']+$/g, ""); if (x) aliases.push(x); }
    else {
      const ab = fm.match(/^aliases:\s*\n((?:\s*-\s*.+\n?)+)/m);
      if (ab) for (const line of ab[1].split("\n")) { const lm = line.match(/-\s*(.+)/); if (lm) aliases.push(lm[1].trim().replace(/^["']+|["']+$/g, "")); }
    }
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

// ---------- detail-token extraction ----------
// Two channels: (a) number+noun bigrams ("drei kupferstuecke", "17 zyklen"),
// (b) rare content words (len>=6). Both folded; df-capped later.
const NUMWORD = /^(zwei|drei|vier|fuenf|sechs|sieben|acht|neun|zehn|elf|zwoelf|zwanzig|dreissig|vierzig|fuenfzig|hundert|tausend|siebzehn|two|three|four|five|six|seven|eight|nine|ten|twelve|twenty|forty|fifty|hundred|thousand|seventeen|\d+[.,]?\d*)$/;
const STOP = new Set(("der die das und oder in im auf mit ein eine einer einem einen den dem des zu zur zum von fuer ist sind war wird werden wie was warum wann wo wer wenn dass nicht auch nur noch schon mehr sehr kann muss soll jahr jahre jahren tag tage mal durch nach vor beim ohne gegen unter ueber wieder immer etwas anders diese dieser dieses jede jeder jedes " +
  "the a an of to in on for is are was were how why what when where who and or with that this it as be at by from has have had not but they them their there here just really only very more most much many then every these those something different against while during would could should about into over some").split(" "));
if (STOP2) for (const w of ["siehe", "each", "vgl"]) STOP.add(w);
const capTokens = new Set(); // folded tokens that appear Capitalized somewhere (proper-noun signal)
const cleanLine = (line) => line.replace(/\$\$[^$]*\$\$/g, " ").replace(/\$[^$]*\$/g, " ")
  .replace(/\\[a-zA-Z]+\{?/g, " ")
  .replace(/\[\[([^\]|]+)(\|[^\]]+)?\]\]/g, "$1").replace(/[*_~`#>]/g, "");
function detailTokens(body, title, lineDf) {
  const tokens = new Map(); // token -> example sentence
  const lines = (title + "\n" + body).split("\n");
  for (const line of lines) {
    if (/^\s*\|/.test(line)) continue; // tables are layout
    if (G1_SENTDF > 0 && (lineDf.get(fold(line)) ?? 0) >= G1_SENTDF) continue; // G1: boilerplate line
    const src = cleanLine(line);
    // record capitalized mid-sentence words (skip sentence-initial) as proper nouns
    const rawWords = src.split(/(\s+)/).filter((t) => /\S/.test(t));
    for (let k = 1; k < rawWords.length; k++) {
      const w = rawWords[k].replace(/^[^\p{L}]+|[^\p{L}]+$/gu, "");
      if (/^\p{Lu}[\p{Ll}]{2,}/u.test(w) && !/[.!?:]$/.test(rawWords[k - 1])) capTokens.add(fold(w));
    }
    const clean = fold(src);
    const words = clean.split(/[^a-z0-9]+/).filter(Boolean);
    for (let i = 0; i < words.length; i++) {
      // channel a: number + following content word
      if (NUMWORD.test(words[i]) && i + 1 < words.length && words[i + 1].length >= 4 && !STOP.has(words[i + 1])) {
        const tok = words[i] + " " + words[i + 1];
        if (!tokens.has(tok)) tokens.set(tok, line.trim().slice(0, 160));
      }
      // channel b: rare long content words
      if (words[i].length >= 6 && !STOP.has(words[i]) && !NUMWORD.test(words[i])) {
        if (!tokens.has(words[i])) tokens.set(words[i], line.trim().slice(0, 160));
      }
      // channel c: content-word bigrams — a shared PHRASE is a detail even
      // without a number ("vernarbte linke", "sealed lattice")
      if (i + 1 < words.length && words[i].length >= 4 && words[i + 1].length >= 4
        && !STOP.has(words[i]) && !STOP.has(words[i + 1])
        && !NUMWORD.test(words[i]) && !NUMWORD.test(words[i + 1])) {
        const big = words[i] + " " + words[i + 1];
        if (!tokens.has(big)) tokens.set(big, line.trim().slice(0, 160));
      }
    }
  }
  return tokens;
}

// ---------- G2: language-frequency lists + corpus term evidence ----------
// Suffix strip for German inflection; one suffix, longest first, stem >= 4.
const SUFFIXES = ["en", "er", "es", "em", "e", "n", "s"];
const stem = (w) => {
  for (const suf of SUFFIXES) if (w.length - suf.length >= 4 && w.endsWith(suf)) return w.slice(0, w.length - suf.length);
  return w;
};
function loadFreq(file) {
  const map = new Map();
  if (!existsSync(file)) return map;
  const lines = readFileSync(file, "utf8").split("\n");
  for (let i = 0; i < lines.length; i++) {
    const w = lines[i].split(" ")[0];
    if (!w) continue;
    const f = fold(w);
    if (!map.has(f)) map.set(f, i + 1);
  }
  return map;
}
// CP_LANGSRC=vault derives "language-common" from the VAULT'S OWN word document
// frequencies instead of a bundled frequency list. If it matches the bundled
// lists on the gate-level confusion readout, the plugin ships without ~200 KB of
// CC-BY-SA data, and it adapts to whatever languages the user actually writes in.
const LANG_SRC = process.env.CP_LANGSRC || "list";
let deFreq = new Map(), enFreq = new Map();
// Populated after the notes are read (it needs them); see initVaultFreq() below.
function initVaultFreq(allNotes) {
  const wordDfAll = new Map();
  for (const n of allNotes) {
    for (const w of new Set(fold(n.body).split(/[^a-z0-9]+/).filter((x) => x.length >= 2))) {
      wordDfAll.set(w, (wordDfAll.get(w) ?? 0) + 1);
    }
  }
  // Rank by descending document frequency, so position in this list plays the
  // same role as position in a published frequency list and CP_DE / CP_EN keep
  // their meaning.
  const ranked = [...wordDfAll.entries()].sort((a, b) => b[1] - a[1]);
  ranked.forEach(([w], i) => { deFreq.set(w, i + 1); enFreq.set(w, i + 1); });
  console.log(`G2 source: vault (${ranked.length} distinct words; commonest "${ranked[0]?.[0]}" in ${ranked[0]?.[1]} notes)`);
}
if ((G2_LANG || REF_RARE > 0) && LANG_SRC !== "vault") {
  deFreq = loadFreq(join(BENCH, "data", "de_50k.txt"));
  enFreq = loadFreq(join(BENCH, "data", "en_50k.txt"));
  if (!deFreq.size || !enFreq.size) { console.error("G2 enabled but bench/data/{de,en}_50k.txt missing"); process.exit(1); }
}
const rankIn = (map, w) => Math.min(map.get(w) ?? Infinity, map.get(stem(w)) ?? Infinity);
const isCommonWord = (w) => rankIn(deFreq, w) < DE_T || rankIn(enFreq, w) < EN_T;
// term evidence: consecutive stem-matched words inside a title/alias/heading/
// bold/link-display span; single-word tokens must match a FULL span.
const evidencePairs = new Set(); // "stemA stemB"
const evidenceFullSpans = new Set(); // full span as stems joined
function addEvidenceSpan(text) {
  const words = fold(cleanLine(text)).split(/[^a-z0-9]+/).filter((w) => w.length >= 3);
  if (!words.length) return;
  const stems = words.map(stem);
  evidenceFullSpans.add(stems.join(" "));
  for (let i = 0; i + 1 < stems.length; i++) evidencePairs.add(stems[i] + " " + stems[i + 1]);
}
const hasTermEvidence = (tok) => {
  const parts = tok.split(" ").map(stem);
  if (parts.length === 1) return parts[0].length >= 5 && evidenceFullSpans.has(parts[0]);
  if (!parts.some((p) => p.length >= 5) || parts.join("").length < 8) return false; // short stems over-rescue
  return evidencePairs.has(parts.join(" "));
};

// ---------- main ----------
const gt = JSON.parse(readFileSync(GT_PATH, "utf8"));
const manifest = JSON.parse(readFileSync("/Users/justus/obsidian_atomized_intermediary/lab/corpus-manifest.json", "utf8"));
const INDEXABLE_REAL = new Set(manifest.answer_paths.map((p) => "real/" + p));
const files = walk(VAULT).filter((abs) => {
  const rel = relative(VAULT, abs);
  return !rel.startsWith("real/") || INDEXABLE_REAL.has(rel);
});

// ---- startup check 1: every GT continuity anchor must resolve ----
{
  let bad = 0;
  for (const section of ["continuity_pairs", "continuity_negatives"]) {
    for (const e of gt[section] || []) {
      for (const side of ["x", "y"]) {
        const f = join(VAULT, e[`${side}_file`]);
        const anchor = e[`${side}_anchor`];
        if (!anchor) continue;
        if (!existsSync(f)) { console.error(`ANCHOR FAIL ${e.id}: missing file ${e[`${side}_file`]}`); bad++; continue; }
        if (!fold(readFileSync(f, "utf8")).includes(fold(anchor))) {
          console.error(`ANCHOR FAIL ${e.id}: "${anchor}" not in ${e[`${side}_file`]}`); bad++;
        }
      }
    }
  }
  if (bad) { console.error(`${bad} anchors unresolved — refusing to run`); process.exit(1); }
  console.log("anchors: all resolve");
}

// pass 1: read notes, collect raw bodies + line/sentence df tables (G1) + evidence spans (G2)
const pre = [];
for (const abs of files) {
  const rel = relative(VAULT, abs);
  const basename = rel.replace(/\.md$/, "").split("/").pop();
  if (/\bMOC\b/i.test(basename) || /(^|\/)Attachments\//.test(rel)) continue;
  const raw = readFileSync(abs, "utf8");
  const { tags, aliases, body } = parseFront(raw);
  pre.push({ rel, basename, tags, aliases, links: wikilinks(body), body });
}
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
const lineDf = new Map(), sentDf = new Map();
for (const n of pre) {
  const seenL = new Set();
  for (const line of n.body.split("\n")) { const f = fold(line); if (f.length >= 6) seenL.add(f); }
  for (const f of seenL) lineDf.set(f, (lineDf.get(f) ?? 0) + 1);
  n.allSents = sentencesOf(n.body);
  const seenS = new Set(n.allSents.map(fold));
  for (const f of seenS) sentDf.set(f, (sentDf.get(f) ?? 0) + 1);
}
if (G2_LANG) {
  const headingDf = new Map();
  for (const n of pre) for (const line of n.body.split("\n")) {
    const hm = line.match(/^\s{0,3}#{1,6}\s+(.+)/);
    if (hm) headingDf.set(fold(hm[1]), (headingDf.get(fold(hm[1])) ?? 0) + 1);
  }
  for (const n of pre) {
    addEvidenceSpan(n.basename);
    for (const a of n.aliases) addEvidenceSpan(a);
    for (const line of n.body.split("\n")) {
      const hm = line.match(/^\s{0,3}#{1,6}\s+(.+)/);
      if (hm && (headingDf.get(fold(hm[1])) ?? 0) < 3) addEvidenceSpan(hm[1]); // boilerplate headings are not terms
      for (const bm of line.matchAll(/\*\*([^*\n]{3,80})\*\*/g)) addEvidenceSpan(bm[1]);
      for (const lm of line.matchAll(/\[\[[^\]|]+\|([^\]]+)\]\]/g)) addEvidenceSpan(lm[1]);
    }
  }
}
const notes = [];
for (const n of pre) {
  n.tokens = detailTokens(n.body, n.basename, lineDf);
  notes.push(n);
}
if ((G2_LANG || REF_RARE > 0) && LANG_SRC === "vault") initVaultFreq(notes);
console.log(`indexed ${notes.length} notes; embedding means with ${MODEL} ...`);

// note means from sentence-level cache (same cache as aq-eval)
let semDropped = 0;
for (const n of notes) {
  let sents = n.allSents;
  if (G1_SENTDF > 0) {
    const kept = sents.filter((s) => (sentDf.get(fold(s)) ?? 0) < G1_SENTDF);
    semDropped += sents.length - kept.length;
    sents = kept;
  }
  n.sents = sents.slice(0, 60);
}
if (G1_SENTDF > 0) console.log(`G1: dropped ${semDropped} boilerplate sentences (df>=${G1_SENTDF})`);
const allTexts = [];
for (const n of notes) allTexts.push(...n.sents);
const extractor = await pipeline("feature-extraction", MODEL, { dtype: "fp32" });
const cachePath = process.env.AQ_CACHE || "/private/tmp/claude-501/-Users-justus-obsidian-atomized-intermediary/fdd9de71-217a-43e1-9ca7-fb239cfd5cb4/scratchpad/aq-embed-cache.json";
let cache = {};
try { cache = JSON.parse(readFileSync(cachePath, "utf8")); if (cache.__model !== MODEL) cache = {}; } catch { }
async function embedInto(texts) {
  const missing = [...new Set(texts.filter((t) => !cache[t]))];
  if (!missing.length) return;
  for (let i = 0; i < missing.length; i += 32) {
    const batch = missing.slice(i, i + 32);
    const t = await extractor(batch, { pooling: "mean", normalize: true });
    const vs = t.tolist();
    batch.forEach((txt, j) => { cache[txt] = vs[j]; });
  }
  cache.__model = MODEL;
  try { writeFileSync(cachePath, JSON.stringify(cache)); } catch { }
}
console.log(`embedding ${[...new Set(allTexts.filter((t) => !cache[t]))].length} new sentences (${allTexts.length} total) ...`);
await embedInto(allTexts);
let nulledMeans = 0;
for (const n of notes) {
  const vs = n.sents.map((s) => cache[s]).filter(Boolean);
  if (!vs.length) { n.mean = null; nulledMeans++; continue; }
  const d = vs[0].length, m = new Array(d).fill(0);
  for (const v of vs) for (let i = 0; i < d; i++) m[i] += v[i];
  n.mean = l2(m);
}
if (nulledMeans) console.log(`notes with no surviving sentences (null mean): ${nulledMeans}`);
// centering
const withMean = notes.filter((n) => n.mean);
const d = withMean[0].mean.length, c = new Array(d).fill(0);
for (const n of withMean) for (let i = 0; i < d; i++) c[i] += n.mean[i];
const cN = l2(c);
for (const n of withMean) { const p = dot(n.mean, cN); n.mean = l2(n.mean.map((x, i) => x - p * cN[i])); }

// ---------- token df + inverted index ----------
const df = new Map();
for (const n of notes) for (const tok of n.tokens.keys()) df.set(tok, (df.get(tok) ?? 0) + 1);
// word-level df: a non-number phrase only counts as a detail if one of its
// component words is itself uncommon ("vernarbte linke" yes, "weathered face"
// yes via weathered, "every surface" no).
const wordDf = new Map();
for (const n of notes) {
  const seen = new Set(fold(n.body).split(/[^a-z0-9]+/).filter((w) => w.length >= 4));
  for (const w of seen) wordDf.set(w, (wordDf.get(w) ?? 0) + 1);
}
const WORD_DF_MAX = Number(process.env.CP_WORDDF ?? 8);
const NUMTOK = (tok) => /^(\d|zwei|drei|vier|fuenf|sechs|sieben|acht|neun|zehn|elf|zwoelf|zwanzig|dreissig|vierzig|fuenfzig|hundert|tausend|siebzehn|two|three|four|five|six|seven|eight|nine|ten|twelve|twenty|forty|fifty|hundred|thousand|seventeen)/.test(tok);
// A number detail is only a detail if its NOUN is uncommon: "drei kupferstuecke"
// yes, "500 credits"/"drei phasen" no — generic units collide on different referents.
const UNIT_DF_MAX = Number(process.env.CP_UNITDF ?? 12);
const phraseOk = (tok) => {
  if (!tok.includes(" ")) return true;
  const [a, b] = tok.split(" ");
  if (NUMTOK(tok)) return (wordDf.get(b) ?? 0) <= UNIT_DF_MAX;
  return [a, b].some((w) => (wordDf.get(w) ?? 99) <= WORD_DF_MAX);
};
// G2: language-common with no term evidence -> not a detail. Number bigrams are
// G3's business (their first word is not in any frequency list anyway).
const g2Kills = (tok) => {
  if (!G2_LANG || NUMTOK(tok)) return false;
  const words = tok.split(" ");
  if (!words.every(isCommonWord)) return false;
  return !hasTermEvidence(tok);
};
const inv = new Map(); // token -> notes
for (const n of notes) for (const tok of n.tokens.keys()) {
  const f = df.get(tok);
  if (f < 2 || f > MAX_DF || !phraseOk(tok)) continue;
  if (NOSINGLE && !tok.includes(" ")) continue;
  if (g2Kills(tok)) continue;
  if (!inv.has(tok)) inv.set(tok, []);
  inv.get(tok).push(n);
}
const linked = (a, b) => a.links.has(fold(b.basename)) || b.links.has(fold(a.basename));
const sharesArea = (a, b) => {
  const top = (n) => n.rel.split("/")[0];
  if (top(a) === "real" && top(b) === "real") return true; // real corpus gates by tags below
  if (top(a) !== top(b)) return false;
  return true;
};
const cos = (a, b) => (a.mean && b.mean) ? dot(a.mean, b.mean) : 0;

// ---- startup check 2: df-invariance of the previously judged pairs' tokens ----
{
  const minedPath = "/Users/justus/obsidian_atomized_intermediary/lab/results/continuity-known-15.json";
  if (existsSync(minedPath)) {
    const known = JSON.parse(readFileSync(minedPath, "utf8"));
    const plantFiles = new Set();
    for (const section of ["continuity_pairs", "continuity_negatives"])
      for (const e of gt[section] || []) { plantFiles.add(e.x_file); plantFiles.add(e.y_file); }
    const plantNotes = notes.filter((n) => plantFiles.has(n.rel));
    let contaminated = 0;
    for (const p of known) for (const tok of p.tokens || []) {
      const hits = plantNotes.filter((n) => n.tokens.has(tok));
      if (hits.length) { console.error(`DF CONTAMINATION: known token "${tok}" appears in plant ${hits.map((h) => h.rel).join(", ")}`); contaminated++; }
      if ((df.get(tok) ?? 0) > MAX_DF) console.error(`DF SHIFT: known token "${tok}" now df=${df.get(tok)} > ${MAX_DF}`), contaminated++;
    }
    console.log(contaminated ? `df-invariance: ${contaminated} PROBLEMS` : "df-invariance: clean (plants do not touch known-pair tokens)");
    if (contaminated) process.exit(1);
  }
}

// ---------- score pairs (lexical channel) ----------
// A continuity find is ONE precise shared detail, not bulk vocabulary overlap.
// Score = best single token (number+noun bigrams boosted hard); pairs sharing
// many rare tokens are topical siblings — the similarity feature's job, not ours.
const BULK_MAX = Number(process.env.CP_BULK ?? 10);
const tokWeight = (tok) => {
  const isNum = /^\S*\d|\b(zwei|drei|vier|fuenf|sechs|sieben|acht|neun|zehn|hundert|tausend|two|three|four|five|ten|hundred|thousand)\b/.test(tok.split(" ")[0]) && tok.includes(" ");
  const w = isNum ? 5 : tok.includes(" ") ? 4 : 1; // number-detail > phrase > single rare word
  return w * Math.log(notes.length / df.get(tok));
};
const pairScores = new Map(); // key -> {a,b,tokens:[],score,cos}
for (const [tok, ns] of inv) {
  for (let i = 0; i < ns.length; i++) for (let j = i + 1; j < ns.length; j++) {
    const a = ns[i], b = ns[j];
    if (a === b || linked(a, b) || !sharesArea(a, b)) continue;
    const cc = cos(a, b);
    if (cc < MIN_COS || cc > DUPE_COS) continue; // unrelated or near-dupe
    const key = [a.rel, b.rel].sort().join(">>");
    if (!pairScores.has(key)) pairScores.set(key, { a, b, tokens: [], score: 0, cos: cc });
    const e = pairScores.get(key);
    e.tokens.push(tok);
  }
}

// ---- v2 per-pair token kills (G3 / G5 / G6), with a gate log for known pairs ----
const knownKeys = new Set();
{
  const minedPath = "/Users/justus/obsidian_atomized_intermediary/lab/results/continuity-known-15.json";
  if (existsSync(minedPath)) for (const p of JSON.parse(readFileSync(minedPath, "utf8"))) knownKeys.add(p.key);
}
const gateLog = [];
// G7 helper: a receipt line with <=3 content words (math spans count as one),
// or ending in ':' at <=5 words, is a list-header, not a detail-bearing sentence
function headerLike(snip) {
  if (!snip) return false;
  let t = snip.trim();
  const colon = t.endsWith(":");
  t = t.replace(/\$\$[^$]*\$\$/g, " MATHX ").replace(/\$[^$]*\$/g, " MATHX ")
    .replace(/\\[a-zA-Z]+\{?/g, " ")
    .replace(/\[\[([^\]|]+)(\|[^\]]+)?\]\]/g, "$1").replace(/[*_~`#>]/g, "");
  const words = t.split(/[^a-zA-Z0-9]+/).filter(Boolean);
  return words.length <= 3 || (colon && words.length <= 5);
}
// G3 helper: next content words after a number bigram, rescanned from the body
function nextWordsAfter(note, tok) {
  const [w1, w2] = tok.split(" ");
  const out = new Set();
  for (const line of (note.basename + "\n" + note.body).split("\n")) {
    if (/^\s*\|/.test(line)) continue;
    const words = fold(cleanLine(line)).split(/[^a-z0-9]+/).filter(Boolean);
    for (let i = 0; i + 1 < words.length; i++) {
      if (words[i] !== w1 || words[i + 1] !== w2) continue;
      let k = i + 2;
      while (k < words.length && (STOP.has(words[k]) || words[k].length < 3)) k++;
      if (k < words.length) out.add(stem(words[k]));
    }
  }
  return out;
}
for (const [key, e] of pairScores) {
  const killed = [];
  e.tokens = e.tokens.filter((tok) => {
    if (G3_TRIGRAM && NUMTOK(tok) && tok.includes(" ")) {
      const na = nextWordsAfter(e.a, tok), nb = nextWordsAfter(e.b, tok);
      if (na.size && nb.size && ![...na].some((w) => nb.has(w))) { killed.push([tok, "G3-trigram"]); return false; }
    }
    if (G5_COCITE) {
      const words = tok.split(" ");
      const inBoth = [...e.a.links].filter((t) => e.b.links.has(t));
      for (const t of inBoth) {
        const tw = t.split(/[^a-z0-9]+/).filter(Boolean);
        for (let i = 0; i + words.length <= tw.length; i++) {
          if (words.every((w, j) => tw[i + j] === w)) { killed.push([tok, "G5-cocite"]); return false; }
        }
      }
    }
    if (G6_EXACTLINE) {
      const la = e.a.tokens.get(tok), lb = e.b.tokens.get(tok);
      if (la && lb && la.length >= 30 && fold(la) === fold(lb)) { killed.push([tok, "G6-exactline"]); return false; }
    }
    if (G7_HEADKILL && (headerLike(e.a.tokens.get(tok)) || headerLike(e.b.tokens.get(tok)))) { killed.push([tok, "G7-header"]); return false; }
    return true;
  });
  for (const [tok, gate] of killed) if (knownKeys.has(key)) gateLog.push({ pair: key, token: tok, gate });
  if (!e.tokens.length) { pairScores.delete(key); continue; }
  e.score = Math.max(...e.tokens.map(tokWeight));
}
// G2 decisions for known pairs (tokens never admitted — reconstruct for the log)
if (G2_LANG && knownKeys.size) {
  const byKey = new Map([...pairScores].map(([k, e]) => [k, e]));
  const minedPath = "/Users/justus/obsidian_atomized_intermediary/lab/results/continuity-known-15.json";
  for (const p of JSON.parse(readFileSync(minedPath, "utf8"))) {
    for (const tok of p.tokens || []) {
      if (g2Kills(tok)) gateLog.push({ pair: p.key, token: tok, gate: "G2-lang" });
      else if (G2_LANG && !NUMTOK(tok) && tok.split(" ").every(isCommonWord)) gateLog.push({ pair: p.key, token: tok, gate: "G2-rescued-by-evidence" });
    }
    if (!byKey.has(p.key)) gateLog.push({ pair: p.key, token: "(pair)", gate: "pair-gone" });
  }
}

// Non-narrative pairs that already MENTION each other's title in prose are
// known adjacencies (concept siblings), not discoveries. A chapter mentioning a
// lore note's name unlinked is exactly the continuity case — keep those.
// PRE-REGISTERED: exact-match only, no suffix-strip here (see EXPERIMENT doc).
const isNarrative = (n) => /(^|\/)Szenen\//.test(n.rel) || /(^|\/)\d+\.\s[^/]+\.md$/.test(n.rel);
const mentionsTitle = (a, b) => fold(a.body).includes(fold(b.basename)) || fold(b.body).includes(fold(a.basename));
// A token is a proper detail if it (or a word in it) is a proper noun, or it is
// a number-detail. Generic sensory prose ("searing pain", "chime rang") is not.
const NAMED_ONLY = process.env.CP_NAMED !== "0";
const isNamedTok = (tok) => NUMTOK(tok) || tok.split(" ").some((w) => capTokens.has(w));
for (const e of pairScores.values()) {
  e.tokens.sort((x, y) => tokWeight(y) - tokWeight(x));
  if (e.tokens.length > BULK_MAX) e.bulk = true; // topical sibling, demote out of the list
  if (!isNarrative(e.a) && !isNarrative(e.b) && mentionsTitle(e.a, e.b)) e.bulk = true;
  // Narrative<>narrative pairs must share a NAMED detail, not just rare prose words.
  if (NAMED_ONLY && isNarrative(e.a) && isNarrative(e.b) && !e.tokens.some(isNamedTok)) e.bulk = true;
  // surface the named detail first in the display
  e.tokens.sort((x, y) => (isNamedTok(y) ? 1 : 0) - (isNamedTok(x) ? 1 : 0));
}
let ranked = [...pairScores.values()].filter((e) => !e.bulk)
  .sort((x, y) => (y.score + y.cos) - (x.score + x.cos));

// mining scope (declared early so the referent gate can cover in-scope pairs)
const SCOPE = process.env.CP_SCOPE || "all";
const TECH_RE = /^real\/(Concepts|ML|Mathematik|Informatik|Programmierung|Physik)\//;
const TECH_FILES = new Set(["real/Eigenwerte.md", "real/Neural Networks.md"]);
const isTechReal = (n) => TECH_RE.test(n.rel) || TECH_FILES.has(n.rel);
const inScope = (e) => SCOPE !== "technical"
  ? e.a.rel.startsWith("real/") && e.b.rel.startsWith("real/")
  : isTechReal(e.a) && isTechReal(e.b);

// Referent gate: the two snippet lines must describe the SAME thing — embed the
// snippets of the top candidates and require agreement.
const REF_MIN = Number(process.env.CP_REFMIN ?? 0.55);
{
  const headSet = new Set(ranked.slice(0, 150));
  if (REF_SCOPE) for (const e of ranked.slice(0, 800)) if (inScope(e)) headSet.add(e);
  const head = [...headSet];
  // self-certifying tokens: a word so rare in BOTH language lists that it IS the
  // referent (a name or coinage) needs no context agreement
  const selfCertifies = (tok) => REF_RARE > 0 && !NUMTOK(tok) && tok.split(" ").some((w) =>
    rankIn(deFreq, w) > REF_RARE && rankIn(enFreq, w) > REF_RARE);
  const refText = (n, tok) => REF_CLEAN ? fold(cleanLine(n.tokens.get(tok) ?? "")) : n.tokens.get(tok);
  const texts = [...new Set(head.flatMap((e) => [refText(e.a, e.tokens[0]), refText(e.b, e.tokens[0])].filter(Boolean)))];
  await embedInto(texts);
  for (const e of head) {
    const va = cache[refText(e.a, e.tokens[0])], vb = cache[refText(e.b, e.tokens[0])];
    e.refCos = va && vb ? dot(l2(va), l2(vb)) : null;
  }
  const bar = (e) => (!e.tokens[0].includes(" ") && REF_SINGLE > 0) ? Math.max(REF_MIN, REF_SINGLE) : REF_MIN;
  ranked = ranked.filter((e) => !headSet.has(e) || e.refCos === null || selfCertifies(e.tokens[0]) || e.refCos >= bar(e));
}

// ---------- semantic channel (cross-language continuity) ----------
// Semantic channel: best cross-note SENTENCE pair (the "because" snippet) over
// the most-similar unlinked note pairs — catches cross-language continuity that
// no lexical token can see.
let semRanked = [];
if (SEM_CHANNEL) {
  const SEM_NOTE_MIN = Number(process.env.CP_SEMNOTE ?? 0.25);
  const SEM_BEST_MIN = Number(process.env.CP_SEMBEST ?? 0.45);
  const lexKeys = new Set([...pairScores.keys()]);
  const cands = [];
  for (let i = 0; i < withMean.length; i++) for (let j = i + 1; j < withMean.length; j++) {
    const a = withMean[i], b = withMean[j];
    if (linked(a, b) || !sharesArea(a, b)) continue;
    const cc = dot(a.mean, b.mean);
    if (cc < SEM_NOTE_MIN || cc > DUPE_COS) continue;
    if (lexKeys.has([a.rel, b.rel].sort().join(">>"))) continue; // semantic-ONLY channel
    cands.push({ a, b, cos: cc });
  }
  cands.sort((x, y) => y.cos - x.cos);
  const centerV = (v) => { const p = dot(v, cN); return l2(v.map((x, i) => x - p * cN[i])); };
  for (const e of cands.slice(0, SEM_CAND)) {
    const fa = e.a.sents.filter((s) => s.length >= SEM_MINLEN);
    const fb = e.b.sents.filter((s) => s.length >= SEM_MINLEN);
    const va = fa.map((s) => cache[s]).filter(Boolean).map(l2).map(centerV);
    const vb = fb.map((s) => cache[s]).filter(Boolean).map(l2).map(centerV);
    let best = -1, bi = 0, bj = 0;
    for (let i = 0; i < va.length; i++) for (let j = 0; j < vb.length; j++) {
      if (SEM_EXACT_SKIP && fold(fa[i]) === fold(fb[j])) continue;
      const s = dot(va[i], vb[j]);
      if (s > best) { best = s; bi = i; bj = j; }
    }
    if (best >= SEM_BEST_MIN) semRanked.push({ a: e.a, b: e.b, cos: e.cos, best, asent: fa[bi], bsent: fb[bj] });
  }
  semRanked.sort((x, y) => y.best - x.best);
}

// ---- debug: why did a GT pair (not) form? ----
if (process.env.CP_DEBUG === "1") {
  const byRel = new Map(notes.map((n) => [n.rel, n]));
  console.log("\n==== GT pair diagnostics ====");
  for (const e of [...(gt.continuity_pairs || []), ...(gt.continuity_negatives || [])]) {
    const a = byRel.get(e.x_file), b = byRel.get(e.y_file);
    if (!a || !b) { console.log(`${e.id}: NOTE MISSING`); continue; }
    const key = [a.rel, b.rel].sort().join(">>");
    const pe = pairScores.get(key);
    const shared = [...a.tokens.keys()].filter((t) => b.tokens.has(t));
    const admissible = shared.filter((t) => { const f = df.get(t); return f >= 2 && f <= MAX_DF && phraseOk(t); });
    console.log(`${e.id}: cos=${cos(a, b).toFixed(3)} linked=${linked(a, b)} sharesArea=${sharesArea(a, b)} sharedToks=${shared.length} admissible=[${admissible.slice(0, 6).join("; ")}] inPairScores=${!!pe} bulk=${pe?.bulk ?? "-"} refCos=${pe?.refCos?.toFixed(3) ?? "-"}`);
  }
  console.log("==== known-15 rank diagnostics ====");
  const scoped = ranked.filter(inScope);
  for (const k of knownKeys) {
    const i = scoped.findIndex((e) => [e.a.rel, e.b.rel].sort().join(">>") === k);
    const e = i >= 0 ? scoped[i] : null;
    console.log(`  ${i >= 0 ? "#" + (i + 1) : "GONE"}  ${k.split(">>").map((p) => p.split("/").pop()).join(" <> ")}${e ? ` score=${e.score.toFixed(2)} cos=${e.cos.toFixed(2)} [${e.tokens.slice(0, 2).join(", ")}]` : ""}`);
  }
}

// ---------- metrics ----------
const findRank = (list, xf, yf) => {
  const want = [xf, yf].sort().join(">>");
  return list.findIndex((e) => [e.a.rel, e.b.rel].sort().join(">>") === want) + 1 || null;
};
const rows = [];
for (const cp of gt.continuity_pairs || []) {
  rows.push({ id: cp.id, channel: cp.channel || "lex", stretch: !!cp.stretch,
    lexRank: findRank(ranked, cp.x_file, cp.y_file), semRank: findRank(semRanked, cp.x_file, cp.y_file), shared: cp.shared });
}
const negRows = [];
for (const cn of gt.continuity_negatives || []) {
  const lr = findRank(ranked, cn.x_file, cn.y_file);
  const sr = findRank(semRanked, cn.x_file, cn.y_file);
  negRows.push({ id: cn.id, stretch: !!cn.stretch, surfacesTopK: (lr !== null && lr <= TOPK) || (sr !== null && sr <= 8), lexRank: lr, semRank: sr });
}
const dupeLeaks = (gt.near_dupes || []).map((nd) => ({ id: nd.id, lexRank: findRank(ranked, nd.x_file, nd.y_file) }))
  .filter((r) => r.lexRank !== null && r.lexRank <= TOPK);
const hub = gt.generic_bridge_trap?.file;
const hubCount = ranked.slice(0, TOPK).filter((e) => e.a.rel === hub || e.b.rel === hub).length;

// ---------- real-vault mining for judging ----------
const SUFFIX = SCOPE === "technical" ? "-technical" : "";
const MINE_N = Number(process.env.CP_MINE ?? 15);
const realPairs = ranked.filter(inScope).slice(0, MINE_N);
// token-centered receipt: window the raw line around the shared token so the
// detail is always visible (display invariant for judging)
function receipt(note, tok) {
  const rawLine = note.tokens.get(tok) ?? "";
  const words = tok.split(" ");
  // build a regex tolerant of umlauts/markup between and inside words
  const wordRe = (w) => w.split("").map((ch) => {
    if (ch === "a") return "(?:a|ä)"; if (ch === "o") return "(?:o|ö)"; if (ch === "u") return "(?:u|ü)";
    if (ch === "s") return "(?:s|ß)"; return ch;
  }).join("[*_`]*");
  const re = new RegExp(words.map(wordRe).join("[^a-z0-9äöüß]{1,6}"), "i");
  // prefer rescanning the full body for an untruncated line
  let bestLine = rawLine, m = null;
  for (const line of (note.basename + "\n" + note.body).split("\n")) {
    const mm = re.exec(line.replace(/ae/gi, "ä").replace(/oe/gi, "ö").replace(/ue/gi, "ü")) || re.exec(line);
    if (mm) { bestLine = line.trim(); m = re.exec(bestLine) || mm; break; }
  }
  if (!m) return { text: rawLine.slice(0, 200), tokenVisible: fold(rawLine).includes(tok) };
  const idx = Math.max(0, bestLine.toLowerCase().indexOf(m[0].toLowerCase()));
  const start = Math.max(0, idx - 90), end = Math.min(bestLine.length, idx + m[0].length + 90);
  const text = (start > 0 ? "…" : "") + bestLine.slice(start, end) + (end < bestLine.length ? "…" : "");
  return { text, tokenVisible: true };
}
const md = ["# Continuity pairs (real vault): same detail, never linked", ""];
for (const [i, e] of realPairs.entries()) {
  const ra = receipt(e.a, e.tokens[0]), rb = receipt(e.b, e.tokens[0]);
  md.push(`## ${i + 1}. score ${e.score.toFixed(2)} cos ${e.cos.toFixed(2)} — shared: ${e.tokens.slice(0, 4).join(", ")}`,
    `- \`${e.a.rel.replace(/^real\//, "")}\`: ${ra.text}`,
    `- \`${e.b.rel.replace(/^real\//, "")}\`: ${rb.text}`, "");
}
const semReal = semRanked.filter(inScope).slice(0, 8);
md.push("# Semantic-only channel (cross-language / paraphrase continuity)", "");
for (const [i, e] of semReal.entries()) {
  md.push(`## S${i + 1}. best-sentence ${e.best.toFixed(2)} (note cos ${e.cos.toFixed(2)})`,
    `- \`${e.a.rel.replace(/^real\//, "")}\`: ${e.asent}`,
    `- \`${e.b.rel.replace(/^real\//, "")}\`: ${e.bsent}`, "");
}
writeFileSync(`/Users/justus/obsidian_atomized_intermediary/lab/results/continuity-review${SUFFIX}.md`, md.join("\n"));
writeFileSync(`/Users/justus/obsidian_atomized_intermediary/lab/results/continuity-mined${SUFFIX}.json`, JSON.stringify(realPairs.map((e) => {
  const ra = receipt(e.a, e.tokens[0]), rb = receipt(e.b, e.tokens[0]);
  return { key: [e.a.rel, e.b.rel].sort().join(">>"), afile: e.a.rel, bfile: e.b.rel,
    tokens: e.tokens.slice(0, 6), asnip: ra.text, bsnip: rb.text, receiptOk: ra.tokenVisible && rb.tokenVisible,
    score: +e.score.toFixed(3), cos: +e.cos.toFixed(3), refCos: e.refCos == null ? null : +e.refCos.toFixed(3),
    novel: !knownKeys.has([e.a.rel, e.b.rel].sort().join(">>")) };
}), null, 2));

// yield + differentiation metrics
const inScopeAll = ranked.filter(inScope);
const yieldAt = (t) => inScopeAll.filter((e) => e.score + e.cos >= t).length;
let relOverlap = null;
{
  const scopeNotes = withMean.filter((n) => SCOPE === "technical" ? isTechReal(n) : n.rel.startsWith("real/"));
  let overlapped = 0, checked = 0;
  for (const e of realPairs) {
    if (!e.a.mean || !e.b.mean) continue;
    checked++;
    const rankIn10 = (x, y) => scopeNotes.filter((n) => n !== x && cos(x, n) > cos(x, y)).length < 10;
    if (rankIn10(e.a, e.b) || rankIn10(e.b, e.a)) overlapped++;
  }
  relOverlap = checked ? +(overlapped / checked).toFixed(2) : null;
}

const config = { G1_SENTDF, G2_LANG, DE_T, EN_T, G3_TRIGRAM, REF_SINGLE, REF_SCOPE, G5_COCITE, STOP2, G6_EXACTLINE, SEM_MINLEN, SEM_EXACT_SKIP, NOSINGLE, REF_MIN, SCOPE, SEM_CAND, REF_RARE, REF_CLEAN, G7_HEADKILL };
const summary = {
  model: MODEL, params: { MAX_DF, MIN_COS, DUPE_COS, TOPK }, config,
  pairs_scored: pairScores.size,
  planted: rows, negatives: negRows,
  hard_negative_leaks: negRows.filter((r) => !r.stretch && r.surfacesTopK).map((r) => r.id),
  stretch_negative_leaks: negRows.filter((r) => r.stretch && r.surfacesTopK).map((r) => r.id),
  dupe_leaks_topK: dupeLeaks, hub_in_topK: hubCount,
  known_pairs_present: [...knownKeys].filter((k) => realPairs.some((e) => [e.a.rel, e.b.rel].sort().join(">>") === k)).length,
  real_pairs_mined: realPairs.length,
  novel_mined: realPairs.filter((e) => !knownKeys.has([e.a.rel, e.b.rel].sort().join(">>"))).length,
  nulled_means: nulledMeans,
  yield: { "t18": yieldAt(18), "t20": yieldAt(20), "t23": yieldAt(23), inScopeTotal: inScopeAll.length },
  related_notes_top10_overlap: relOverlap,
  gate_log: gateLog,
};
console.log("\n==== planted continuity pairs ====");
for (const r of rows) console.log(`${r.id}${r.stretch ? " (stretch)" : ""}  [${r.channel}]  lex#${r.lexRank}  sem#${r.semRank}  (${r.shared})`);
console.log("==== negatives ====");
for (const r of negRows) console.log(`${r.id}${r.stretch ? " (stretch)" : ""}  surfacesTop${TOPK}=${r.surfacesTopK}  lex#${r.lexRank} sem#${r.semRank}`);
console.log(`dupe leaks in top${TOPK}: ${dupeLeaks.length}, hub-trap appearances: ${hubCount}`);
if (gateLog.length) { console.log("==== gate log (known pairs) ===="); for (const g of gateLog) console.log(`  ${g.gate}  ${g.token}  ${g.pair.split(">>").map((p) => p.split("/").pop()).join(" <> ")}`); }
console.log("\n==== top lexical pairs (all) ====");
for (const e of ranked.slice(0, 10)) console.log(`  ${(e.score + e.cos).toFixed(2)}  ${e.a.basename} <> ${e.b.basename}  [${e.tokens.slice(0, 3).join(", ")}]`);
console.log("\n==== top real pairs ====");
for (const e of realPairs.slice(0, 8)) console.log(`  ${(e.score + e.cos).toFixed(2)}  ${e.a.basename} <> ${e.b.basename}  [${e.tokens.slice(0, 3).join(", ")}]`);
console.log(`\nknown pairs still present: ${summary.known_pairs_present}/${knownKeys.size} | novel mined: ${summary.novel_mined}/${realPairs.length} | yield@20: ${summary.yield.t20} | rn-top10 overlap: ${relOverlap}`);
writeFileSync(OUT, JSON.stringify({ summary, top: ranked.slice(0, 30).map((e) => ({ a: e.a.rel, b: e.b.rel, tokens: e.tokens.slice(0, 5), score: +e.score.toFixed(2), cos: +e.cos.toFixed(2) })) }, null, 2));
console.log("wrote", OUT);
