// Reader spike quality evals. Usage:
//   node bench/reader-eval.mjs --model=8B --task=judge|tags|gists [--n=60] [--cpu]
// Results land in lab/results/reader-<task>-<model>.json
import {
  loadReader, resolveModel, realNotes, readNote, langOf, stableSample,
  noteContext, anchorSentence, groundTruth, writeResult, rssGb,
} from "./reader-lib.mjs";

const arg = (k, d) => (process.argv.find((a) => a.startsWith(`--${k}=`)) ?? `=${d}`).split("=")[1];
const has = (k) => process.argv.includes(`--${k}`);
const modelPath = resolveModel(arg("model", "1.7B"));
const task = arg("task", "judge");
const N = Number(arg("n", "60"));
const tag = `${task}-${modelPath.match(/(\d+(?:\.\d+)?B)/i)?.[1] ?? "x"}${has("cpu") ? "-cpu" : ""}`;

const reader = await loadReader(modelPath, { gpu: has("cpu") ? false : "auto" });
console.log(`[eval] ${task} on ${reader.name} (gpu=${reader.gpu}, load ${reader.loadMs}ms)`);

// ---------------------------------------------------------------- judge ----
// Does B ANSWER question A, RESTATE it, or is it UNRELATED? The exact
// distinction that killed Answered Questions under cosine.
if (task === "judge") {
  const gt = groundTruth();
  const cases = [];
  for (const qa of gt.qa_pairs) {
    const q = qa.q_anchor;
    cases.push({ id: `${qa.id}+`, q, text: anchorSentence(qa.a_file, qa.a_anchor), want: "ANSWERS" });
    for (const other of stableSample(gt.qa_pairs.filter((o) => o.id !== qa.id), 6, qa.id)) {
      cases.push({ id: `${qa.id}-vs-${other.id}`, q, text: anchorSentence(other.a_file, other.a_anchor), want: "NEITHER" });
    }
  }
  for (const t of gt.paraphrase_traps) {
    const qa = gt.qa_pairs.find((x) => x.id === t.of);
    if (!qa) continue;
    cases.push({ id: t.id, q: qa.q_anchor, text: anchorSentence(t.file, t.anchor), want: "RESTATES" });
  }
  const rows = [];
  for (const c of cases) {
    const prompt = [
      `A note asks: "${c.q}"`,
      `Another note contains this passage:`,
      `"""${c.text}"""`,
      ``,
      `Does the passage directly ANSWER this specific question (explains or resolves exactly what it asks),`,
      `merely RESTATE the question (asks or wonders the same thing again, in any wording or language),`,
      `or NEITHER (anything else, including passages about the same topic that do not resolve this exact question)?`,
      `Reply with exactly one word: ANSWERS, RESTATES, or NEITHER.`,
    ].join("\n");
    const r = await reader.generate(prompt, { maxTokens: 8 });
    const got = (r.text.toUpperCase().match(/ANSWERS|RESTATES|NEITHER/) ?? ["?"])[0];
    rows.push({ id: c.id, want: c.want, got, ok: got === c.want, ms: r.ms });
    if (got !== c.want) console.log(`  MISS ${c.id}: want ${c.want} got ${got}`);
  }
  const by = (w) => rows.filter((r) => r.want === w);
  const acc = (rs) => (rs.length ? rs.filter((r) => r.ok).length / rs.length : NaN);
  const summary = {
    model: reader.name, cases: rows.length,
    answers_recall: acc(by("ANSWERS")),
    paraphrase_rejected: acc(by("RESTATES")),
    unrelated_rejected: acc(by("NEITHER")),
    mean_ms: Math.round(rows.reduce((s, r) => s + r.ms, 0) / rows.length),
  };
  console.log(summary);
  console.log("written:", writeResult(`reader-${tag}.json`, { summary, rows }));
}

// ----------------------------------------------------------------- tags ----
// Closed-set pick: hide a note's real tags, offer them among plausible
// distractors, reader picks 0-3. Validator drops anything not offered.
if (task === "tags") {
  const notes = realNotes().map(readNote).filter((n) => n.tags.length > 0 && n.body.length > 200);
  const df = new Map();
  for (const n of notes) for (const t of new Set(n.tags)) df.set(t, (df.get(t) ?? 0) + 1);
  const discriminative = [...df.entries()].filter(([, c]) => c >= 3 && c <= notes.length * 0.5).map(([t]) => t);
  const byFolder = new Map();
  for (const n of notes) {
    const top = n.rel.split("/")[1] ?? "";
    if (!byFolder.has(top)) byFolder.set(top, new Set());
    for (const t of n.tags) if (discriminative.includes(t)) byFolder.get(top).add(t);
  }
  const sample = stableSample(notes.map((n) => n.path), N, "tags").map(readNote);
  const rows = [];
  for (const n of sample) {
    const hidden = n.tags.filter((t) => discriminative.includes(t));
    if (hidden.length === 0) continue;
    const top = n.rel.split("/")[1] ?? "";
    const domain = [...(byFolder.get(top) ?? [])].filter((t) => !hidden.includes(t));
    const distractors = [
      ...stableSample(domain, 6, n.rel),
      ...stableSample(discriminative.filter((t) => !hidden.includes(t) && !domain.includes(t)), 6, n.rel),
    ].slice(0, 12 - Math.min(hidden.length, 3));
    const offered = stableSample([...new Set([...hidden, ...distractors])], 99, "shuffle" + n.rel);
    const prompt = [
      `Read this note:`, `"""${noteContext(n, 2600)}"""`, ``,
      `Which of these tags fit the note? Pick 0 to 3, ONLY from this list, best first:`,
      offered.map((t) => `#${t}`).join(" "),
      `Reply with just the chosen tags (or NONE), no other text.`,
    ].join("\n");
    const r = await reader.generate(prompt, { maxTokens: 32 });
    const picked = [...r.text.matchAll(/#?([a-z0-9äöüß/_-]+)/gi)].map((m) => m[1].toLowerCase()).filter((t) => offered.includes(t)).slice(0, 3);
    const invented = /[a-z]/i.test(r.text) && picked.length === 0 && !/none/i.test(r.text);
    rows.push({
      note: n.rel, hidden, picked, ms: r.ms,
      top1: picked.length > 0 ? hidden.includes(picked[0]) : null,
      anyHit: picked.some((t) => hidden.includes(t)),
      invented,
    });
  }
  const withPick = rows.filter((r) => r.top1 !== null);
  const summary = {
    model: reader.name, notes: rows.length,
    top1_precision: withPick.filter((r) => r.top1).length / Math.max(1, withPick.length),
    any_overlap: rows.filter((r) => r.anyHit).length / Math.max(1, rows.length),
    abstained: rows.length - withPick.length,
    invented_rate: rows.filter((r) => r.invented).length / Math.max(1, rows.length),
    mean_ms: Math.round(rows.reduce((s, r) => s + r.ms, 0) / Math.max(1, rows.length)),
  };
  console.log(summary);
  console.log("written:", writeResult(`reader-${tag}.json`, { summary, rows }));
}

// ---------------------------------------------------------------- gists ----
// One-liner (chat-title style) + gist per note, for blind judging.
if (task === "gists") {
  const notes = realNotes().map(readNote).filter((n) => n.body.length > 300);
  const de = stableSample(notes.filter((n) => langOf(n.body) === "de").map((n) => n.path), Math.ceil(N / 2), "g-de").map(readNote);
  const en = stableSample(notes.filter((n) => langOf(n.body) === "en").map((n) => n.path), Math.floor(N / 2), "g-en").map(readNote);
  const rows = [];
  for (const n of [...de, ...en]) {
    const lang = langOf(n.body) === "de" ? "German" : "English";
    const ctx = noteContext(n, 3200);
    const one = await reader.generate(
      `Write ONE line of at most 10 words that captures what this note is about, in ${lang}. Output only that line, no quotes.\n\n"""${ctx}"""`,
      { maxTokens: 40 },
    );
    const gist = await reader.generate(
      `Summarize this note in 1-2 sentences, in ${lang}. Be concrete and faithful; no meta-phrases like "this note describes". Output only the summary.\n\n"""${ctx}"""`,
      { maxTokens: 96 },
    );
    rows.push({ note: n.rel, lang, oneLiner: one.text.split("\n")[0], gist: gist.text, ms: one.ms + gist.ms });
    if (rows.length % 10 === 0) console.log(`  ${rows.length} notes read (rss ${rssGb().toFixed(1)} GB)`);
  }
  const summary = {
    model: reader.name, notes: rows.length,
    mean_ms_per_note: Math.round(rows.reduce((s, r) => s + r.ms, 0) / Math.max(1, rows.length)),
    over_10_words: rows.filter((r) => r.oneLiner.split(/\s+/).length > 12).length,
  };
  console.log(summary);
  console.log("written:", writeResult(`reader-${tag}.json`, { summary, rows }));
}

await reader.dispose();
