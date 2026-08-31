// Reader task registry: every reader use is a declared, budgeted, validated
// task. Context builders receive only what the index already distilled; every
// output passes a validator or is dropped — nothing unvalidated ever renders.

export interface NoteContextInput {
  title: string;
  body: string; // raw markdown body (frontmatter stripped by the caller)
  language: "de" | "en";
}

const langName = (l: "de" | "en"): string => (l === "de" ? "German" : "English");

// Shared input shaping: strip code/comment blocks, cap hard. The reader never
// sees a raw whole note.
export function budgetedContext(n: NoteContextInput, cap = 3200): string {
  const body = n.body
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/%%[\s\S]*?%%/g, " ")
    .replace(/\n{3,}/g, "\n\n");
  return `# ${n.title}\n${body}`.slice(0, cap);
}

// Cheap DE/EN pick, mirroring the idea-unit stopword approach.
const DE = new Set("der die das und ist nicht mit ein eine von für auf dem den als auch wird sind oder wenn aber durch nur kann wie bei nach über man".split(" "));
const EN = new Set("the and is not with of for on as also or if but through only can how at after about one this that are was be it".split(" "));
export function detectLanguage(text: string): "de" | "en" {
  let de = 0;
  let en = 0;
  for (const w of text.toLowerCase().split(/[^a-zäöüß]+/).slice(0, 400)) {
    if (DE.has(w)) de++;
    if (EN.has(w)) en++;
  }
  return de >= en ? "de" : "en";
}

export interface TaskResult {
  prompt: string;
  maxTokens: number;
  validate(raw: string): string | string[] | null;
}

// Chat-title-style one-liner: <= 10 words, note's language, single line.
export function oneLinerTask(n: NoteContextInput): TaskResult {
  return {
    prompt: `Write ONE line of at most 10 words that captures what this note is about, in ${langName(n.language)}. Output only that line, no quotes.\n\n"""${budgetedContext(n)}"""`,
    maxTokens: 40,
    validate(raw) {
      const line = raw.split("\n")[0].trim().replace(/^["'«»]+|["'«»]+$/g, "");
      if (line.length < 4 || line.length > 140) return null;
      if (line.split(/\s+/).length > 14) return null;
      if (/^(this note|diese notiz|the note)/i.test(line)) return null;
      return line;
    },
  };
}

// 1-2 sentence gist, shown on hover and read by every downstream task.
export function gistTask(n: NoteContextInput): TaskResult {
  return {
    prompt: `Summarize this note in 1-2 sentences, in ${langName(n.language)}. Be concrete and faithful; no meta-phrases like "this note describes". Output only the summary.\n\n"""${budgetedContext(n)}"""`,
    maxTokens: 96,
    validate(raw) {
      const text = raw.trim();
      if (text.length < 12 || text.length > 480) return null;
      if (/^(this note|diese notiz)/i.test(text)) return null;
      return text;
    },
  };
}

// Closed-set tag pick: 0-3 tags, ONLY from the offered candidates; anything
// else is dropped by the validator, so hallucinated tags cannot exist.
export function pickTagsTask(n: NoteContextInput, candidates: string[]): TaskResult {
  return {
    prompt: `Read this note:\n"""${budgetedContext(n, 2600)}"""\n\nWhich of these tags fit the note? Pick 0 to 3, ONLY from this list, best first:\n${candidates.map((t) => `#${t}`).join(" ")}\nReply with just the chosen tags (or NONE), no other text.`,
    maxTokens: 32,
    validate(raw) {
      const offered = new Set(candidates);
      const picked: string[] = [];
      for (const m of raw.matchAll(/#?([\p{L}\p{N}/_-]+)/gu)) {
        const t = m[1].toLowerCase();
        if (offered.has(t) && !picked.includes(t)) picked.push(t);
        if (picked.length === 3) break;
      }
      return picked;
    },
  };
}
