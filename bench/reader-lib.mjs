// Shared plumbing for the reader spike: stateless calls on a shared model,
// strict output handling, corpus + ground-truth utilities.
import { getLlama, LlamaChatSession } from "node-llama-cpp";
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import crypto from "node:crypto";

export const MODELS_DIR = path.join(os.homedir(), ".cache", "srn-lab", "models");
export const LAB = path.resolve(import.meta.dirname, "../../lab");

export function resolveModel(nameOrPath) {
  if (fs.existsSync(nameOrPath)) return nameOrPath;
  const hit = fs.readdirSync(MODELS_DIR).find((f) => f.toLowerCase().includes(nameOrPath.toLowerCase()) && f.endsWith(".gguf"));
  if (!hit) throw new Error(`no model matching "${nameOrPath}" in ${MODELS_DIR}`);
  return path.join(MODELS_DIR, hit);
}

// One loaded model; every call gets a FRESH session on a fresh sequence (the
// plugin's context-reset-per-task design, literally). Falls back to
// recreating the context if sequence slots don't recycle.
export async function loadReader(modelPath, { gpu = "auto", contextSize = 4096 } = {}) {
  const llama = await getLlama({ gpu });
  const t0 = Date.now();
  const model = await llama.loadModel({ modelPath });
  const loadMs = Date.now() - t0;
  const noThink = !/instruct-2507/i.test(modelPath);
  let context = await model.createContext({ contextSize, sequences: 1 });

  async function generate(prompt, { maxTokens = 96 } = {}) {
    const full = (noThink ? "/no_think " : "") + prompt;
    const run = async () => {
      const seq = context.getSequence();
      try {
        const session = new LlamaChatSession({ contextSequence: seq });
        const t = Date.now();
        const out = await session.prompt(full, { maxTokens });
        const ms = Date.now() - t;
        session.dispose();
        return { out, ms };
      } finally {
        try { seq.dispose(); } catch { /* recycled below */ }
      }
    };
    let r;
    try {
      r = await run();
    } catch {
      await context.dispose();
      context = await model.createContext({ contextSize, sequences: 1 });
      r = await run();
    }
    const text = r.out.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
    const inTokens = model.tokenize(full).length;
    const outTokens = model.tokenize(r.out).length;
    return { text, ms: r.ms, inTokens, outTokens };
  }

  async function dispose() {
    await context.dispose();
    await model.dispose();
  }
  return { generate, dispose, loadMs, gpu: llama.gpu, name: path.basename(modelPath) };
}

export const rssGb = () => process.memoryUsage.rss() / 1e9;

// --- corpus utilities ------------------------------------------------------

export function realNotes() {
  const out = [];
  const walk = (d) => {
    for (const e of fs.readdirSync(d, { withFileTypes: true })) {
      const p = path.join(d, e.name);
      if (e.isDirectory()) walk(p);
      else if (e.name.endsWith(".md")) out.push(p);
    }
  };
  walk(path.join(LAB, "vault", "real"));
  return out.sort();
}

export function readNote(p) {
  const raw = fs.readFileSync(p, "utf8");
  const m = raw.match(/^---\n([\s\S]*?)\n---\n?/);
  const fm = m ? m[1] : "";
  const body = m ? raw.slice(m[0].length) : raw;
  const tags = [];
  const tm = fm.match(/^tags:\s*\n((?:\s*-\s*.+\n?)+)/m) || fm.match(/^tags:\s*\[(.*?)\]/m);
  if (tm) {
    const items = tm[1].includes("-") ? tm[1].split("\n").map((l) => l.replace(/^\s*-\s*/, "").trim()) : tm[1].split(",");
    for (const t of items) { const c = t.replace(/["'#\s]/g, ""); if (c) tags.push(c.toLowerCase()); }
  }
  return { path: p, rel: path.relative(path.join(LAB, "vault"), p), title: path.basename(p, ".md"), fm, body, tags };
}

// Cheap DE/EN detection, same spirit as assignIdeas' stopword lists.
const DE = new Set("der die das und ist nicht mit ein eine von für auf dem den als auch wird sind oder wenn aber durch nur kann wie bei nach über man".split(" "));
const EN = new Set("the and is not with of for on as also or if but through only can how at after about one this that are was be it".split(" "));
export function langOf(text) {
  let de = 0, en = 0;
  for (const w of text.toLowerCase().split(/[^a-zäöüß]+/).slice(0, 400)) {
    if (DE.has(w)) de++;
    if (EN.has(w)) en++;
  }
  return de >= en ? "de" : "en";
}

// Deterministic sampling: stable hash order, no Math.random.
export function stableSample(arr, n, salt = "") {
  return [...arr]
    .sort((a, b) => md5(salt + str(a)).localeCompare(md5(salt + str(b))))
    .slice(0, n);
}
const str = (x) => (typeof x === "string" ? x : JSON.stringify(x));
const md5 = (s) => crypto.createHash("md5").update(s).digest("hex");

// Budgeted context: title + first chunks of the body, hard char cap.
export function noteContext(note, cap = 3500) {
  const body = note.body
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/%%[\s\S]*?%%/g, " ")
    .replace(/\n{3,}/g, "\n\n");
  return `# ${note.title}\n${body}`.slice(0, cap);
}

// Umlaut-folded anchor search -> surrounding window (for ground-truth plants).
const fold = (s) => s.toLowerCase().replace(/ä/g, "ae").replace(/ö/g, "oe").replace(/ü/g, "ue").replace(/ß/g, "ss").replace(/\s+/g, " ");
export function anchorWindow(file, anchor, win = 420) {
  const raw = fs.readFileSync(path.join(LAB, "vault", file), "utf8");
  const idx = fold(raw).indexOf(fold(anchor));
  if (idx < 0) throw new Error(`anchor not found in ${file}: ${anchor.slice(0, 40)}`);
  // fold() preserves length except umlauts (+1 each); good enough to center a window.
  const start = Math.max(0, idx - Math.floor(win / 3));
  return raw.slice(start, start + win).replace(/\s+/g, " ").trim();
}

// Sentence-bounded variant: the anchor's sentence plus the one before it,
// never spilling FORWARD past the anchor's sentence end. Traps plant a
// restated QUESTION; a forward window can swallow the writer's own answer
// attempt right after it and defeat the trap (measured: pt3).
export function anchorSentence(file, anchor) {
  const raw = fs.readFileSync(path.join(LAB, "vault", file), "utf8").replace(/\s+/g, " ");
  const idx = fold(raw).indexOf(fold(anchor));
  if (idx < 0) throw new Error(`anchor not found in ${file}: ${anchor.slice(0, 40)}`);
  const ends = [...raw.matchAll(/[.!?](?=\s|$)/g)].map((m) => m.index + 1);
  const sentEnd = ends.find((e) => e >= idx + anchor.length - 5) ?? raw.length;
  const before = ends.filter((e) => e <= idx);
  const sentStart = before.length >= 2 ? before[before.length - 2] : 0;
  return raw.slice(sentStart, sentEnd).trim().slice(0, 420);
}

export function groundTruth() {
  return JSON.parse(fs.readFileSync(path.join(LAB, "ground-truth.json"), "utf8"));
}

export function writeResult(name, data) {
  const p = path.join(LAB, "results", name);
  fs.writeFileSync(p, typeof data === "string" ? data : JSON.stringify(data, null, 2));
  return p;
}
