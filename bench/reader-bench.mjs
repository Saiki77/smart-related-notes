// Reader spike perf bench: load time, per-call wall clock, throughput, RSS,
// dispose recovery. Usage:
//   node bench/reader-bench.mjs --model=8B [--cpu] [--n=6]
import { loadReader, resolveModel, realNotes, readNote, noteContext, stableSample, writeResult, rssGb } from "./reader-lib.mjs";

const arg = (k, d) => (process.argv.find((a) => a.startsWith(`--${k}=`)) ?? `=${d}`).split("=")[1];
const has = (k) => process.argv.includes(`--${k}`);
const modelPath = resolveModel(arg("model", "1.7B"));
const N = Number(arg("n", "6"));

const rss0 = rssGb();
const reader = await loadReader(modelPath, { gpu: has("cpu") ? false : "auto" });
console.log(`[bench] ${reader.name} gpu=${reader.gpu} load=${reader.loadMs}ms rss ${rss0.toFixed(2)} -> ${rssGb().toFixed(2)} GB`);

const notes = stableSample(realNotes().map(readNote).filter((n) => n.body.length > 400).map((n) => n.path), N, "bench").map(readNote);
const calls = [];
let peak = rssGb();
for (const n of notes) {
  const ctx = noteContext(n, 3200);
  const r = await reader.generate(
    `Summarize this note in 1-2 sentences, in the note's language. Output only the summary.\n\n"""${ctx}"""`,
    { maxTokens: 96 },
  );
  peak = Math.max(peak, rssGb());
  calls.push({ note: n.title, chars: ctx.length, inTokens: r.inTokens, outTokens: r.outTokens, ms: r.ms });
  console.log(`  ${r.ms}ms  in=${r.inTokens} out=${r.outTokens}  (${n.title.slice(0, 40)})`);
}
const tot = (k) => calls.reduce((s, c) => s + c[k], 0);
const summary = {
  model: reader.name,
  gpu: reader.gpu,
  loadMs: reader.loadMs,
  calls: calls.length,
  mean_ms: Math.round(tot("ms") / calls.length),
  // Lower bound: includes prefill. The honest per-note planning number.
  toks_per_s_incl_prefill: +(tot("outTokens") / (tot("ms") / 1000)).toFixed(1),
  prompt_toks_per_s: +(tot("inTokens") / (tot("ms") / 1000)).toFixed(0),
  rss_peak_gb: +peak.toFixed(2),
};
await reader.dispose();
await new Promise((r) => setTimeout(r, 300));
summary.rss_after_dispose_gb = +rssGb().toFixed(2);
console.log(summary);
console.log("written:", writeResult(`reader-bench-${summary.model.replace(/[^\w.-]/g, "_")}${has("cpu") ? "-cpu" : ""}.json`, { summary, calls }));
