// Reader spike smoke test: load a GGUF with node-llama-cpp, generate one gist,
// report timing + memory. Usage:
//   node bench/reader-smoke.mjs ~/.cache/srn-lab/models/Qwen3-1.7B-Q4_K_M.gguf
import { getLlama, LlamaChatSession } from "node-llama-cpp";
import path from "node:path";
import os from "node:os";

const modelPath = process.argv[2]
  ? process.argv[2].replace(/^~/, os.homedir())
  : path.join(os.homedir(), ".cache", "srn-lab", "models", "Qwen3-1.7B-Q4_K_M.gguf");

const rss = () => `${(process.memoryUsage().rss / 1e9).toFixed(2)} GB rss`;
const NOTE = `# Mean-Centering
Rohe Embedding-Aehnlichkeit hat einen Anisotropie-Rauschboden: jede Notiz teilt
eine gemeinsame Richtung, daher scoren voellig fremde Notizen ~40-50%. Abhilfe:
den Korpus-Zentroid von allen Vektoren abziehen und renormalisieren. Nach der
Zentrierung faellt die mittlere Anisotropie von 0.494 auf -0.037, unrelated
Notizen gehen gegen 0.`;

console.log(`[smoke] loading ${path.basename(modelPath)} ... (${rss()})`);
let t = Date.now();
const llama = await getLlama();
const model = await llama.loadModel({ modelPath });
console.log(`[smoke] model loaded in ${Date.now() - t}ms (${rss()}) gpu=${llama.gpu}`);

t = Date.now();
const context = await model.createContext({ contextSize: 4096 });
const session = new LlamaChatSession({ contextSequence: context.getSequence() });
console.log(`[smoke] context ready in ${Date.now() - t}ms (${rss()})`);

t = Date.now();
const out = await session.prompt(
  `/no_think Schreibe genau EINE Zeile (max 10 Woerter), die diese Notiz zusammenfasst, in der Sprache der Notiz. Nur die Zeile, nichts sonst.\n\n${NOTE}`,
  { maxTokens: 64 },
);
const ms = Date.now() - t;
const text = out.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
console.log(`[smoke] one-liner in ${ms}ms: "${text}"`);

t = Date.now();
await context.dispose();
await model.dispose();
console.log(`[smoke] disposed in ${Date.now() - t}ms (${rss()})`);
console.log("[smoke] OK");
