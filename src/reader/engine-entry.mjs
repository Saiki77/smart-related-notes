// Entry for the reader engine bundle (built by esbuild into reader-bundle.mjs,
// platform=node, format=esm). Runs INSIDE the renderer via dynamic import(),
// but resolves llama.cpp binaries relative to its own location on disk:
// <assets>/dist/reader-bundle.mjs -> <assets>/{llama,bins}. Keep this file
// plain JS: it must not drag renderer-only imports into the node bundle.
import { getLlama, LlamaChatSession } from "node-llama-cpp";

export async function createReaderEngine(modelPath, options = {}) {
  const { contextSize = 4096, noThink = true } = options;
  const llama = await getLlama({ build: "never" });
  const model = await llama.loadModel({ modelPath });
  let context = await model.createContext({ contextSize, sequences: 1 });
  let disposed = false;

  // One task = one fresh session on a fresh sequence: full context reset per
  // call, no chat state. If sequence slots ever fail to recycle, the context
  // is rebuilt once and the call retried.
  async function generate(prompt, { maxTokens = 96 } = {}) {
    if (disposed) throw new Error("reader engine disposed");
    const full = (noThink ? "/no_think " : "") + prompt;
    const run = async () => {
      const seq = context.getSequence();
      try {
        const session = new LlamaChatSession({ contextSequence: seq });
        const out = await session.prompt(full, { maxTokens });
        session.dispose();
        return out;
      } finally {
        try { seq.dispose(); } catch { /* context rebuild below covers it */ }
      }
    };
    let out;
    try {
      out = await run();
    } catch {
      await context.dispose();
      context = await model.createContext({ contextSize, sequences: 1 });
      out = await run();
    }
    return out.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
  }

  async function dispose() {
    if (disposed) return;
    disposed = true;
    await context.dispose();
    await model.dispose();
  }

  return { generate, dispose, gpu: String(llama.gpu) };
}
