import { getLlama, LlamaChatSession } from "node-llama-cpp";
export async function readerLoadTest(modelPath) {
  const t0 = Date.now();
  const llama = await getLlama();
  const model = await llama.loadModel({ modelPath });
  const context = await model.createContext({ contextSize: 2048 });
  const session = new LlamaChatSession({ contextSequence: context.getSequence() });
  const out = await session.prompt("/no_think Reply with exactly: reader-engine-ok", { maxTokens: 16 });
  const ms = Date.now() - t0;
  await context.dispose(); await model.dispose();
  return { ok: /reader-engine-ok/i.test(out), out: out.replace(/<think>[\s\S]*?<\/think>/g, "").trim(), ms, gpu: llama.gpu };
}
if (process.argv[2]) readerLoadTest(process.argv[2]).then((r) => { console.log("[load-test]", JSON.stringify(r)); process.exit(r.ok ? 0 : 1); });
