// Download a GGUF into the durable lab cache. Usage:
//   node bench/reader-pull.mjs hf:Qwen/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q4_K_M.gguf
import { resolveModelFile } from "node-llama-cpp";
import os from "node:os";
import path from "node:path";

const uri = process.argv[2];
if (!uri) { console.error("usage: reader-pull.mjs <model uri>"); process.exit(1); }
const dir = path.join(os.homedir(), ".cache", "srn-lab", "models");
const t0 = Date.now();
const file = await resolveModelFile(uri, { directory: dir, verify: false });
console.log(`pulled ${file} in ${((Date.now() - t0) / 1000).toFixed(0)}s`);
