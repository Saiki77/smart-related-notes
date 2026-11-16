// Resolver hooks that let node import the plugin's real TypeScript sources:
//   - bare `obsidian`        -> bench/obsidian-stub.mjs
//   - `virtual:embed-worker`, `virtual:reader-engine`, `virtual:plugin-styles`
//                            -> an inert string (never started or written here)
//   - `./foo` with no extension -> `./foo.ts` (esbuild resolves these; node does not)
import { existsSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";

const STUB = new URL("./obsidian-stub.mjs", import.meta.url).href;
const WORKER = new URL("./worker-stub.mjs", import.meta.url).href;

export async function resolve(specifier, context, next) {
  if (specifier === "obsidian") return { url: STUB, shortCircuit: true };
  if (specifier === "virtual:embed-worker" || specifier === "virtual:reader-engine" || specifier === "virtual:plugin-styles") {
    return { url: WORKER, shortCircuit: true };
  }
  if (specifier.startsWith(".") && !/\.[a-z]+$/i.test(specifier) && context.parentURL) {
    const abs = fileURLToPath(new URL(specifier, context.parentURL));
    for (const ext of [".ts", ".mjs", ".js"]) {
      if (existsSync(abs + ext)) {
        return { url: pathToFileURL(abs + ext).href, shortCircuit: true };
      }
    }
  }
  return next(specifier, context);
}
