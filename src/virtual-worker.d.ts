// Ambient type for the build-time virtual module carrying the bundled embed
// worker source text. esbuild.config.mjs resolves it (an inline-worker plugin
// bundles src/worker/embed-worker.ts and injects the result as a string);
// tsc only ever sees this declaration.
declare module "virtual:embed-worker" {
  const workerSource: string;
  export default workerSource;
}

// Same pattern for the reader engine bundle (node-llama-cpp + entry, ESM with
// top-level await): inlined as a string and written to disk at first enable.
declare module "virtual:reader-engine" {
  const engineSource: string;
  export default engineSource;
}

// The plugin stylesheet, inlined so a hand-updated install that replaces only
// main.js still gets every CSS change; injected at load in onload().
declare module "virtual:plugin-styles" {
  const css: string;
  export default css;
}
