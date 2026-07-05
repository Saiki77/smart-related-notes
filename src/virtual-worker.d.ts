// Ambient type for the build-time virtual module carrying the bundled embed
// worker source text. esbuild.config.mjs resolves it (an inline-worker plugin
// bundles src/worker/embed-worker.ts and injects the result as a string);
// tsc only ever sees this declaration.
declare module "virtual:embed-worker" {
  const workerSource: string;
  export default workerSource;
}
