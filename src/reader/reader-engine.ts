// Native reader backend: loads the engine bundle and manages the model
// lifecycle (load on demand, unload after idle). The bundle is ESM with
// top-level await, so it must go through a real dynamic import() (esbuild's
// supported["dynamic-import"] keeps it untransformed in the CJS build), and
// Obsidian's renderer refuses import() of file:// URLs, so the staged file
// is read from disk and imported as a blob: module. Two globals bridge the
// gap: __srnReaderEngineUrl carries the real on-disk URL (the bundle's
// import.meta.url is compiled to it, so llama.cpp finds its data and
// binaries beside the staged file), and __srnRequire hands the bundle
// Electron's require for node builtins.
import fsp from "node:fs/promises";
import { pathToFileURL } from "node:url";
import { bundlePath, modelPath, RungSpec } from "./reader-assets";

interface EngineInstance {
  generate(prompt: string, opts?: { maxTokens?: number }): Promise<string>;
  dispose(): Promise<void>;
  gpu: string;
}

interface EngineModule {
  createReaderEngine(path: string, opts: { noThink: boolean }): Promise<EngineInstance>;
}

export class ReaderEngine {
  private engine: EngineInstance | null = null;
  private loading: Promise<EngineInstance> | null = null;
  private lastUse = 0;
  private unloadTimer: number | null = null;
  private rung: RungSpec;
  gpu = "";
  // Sticky load failure: surfaced in settings, cleared on retry/re-enable.
  lastError: string | null = null;

  constructor(rung: RungSpec, private idleUnloadMinutes: number) {
    this.rung = rung;
  }

  setRung(rung: RungSpec): void {
    if (rung.id !== this.rung.id) {
      this.rung = rung;
      void this.unload();
    }
  }

  private async load(): Promise<EngineInstance> {
    if (this.engine) return this.engine;
    if (this.loading) return this.loading;
    this.loading = (async () => {
      let blobUrl: string | null = null;
      try {
        const source = await fsp.readFile(bundlePath(), "utf8");
        // The globals are read by the engine module itself, which Chromium
        // evaluates in the main window realm; window IS that realm's global.
        const g = window as typeof window & {
          __srnReaderEngineUrl?: string;
          __srnRequire?: NodeJS.Require;
        };
        g.__srnReaderEngineUrl = pathToFileURL(bundlePath()).href;
        g.__srnRequire = window.require;
        blobUrl = URL.createObjectURL(new Blob([source], { type: "text/javascript" }));
        // eslint-disable-next-line no-unsanitized/method -- a blob: URL of the plugin's own staged engine bundle, never vault or network content
        const mod = (await import(blobUrl)) as EngineModule;
        const e = await mod.createReaderEngine(modelPath(this.rung), { noThink: this.rung.noThink });
        this.engine = e;
        this.gpu = e.gpu;
        this.lastError = null;
        return e;
      } catch (err) {
        this.lastError = err instanceof Error ? err.message : String(err);
        throw err;
      } finally {
        if (blobUrl !== null) URL.revokeObjectURL(blobUrl);
        this.loading = null;
      }
    })();
    return this.loading;
  }

  async generate(prompt: string, maxTokens: number): Promise<string> {
    const engine = await this.load();
    this.lastUse = Date.now();
    this.armUnload();
    const out = await engine.generate(prompt, { maxTokens });
    this.lastUse = Date.now();
    return out;
  }

  get loaded(): boolean {
    return this.engine !== null;
  }

  private armUnload(): void {
    if (this.unloadTimer !== null) window.clearTimeout(this.unloadTimer);
    if (this.idleUnloadMinutes <= 0) return;
    const ms = this.idleUnloadMinutes * 60_000;
    this.unloadTimer = window.setTimeout(() => {
      if (Date.now() - this.lastUse >= ms - 500) void this.unload();
      else this.armUnload();
    }, ms);
  }

  async unload(): Promise<void> {
    if (this.unloadTimer !== null) {
      window.clearTimeout(this.unloadTimer);
      this.unloadTimer = null;
    }
    const e = this.engine;
    this.engine = null;
    if (e) await e.dispose().catch(() => undefined);
  }
}
