// Native reader backend: loads the engine bundle via dynamic import() and
// manages the model lifecycle (load on demand, unload after idle). The
// import() below stays a REAL dynamic import in the CJS main bundle
// (esbuild's supported["dynamic-import"] flag keeps it untransformed),
// because the engine bundle is ESM with top-level await and cannot be
// require()d.
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
      try {
        // eslint-disable-next-line no-unsanitized/method -- the URL is built from the plugin's own asset root (bundlePath), never from vault or network content
        const mod = (await import(pathToFileURL(bundlePath()).href)) as EngineModule;
        const e = await mod.createReaderEngine(modelPath(this.rung), { noThink: this.rung.noThink });
        this.engine = e;
        this.gpu = e.gpu;
        this.lastError = null;
        return e;
      } catch (err) {
        this.lastError = err instanceof Error ? err.message : String(err);
        throw err;
      } finally {
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
