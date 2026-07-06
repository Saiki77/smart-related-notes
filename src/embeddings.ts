// The renderer-side EmbeddingEngine: a same-signature RPC PROXY in front of a
// dedicated Web Worker that hosts transformers.js + onnxruntime-web (see
// worker/embed-worker.ts for the full why). The renderer bundle no longer
// evaluates either library — this file only spawns the worker (from an inlined
// blob: URL), feeds it the ORT wasm assets, and marshals embed calls.
//
// The payoff is deterministic memory release: the ORT wasm heap can never
// shrink and its runtime has no teardown API, so the pre-worker engine pinned
// hundreds of MB to several GB from the first embed until Obsidian quit.
// worker.terminate() (the idle auto-unload below, plus dispose() on unload and
// engine swaps) now returns ALL of it — heap, pthread pool, session, JS objects
// — to the OS, and the next embed transparently respawns the worker in a few
// seconds (model weights come from the browser Cache API, no re-download).
//
// cosineSimilarity lives in the dependency-free vector-math module; the model
// behaviour table lives in model-spec.ts (shared with the worker bundle). Both
// are re-exported here for back-compat with existing importers.
export { cosineSimilarity } from "./vector-math";
export {
  modelSpec,
  modelUsesWholeNote,
  type ModelSpec,
  type DevicePref,
  type EmbedKind,
  type ProgressInfo,
  type ProgressCallback,
} from "./model-spec";

import workerSource from "virtual:embed-worker";
import type { DevicePref, EmbedKind, ProgressCallback } from "./model-spec";
import type {
  EmbedRequest,
  InitRequest,
  WorkerResponse,
} from "./worker/protocol";

// WASM worker-thread count, set by the plugin from the "Indexing speed" setting.
// 1 = single-threaded (slowest); higher = faster full reindexes. Read at every
// spawn, so a respawn after idle picks up the current value. null -> default.
let embedThreads: number | null = null;

export function setEmbedThreads(n: number): void {
  embedThreads = n > 0 ? Math.floor(n) : 1;
}

// The exact ORT glue+wasm pair a worker spawn runs. Supplied by the PLUGIN via
// setOrtAssetLoader: this module stays free of Obsidian APIs, and the plugin
// knows where the assets live (its own folder via the vault adapter, with a
// one-time pinned-CDN download when absent — see main.ts loadOrtAssets). The
// wasmBinary is TRANSFERRED into the worker, so the loader must return a fresh
// buffer per call.
export interface OrtAssets {
  glueText: string;
  wasmBinary: ArrayBuffer;
}
export type OrtAssetLoader = () => Promise<OrtAssets>;
let ortAssetLoader: OrtAssetLoader | null = null;

export function setOrtAssetLoader(loader: OrtAssetLoader): void {
  ortAssetLoader = loader;
}

function loadOrtAssets(): Promise<OrtAssets> {
  if (!ortAssetLoader) {
    // Only reachable if an embed ran before the plugin's onload wired the
    // loader — a programming error, not an environment failure.
    return Promise.reject(new Error("ORT asset loader not configured"));
  }
  return ortAssetLoader();
}

function resolveThreads(): number {
  if (embedThreads !== null) return embedThreads;
  const cores =
    typeof navigator !== "undefined" ? navigator.hardwareConcurrency || 4 : 4;
  return Math.max(1, Math.min(Math.ceil(cores / 3), 4)); // balanced default
}

interface PendingEntry {
  resolve: (rows: Float32Array[]) => void;
  reject: (e: Error) => void;
  onProgress?: ProgressCallback;
}

// One spawned worker + its in-flight request table. Sessions are single-use:
// a terminated (or crashed) session is never revived — the engine drops it and
// spawns a fresh one on the next demand.
class WorkerSession {
  dead = false;
  resolvedDevice: "webgpu" | "wasm" | null = null;
  private readonly worker: Worker;
  private readonly blobUrl: string;
  private readonly pending = new Map<number, PendingEntry>();
  private nextId = 1;
  private terminated = false;

  constructor(source: string) {
    this.blobUrl = URL.createObjectURL(
      new Blob([source], { type: "text/javascript" }),
    );
    // A MODULE worker, not a classic one: ort dynamic-import()s its glue .mjs,
    // which classic workers disallow.
    this.worker = new Worker(this.blobUrl, { type: "module" });
    this.worker.onmessage = (e: MessageEvent): void =>
      this.handleMessage(e.data as WorkerResponse);
    this.worker.onerror = (e: ErrorEvent): void => {
      // A script-level error does NOT destroy the worker realm by itself — the
      // wasm heap would stay resident in a zombie worker. Terminate outright
      // (rejecting pending requests with the real error message).
      this.terminate(new Error(e.message || "embed worker error"));
    };
  }

  // Send the one-time init (model + device + the transferred wasm assets) and
  // resolve when the worker reports the pipeline ready.
  initModel(
    modelId: string,
    devicePref: DevicePref,
    numThreads: number,
    assets: { glueText: string; wasmBinary: ArrayBuffer },
    onProgress?: ProgressCallback,
  ): Promise<void> {
    const id = this.nextId++;
    const req: InitRequest = {
      id,
      type: "init",
      modelId,
      devicePref,
      numThreads,
      wasmGlueText: assets.glueText,
      wasmBinary: assets.wasmBinary,
    };
    return new Promise<void>((resolve, reject) => {
      if (this.dead) {
        reject(new Error("embed worker terminated"));
        return;
      }
      this.pending.set(id, { resolve: () => resolve(), reject, onProgress });
      this.worker.postMessage(req, [assets.wasmBinary]);
    });
  }

  embed(texts: string[], kind: EmbedKind): Promise<Float32Array[]> {
    const id = this.nextId++;
    const req: EmbedRequest = { id, type: "embed", texts, kind };
    return new Promise<Float32Array[]>((resolve, reject) => {
      if (this.dead) {
        reject(new Error("embed worker terminated"));
        return;
      }
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage(req);
    });
  }

  // Kill the realm. This is THE memory-release primitive: it synchronously
  // destroys the wasm heap/SharedArrayBuffer, the nested pthread workers, the
  // ONNX session and every transformers object in one shot. Idempotent.
  terminate(reason?: Error): void {
    if (this.terminated) return;
    this.terminated = true;
    this.worker.terminate();
    URL.revokeObjectURL(this.blobUrl);
    this.fail(reason ?? new Error("embed worker terminated"));
  }

  private handleMessage(msg: WorkerResponse): void {
    const entry = this.pending.get(msg.id);
    if (!entry) return;
    if (msg.type === "progress") {
      entry.onProgress?.(msg.info);
      return;
    }
    this.pending.delete(msg.id);
    if (msg.type === "error") {
      entry.reject(new Error(msg.message));
      return;
    }
    if (msg.type === "ready") {
      this.resolvedDevice = msg.device;
      entry.resolve([]);
      return;
    }
    // "result": slice the flat transferred buffer into independent per-row
    // COPIES (subarray views would pin the whole batch buffer for as long as
    // any single row — e.g. a stored mean vector — stays referenced).
    const rows: Float32Array[] = [];
    for (let i = 0; i < msg.count; i++) {
      rows.push(msg.data.slice(i * msg.dims, (i + 1) * msg.dims));
    }
    entry.resolve(rows);
  }

  private fail(err: Error): void {
    this.dead = true;
    const entries = [...this.pending.values()];
    this.pending.clear();
    for (const entry of entries) entry.reject(err);
  }
}

// One feature-extraction engine, bound to a single model id. The plugin holds
// one instance and rebuilds it when the model id / device / thread preference
// changes. The public surface (modelId, devicePref, device, init, embed,
// embedBatch, dispose, setIdleUnload) is unchanged from the pre-worker engine —
// index-store.ts and main.ts call sites are byte-identical.
export class EmbeddingEngine {
  readonly modelId: string;
  readonly devicePref: DevicePref;
  // Fires after an idle unload actually released the worker, so the plugin can
  // trim caches that only pay off while the engine is warm.
  onIdleUnload: (() => void) | null = null;

  private session: WorkerSession | null = null;
  private sessionPromise: Promise<WorkerSession> | null = null;
  // Bumped on every unload/dispose; an in-flight spawn that outlives its
  // generation terminates itself instead of resurrecting the engine.
  private generation = 0;
  // TERMINALLY disposed (plugin unload / engine replaced): unlike an idle
  // unload, later calls must reject instead of respawning the worker —
  // otherwise an in-flight build loop resurrects a realm nobody owns.
  private disposed = false;
  // In-flight embed requests. The idle timer must never terminate under one —
  // the request would silently never resolve.
  private inFlight = 0;
  // In-flight spawn+model-load. Also activity for the idle timer: a settings
  // save can arm the timer mid-download, and firing then would abort the init
  // and throw the whole download away.
  private initInFlight = 0;
  // Idle auto-unload: after this many ms with no embed activity, terminate the
  // worker (null = never). Re-armed after init and after every embed call.
  private idleUnloadMs: number | null = null;
  private idleTimer: number | null = null;

  constructor(modelId: string, devicePref: DevicePref) {
    this.modelId = modelId;
    this.devicePref = devicePref;
  }

  // The device the pipeline actually initialised on, once init() has resolved.
  get device(): "webgpu" | "wasm" | null {
    return this.session?.resolvedDevice ?? null;
  }

  // True while the engine holds (or is creating) a live worker. The view uses
  // this to explain the wait when a search must first re-load the model.
  get loaded(): boolean {
    return this.sessionPromise !== null;
  }

  // Configure idle auto-unload. Applies immediately: a shorter timeout re-arms
  // the running timer, null/0 cancels it. Changing this NEVER rebuilds anything.
  // Clamped to the int32 setTimeout ceiling — past it Chromium fires the timer
  // IMMEDIATELY (overflow), which would unload the engine after every embed.
  setIdleUnload(ms: number | null): void {
    this.idleUnloadMs = ms !== null && ms > 0 ? Math.min(ms, 2 ** 31 - 1) : null;
    this.armIdleTimer();
  }

  // Lazily spawn the worker and load the model. Safe to call repeatedly: the
  // first call wins and every later caller awaits the same promise. On failure
  // the promise is cleared so a later retry can re-attempt — the failure is
  // never swallowed into a silent keyword-only mode.
  init(onProgress?: ProgressCallback): Promise<void> {
    return this.ensureSession(onProgress).then(() => undefined);
  }

  // Embed one string into a normalized Float32Array. Vectors are L2-normalized,
  // so cosine similarity reduces to a dot product. `kind` selects the e5
  // prefix; it is ignored for prefix-free models (applied worker-side).
  async embed(
    text: string,
    kind: EmbedKind = "query",
    onProgress?: ProgressCallback,
  ): Promise<Float32Array> {
    const rows = await this.request([text], kind, onProgress);
    return rows[0];
  }

  // Embed a batch in ONE worker round-trip (the worker runs it through the ORT
  // session in sub-batches — real batch throughput, order preserved). Returns
  // [] for an empty batch. Throws on backend failure (the caller decides how
  // to surface it).
  async embedBatch(
    texts: string[],
    kind: EmbedKind = "passage",
    onProgress?: ProgressCallback,
  ): Promise<Float32Array[]> {
    if (texts.length === 0) return [];
    return this.request(texts, kind, onProgress);
  }

  // TERMINAL teardown, for plugin unload and for an engine being REPLACED on a
  // settings swap: after this, every embed/init call REJECTS ("engine
  // disposed") instead of respawning — the store's per-batch catches and the
  // keyword/recency fallbacks already handle that. Without the terminal flag,
  // an in-flight reindex loop would transparently resurrect the worker realm
  // right after onunload freed it.
  async dispose(): Promise<void> {
    this.disposed = true;
    await this.unload();
  }

  // NON-terminal teardown — the idle-unload primitive. Terminates the worker
  // and drops every reference so the next embed starts fresh. Deterministically
  // returns the entire engine memory (wasm heap, pthread pool, session,
  // model/tokenizer objects) to the OS. Safe to call repeatedly and while calls
  // are in flight: new embeds respawn a new worker, while the doomed one is
  // given a bounded grace period to finish its work.
  private async unload(): Promise<void> {
    if (this.idleTimer !== null) {
      window.clearTimeout(this.idleTimer);
      this.idleTimer = null;
    }
    this.generation++;
    const pending = this.sessionPromise;
    const session = this.session;
    this.sessionPromise = null;
    this.session = null;
    let doomed = session;
    if (pending) {
      try {
        doomed = await pending;
      } catch {
        // Init failed or was cancelled — spawnAndInit already tore it down.
        doomed = session;
      }
    }
    if (!doomed) return;
    // Bounded grace period for in-flight requests (terminating mid-request
    // would leave their promises hanging; fail() below rejects any leftovers).
    const deadline = Date.now() + 30_000;
    while (this.inFlight > 0 && Date.now() < deadline) {
      await new Promise((r) => window.setTimeout(r, 200));
    }
    doomed.terminate();
  }

  private ensureSession(onProgress?: ProgressCallback): Promise<WorkerSession> {
    if (this.disposed) {
      return Promise.reject(new Error("engine disposed"));
    }
    if (this.sessionPromise) return this.sessionPromise;
    const promise = this.spawnAndInit(this.generation, onProgress);
    this.sessionPromise = promise;
    return promise;
  }

  private async spawnAndInit(
    gen: number,
    onProgress?: ProgressCallback,
  ): Promise<WorkerSession> {
    let session: WorkerSession | null = null;
    this.initInFlight++;
    try {
      const assets = await loadOrtAssets();
      if (this.generation !== gen) throw new Error("engine disposed");
      session = new WorkerSession(workerSource);
      this.session = session;
      const threads = resolveThreads();
      try {
        await session.initModel(
          this.modelId,
          this.devicePref,
          threads,
          assets,
          onProgress,
        );
      } catch (e) {
        // ort latches its wasm-init state for a realm's lifetime, so a failed
        // THREADED bring-up (nested pthread workers, SAB) is unrecoverable
        // in-realm: the only real retry is a FRESH realm running
        // single-threaded from the start. (The first wasmBinary was
        // transferred away, so the assets must be re-fetched.) Costs one extra
        // model-load attempt when the failure wasn't threads-related — e.g.
        // offline first run — which fails fast on the same fetch error.
        if (this.generation !== gen || threads <= 1) throw e;
        console.warn(
          "[related-notes] embed worker init failed; retrying single-threaded in a fresh worker",
          e,
        );
        session.terminate();
        const retryAssets = await loadOrtAssets();
        if (this.generation !== gen) throw new Error("engine disposed");
        session = new WorkerSession(workerSource);
        this.session = session;
        await session.initModel(
          this.modelId,
          this.devicePref,
          1,
          retryAssets,
          onProgress,
        );
      }
      if (this.generation !== gen) throw new Error("engine disposed");
      // The heavy resources now exist — start the idle-unload countdown
      // (each embed re-arms it, so this only fires after true inactivity).
      this.armIdleTimer();
      return session;
    } catch (e) {
      session?.terminate();
      if (this.session === session) this.session = null;
      // Reset ONLY our own generation's promise so a later init can retry —
      // never a newer session created by a concurrent dispose+re-init.
      if (this.generation === gen) this.sessionPromise = null;
      console.warn("[related-notes] embed worker init failed", e);
      throw e instanceof Error ? e : new Error(String(e));
    } finally {
      this.initInFlight--;
    }
  }

  private async request(
    texts: string[],
    kind: EmbedKind,
    onProgress?: ProgressCallback,
  ): Promise<Float32Array[]> {
    for (let attempt = 0; attempt < 2; attempt++) {
      const session = await this.ensureSession(onProgress);
      if (session.dead) {
        // The worker crashed or was superseded after init — drop and respawn.
        this.dropSession(session);
        continue;
      }
      this.inFlight++;
      try {
        return await session.embed(texts, kind);
      } catch (e) {
        // A dead session (wasm fault, worker crash) must ALWAYS be dropped and
        // terminated — otherwise the zombie realm stays the engine's current
        // session, pinning its heap. One respawn+retry keeps a single hiccup
        // from failing a whole reindex.
        if (session.dead) {
          this.dropSession(session);
          if (attempt === 0) continue;
        }
        throw e;
      } finally {
        this.inFlight--;
        this.armIdleTimer();
      }
    }
    throw new Error("embedding worker unavailable");
  }

  private dropSession(session: WorkerSession): void {
    if (this.session === session) {
      this.session = null;
      this.sessionPromise = null;
    }
    session.terminate();
  }

  // (Re-)arm the idle timer. No-op until a worker actually exists — an engine
  // that never embedded holds nothing worth unloading.
  private armIdleTimer(): void {
    if (this.idleTimer !== null) {
      window.clearTimeout(this.idleTimer);
      this.idleTimer = null;
    }
    if (this.idleUnloadMs === null || this.sessionPromise === null) return;
    this.idleTimer = window.setTimeout(() => {
      this.idleTimer = null;
      if (this.inFlight > 0 || this.initInFlight > 0) {
        // A call (or the spawn/model-load itself) is still running — try again
        // after the current work completes. Firing during an init would abort
        // it and throw away a possibly minutes-long first model download.
        this.armIdleTimer();
        return;
      }
      void this.unload().then(() => this.onIdleUnload?.());
    }, this.idleUnloadMs);
  }
}
