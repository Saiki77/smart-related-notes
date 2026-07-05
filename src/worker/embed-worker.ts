// Entry point of the EMBED WORKER: the only realm that ever evaluates
// @huggingface/transformers and onnxruntime-web.
//
// Why a worker at all: the ORT wasm runtime is a realm-lifetime singleton whose
// WebAssembly.Memory (a SharedArrayBuffer when threaded) can GROW BUT NEVER
// SHRINK, and whose pthread worker pool is never torn down — session.release()
// only frees pages INSIDE the heap for reuse. The one reliable way to return
// the hundreds-of-MB..multi-GB peak to the OS (the ORT maintainers' own
// recommendation) is to host the runtime in a worker and terminate() it. The
// renderer-side EmbeddingEngine (src/embeddings.ts) spawns this worker from an
// inlined blob: URL, RPCs embeds through it (see worker/protocol.ts), and
// terminates it after an idle timeout — respawning transparently on demand.
//
// This module must stay free of obsidian/electron/@codemirror imports: it is
// bundled separately (see esbuild.config.mjs) for a realm where none exist.

// MUST be first (see ort-shim.ts): if this worker DOES see a Node-like process
// (nodeIntegrationInWorker environments), the shim keeps transformers on the web
// branch. Without one (the normal case) it is a harmless no-op.
import "../ort-shim";
import {
  pipeline,
  env,
  type FeatureExtractionPipeline,
} from "@huggingface/transformers";
import { modelSpec, type EmbedKind, type ProgressInfo } from "../model-spec";
import type {
  EmbedRequest,
  InitRequest,
  WorkerRequest,
  WorkerResponse,
} from "./protocol";

// The worker global, narrowed to what we use. (The project compiles against the
// DOM lib, whose `self.postMessage` carries the window signature — cast once.)
const ctx = self as unknown as {
  postMessage(msg: WorkerResponse, transfer?: Transferable[]): void;
  onmessage: ((e: MessageEvent) => void) | null;
};

// embedBatch processes its inputs in SUB-BATCHES of this many texts. In the
// renderer this also served UI-responsiveness; in the worker it remains for the
// other half of its old job: keeping the per-forward-pass activation peak (and
// with it the high-water mark of the unshrinkable wasm heap) bounded, instead of
// letting one 130-text pass size the arena. Row order is unchanged.
const EMBED_SUB_BATCH = 12;

// The onnxruntime-web wasm flags. ort reads these from ort.env.wasm in the web
// build, which transformers exposes as env.backends.onnx.env.wasm (NOT
// env.backends.onnx.wasm — that's the Node-build shape). Set both shapes anyway.
interface OrtWasmFlags {
  wasmPaths?: string | { mjs?: string; wasm?: string };
  wasmBinary?: ArrayBuffer;
  numThreads?: number;
  proxy?: boolean;
}

type PoolNoneOut = { data: Float32Array; dims: readonly number[]; dispose?: () => void };

// L2-normalized LAST-TOKEN vector from a pooling:"none" output of a SINGLE input
// (shape [1, seq, dim] or [seq, dim]). jina v5 right-appends an EOS token; with no
// padding (single input) the final position IS the EOS the model pools on.
function lastTokenNorm(out: { data: Float32Array; dims: readonly number[] }): Float32Array {
  const data = out.data;
  const d = out.dims;
  const seq = d.length === 3 ? d[1] : d[0];
  const dim = d.length === 3 ? d[2] : d[1];
  const v = new Float32Array(data.subarray((seq - 1) * dim, seq * dim));
  let norm = 0;
  for (let i = 0; i < dim; i++) norm += v[i] * v[i];
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < dim; i++) v[i] /= norm;
  return v;
}

// Configure the transformers.js / ort environment from the init payload. Unlike
// the old renderer-side configureEnv there is no app://-vs-CDN resolution here:
// the renderer already fetched the exact glue+wasm pair and handed them over.
function configureEnv(req: InitRequest, numThreads: number): void {
  env.allowRemoteModels = true; // weights come from the HF Hub (cached below)
  env.allowLocalModels = false; // never take the Node FS path
  // Cache model weights in the browser Cache API. A blob: worker inherits the
  // renderer's origin, so this is the SAME "transformers-cache" the pre-worker
  // engine populated — respawns re-use the downloaded weights, no re-download.
  env.useBrowserCache = true;

  // The glue must be import()able from inside this (blob:) worker, so wrap the
  // transferred source text in the worker's own blob URL. NOT revoked: the
  // threaded runtime's pthread pool may import it again when spawning a worker.
  const glueUrl = URL.createObjectURL(
    new Blob([req.wasmGlueText], { type: "text/javascript" }),
  );
  const onnx = env.backends?.onnx as
    | { wasm?: OrtWasmFlags; env?: { wasm?: OrtWasmFlags } }
    | undefined;
  for (const wasm of [onnx?.wasm, onnx?.env?.wasm]) {
    if (!wasm) continue;
    // wasmBinary short-circuits ort's own .wasm fetch (which couldn't resolve
    // relative to a blob URL anyway); the mjs entry is where import() loads the
    // glue from. The object form is REQUIRED (v4's renderer-safe load path).
    wasm.wasmPaths = { mjs: glueUrl };
    wasm.wasmBinary = req.wasmBinary;
    wasm.numThreads = numThreads;
    wasm.proxy = false;
  }
}

// Probe WebGPU the way this realm exposes it (WebGPU is available in dedicated
// workers on current Chromium). False on any error -> transparent WASM fallback.
async function webgpuAvailable(): Promise<boolean> {
  try {
    const nav = navigator as Navigator & {
      gpu?: { requestAdapter(): Promise<unknown> };
    };
    if (!nav.gpu) return false;
    const adapter = await nav.gpu.requestAdapter();
    return adapter !== null && adapter !== undefined;
  } catch {
    return false;
  }
}

let pipePromise: Promise<FeatureExtractionPipeline> | null = null;
let modelId = "";

async function handleInit(req: InitRequest): Promise<void> {
  modelId = req.modelId;
  const onProgress = (info: ProgressInfo): void => {
    // Forward only the fields the protocol declares (the raw object can carry
    // extra, potentially non-cloneable payloads in future transformers versions).
    ctx.postMessage({
      id: req.id,
      type: "progress",
      info: {
        status: info.status,
        file: info.file,
        loaded: info.loaded,
        total: info.total,
        progress: info.progress,
      },
    });
  };

  const build = (device: "webgpu" | "wasm"): Promise<FeatureExtractionPipeline> =>
    pipeline("feature-extraction", req.modelId, {
      device,
      // q8 keeps the WASM download small and fast; fp32 is the WebGPU default
      // (accuracy over a slightly larger download).
      dtype: device === "webgpu" ? "fp32" : "q8",
      progress_callback: onProgress,
    });

  // The web runtime supports exactly "webgpu" (when an adapter exists) and
  // "wasm". Every order ends in "wasm" — the always-available CPU path. "auto"
  // uses WASM, not WebGPU: ort-web's WebGPU backend accumulates GPU/unified
  // memory across reindexes until Obsidian crashes (observed ~70GB), so WebGPU
  // stays an EXPLICIT pin only. (The worker realm makes even that pin safer:
  // terminate() now reclaims it.)
  const order: Array<"webgpu" | "wasm"> =
    req.devicePref === "webgpu" && (await webgpuAvailable())
      ? ["webgpu", "wasm"]
      : ["wasm"];

  // NO in-realm thread-count retry here: ort latches its wasm-init state for
  // the realm's lifetime (and reads numThreads only once), so re-running
  // configureEnv+pipeline() after a failed threaded bring-up cannot actually
  // downgrade to single-threaded. Recovery from a threads failure is the
  // PROXY's job: it spawns a FRESH worker realm with numThreads=1.
  configureEnv(req, req.numThreads);
  let lastErr: unknown;
  for (const device of order) {
    try {
      pipePromise = build(device);
      await pipePromise;
      ctx.postMessage({ id: req.id, type: "ready", device });
      return;
    } catch (e) {
      lastErr = e;
      pipePromise = null;
      console.warn(`[related-notes] ${device} init failed in the embed worker`, e);
    }
  }
  // Name the model in the error: the raw transformers/ort message ("Failed to
  // fetch", "Unauthorized access") often doesn't say what was being loaded.
  const reason =
    lastErr instanceof Error ? lastErr.message : "No embedding backend available.";
  ctx.postMessage({
    id: req.id,
    type: "error",
    message: `loading model "${req.modelId}" failed: ${reason}`,
  });
}

async function handleEmbed(req: EmbedRequest): Promise<void> {
  if (!pipePromise) {
    ctx.postMessage({ id: req.id, type: "error", message: "engine not initialised" });
    return;
  }
  const pipe = await pipePromise;
  const spec = modelSpec(modelId);
  const kind: EmbedKind = req.kind;
  const inputs = spec.prefixByKind
    ? req.texts.map((t) => `${kind}: ${t}`)
    : spec.prefix
      ? req.texts.map((t) => spec.prefix + t)
      : req.texts;

  const rows: Float32Array[] = [];
  if (spec.pooling === "lastToken") {
    // LAST-TOKEN models (jina): last-token pooling is only correct WITHOUT
    // right-padding, so embed one input at a time (no batch padding).
    for (const input of inputs) {
      const o = (await pipe(input, { pooling: "none" })) as unknown as PoolNoneOut;
      rows.push(lastTokenNorm(o));
      o.dispose?.();
    }
  } else {
    // Sub-batch the forward passes (see EMBED_SUB_BATCH). Rows are concatenated
    // IN INPUT ORDER, so the flattened result below is byte-identical to the old
    // whole-batch renderer-side result.
    for (let start = 0; start < inputs.length; start += EMBED_SUB_BATCH) {
      const slice = inputs.slice(start, start + EMBED_SUB_BATCH);
      const out = await pipe(slice, { pooling: "mean", normalize: true });
      const data = out.data as Float32Array;
      const n = slice.length;
      // dims is [N, D]; derive D defensively from the flat length when shapes drift.
      const dims =
        Array.isArray(out.dims) && out.dims.length === 2
          ? out.dims[1]
          : Math.floor(data.length / n);
      for (let i = 0; i < n; i++) {
        rows.push(new Float32Array(data.subarray(i * dims, (i + 1) * dims)));
      }
      // Free the backing tensor (the rows above are independent copies).
      (out as { dispose?: () => void }).dispose?.();
    }
  }

  // Flatten into one transferable buffer: row i occupies [i*dims, (i+1)*dims).
  const dims = rows[0]?.length ?? 0;
  const flat = new Float32Array(rows.length * dims);
  for (let i = 0; i < rows.length; i++) flat.set(rows[i], i * dims);
  ctx.postMessage(
    { id: req.id, type: "result", data: flat, dims, count: rows.length },
    [flat.buffer],
  );
}

ctx.onmessage = (e: MessageEvent): void => {
  const req = e.data as WorkerRequest;
  const run = req.type === "init" ? handleInit(req) : handleEmbed(req);
  run.catch((err: unknown) => {
    ctx.postMessage({
      id: req.id,
      type: "error",
      message: err instanceof Error ? err.message : String(err),
    });
  });
};
