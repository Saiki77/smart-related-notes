// Messages exchanged between the EmbeddingEngine proxy (renderer main thread)
// and the embed worker. Plain structured-cloneable data only — every field here
// crosses postMessage. Imported by BOTH bundles; must stay dependency-free.

import type { DevicePref, EmbedKind, ProgressInfo } from "../model-spec";

// --- renderer -> worker ------------------------------------------------------

// Sent exactly once, immediately after the worker is spawned. Carries the ORT
// wasm assets because the worker cannot reliably resolve them itself: it runs
// from a blob: URL (no import.meta.url base) and the renderer already knows the
// right source (the plugin's self-hosted app:// ort/ folder, or the pinned CDN).
export interface InitRequest {
  id: number;
  type: "init";
  modelId: string;
  devicePref: DevicePref;
  // WASM worker-thread count (1 = single-threaded). A failed THREADED bring-up
  // is unrecoverable in-realm (ort latches its wasm-init state), so the PROXY
  // retries by spawning a fresh worker with numThreads=1.
  numThreads: number;
  // The ORT JS glue source text; the worker wraps it in its OWN blob URL so the
  // runtime can import() it regardless of where the worker itself came from.
  wasmGlueText: string;
  // The matching .wasm binary, TRANSFERRED (not copied) into the worker.
  wasmBinary: ArrayBuffer;
}

// One shape for single and batch embeds (a single embed is a batch of one).
// Prefixing, pooling and sub-batching happen worker-side via modelSpec, exactly
// as the pre-worker EmbeddingEngine did them.
export interface EmbedRequest {
  id: number;
  type: "embed";
  texts: string[];
  kind: EmbedKind;
}

// Reply to a FetchRequestMessage: the renderer fetched the URL on the
// worker's behalf. Negative ids, chosen by the worker, so they can never
// collide with the renderer's own request ids.
export interface FetchResultMessage {
  id: number;
  type: "fetchResult";
  ok: boolean;
  status: number;
  headers: Record<string, string>;
  buffer?: ArrayBuffer;
  error?: string;
}

export type WorkerRequest = InitRequest | EmbedRequest | FetchResultMessage;

// --- worker -> renderer ------------------------------------------------------

export interface ReadyResponse {
  id: number; // the init request's id
  type: "ready";
  device: "webgpu" | "wasm";
}

// Model download/load progress for the init request (forwarded from
// transformers.js' progress_callback, same shape the store already consumes).
export interface ProgressResponse {
  id: number;
  type: "progress";
  info: ProgressInfo;
}

// All result rows concatenated into ONE Float32Array (row i occupies
// [i*dims, (i+1)*dims)), with its buffer TRANSFERRED back. The proxy slices it
// into independent per-row copies, byte-identical to the old in-realm result.
export interface ResultResponse {
  id: number;
  type: "result";
  data: Float32Array;
  dims: number;
  count: number;
}

export interface ErrorResponse {
  id: number;
  type: "error";
  message: string;
}

// Worker -> renderer: fetch this URL for me. Sent only on ancient Electron
// runtimes whose worker scope lacks the Fetch globals; the renderer (which
// always has them, plus the system proxy configuration) downloads and
// replies with a FetchResultMessage carrying the transferred bytes.
export interface FetchRequestMessage {
  id: number; // negative, worker-assigned
  type: "fetchRequest";
  url: string;
}

export type WorkerResponse =
  | ReadyResponse
  | ProgressResponse
  | ResultResponse
  | ErrorResponse
  | FetchRequestMessage;
