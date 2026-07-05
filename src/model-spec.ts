// Model-behaviour table + the small shared types for the embedding engine.
//
// Deliberately dependency-free (no transformers.js, no obsidian): this module is
// imported BOTH by the renderer bundle (index-store/main via ./embeddings) and by
// the embed worker bundle (see worker/embed-worker.ts), and pulling transformers
// into the renderer bundle would defeat the whole point of the worker realm.

// Device preference for the inference backend. The onnxruntime-web backend only
// supports "webgpu" (when navigator.gpu is present) and "wasm" — there is NO
// "cpu" provider in the web runtime ("wasm" IS the CPU path). "auto" probes
// WebGPU and falls back to WASM.
export type DevicePref = "auto" | "webgpu" | "wasm";

// Progress events surfaced from pipeline() during the one-time model download.
// transformers.js types this as `any`; narrow it to just the fields we read.
export interface ProgressInfo {
  // One of "initiate" | "download" | "progress" | "done" | "ready", but typed as
  // a plain string since transformers.js may add states and we only switch on a few.
  status: string;
  file?: string;
  loaded?: number;
  total?: number;
  progress?: number;
}

export type ProgressCallback = (info: ProgressInfo) => void;

export type EmbedKind = "query" | "passage";

// Per-model embedding behaviour, keyed by a substring of the model id. Paraphrase
// models (default) are symmetric: no prefix, mean pooling, chunked. e5 is retrieval:
// "query:"/"passage:" prefix, mean pooling. jina-embeddings-v5 text-matching is
// symmetric but uses a literal "Document: " prefix on BOTH sides, LAST-TOKEN pooling,
// and is embedded as a WHOLE NOTE (its 8192-token window holds a full note/idea), so
// the index uses a whole-note + idea-unit strategy for it (see wholeNote).
export interface ModelSpec {
  prefixByKind: boolean; // true -> `${kind}: ` (e5); false -> the fixed `prefix`
  prefix: string; // fixed prefix when !prefixByKind ("" = none)
  pooling: "mean" | "lastToken";
  wholeNote: boolean; // index embeds the whole note + idea-units, not <=480-char windows
}

export function modelSpec(modelId: string): ModelSpec {
  if (/jina-embeddings-v5/i.test(modelId)) {
    return { prefixByKind: false, prefix: "Document: ", pooling: "lastToken", wholeNote: true };
  }
  if (/paraphrase-multilingual/i.test(modelId)) {
    return { prefixByKind: false, prefix: "", pooling: "mean", wholeNote: false };
  }
  return { prefixByKind: true, prefix: "", pooling: "mean", wholeNote: false }; // e5
}

// True when the model is embedded with the whole-note + idea-unit strategy.
export function modelUsesWholeNote(modelId: string): boolean {
  return modelSpec(modelId).wholeNote;
}
