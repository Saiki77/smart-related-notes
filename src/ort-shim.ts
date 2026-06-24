// MUST be the very first import in main.ts, before anything that pulls in
// @huggingface/transformers.
//
// Why this exists: in the Obsidian (Electron) renderer, transformers.js detects a
// Node environment (process.release.name === "node" is true), so its backend
// selector takes the onnxruntime-NODE branch. But onnxruntime-node is externalized
// (no native binding loads in the renderer), leaving its InferenceSession
// undefined — so the first pipeline() call throws "Cannot read properties of
// undefined (reading 'create')", and webgpu/wasm are rejected as "Unsupported
// device" because only the Node provider list (["cpu"]) was ever built.
//
// The fix: transformers' backend module checks `Symbol.for("onnxruntime") in
// globalThis` FIRST, before its Node/Web detection. By installing our own bundled
// onnxruntime-web instance under that symbol BEFORE transformers is imported, the
// renderer uses the real web runtime: InferenceSession is defined, and
// supportedDevices is rebuilt from the web branch (webgpu when available, plus
// wasm). No onnxruntime-node, no server, fully self-contained.
import * as ortWeb from "onnxruntime-web";

// transformers' backend selector checks `Symbol.for("onnxruntime") in globalThis`
// against globalThis SPECIFICALLY, so we must write to globalThis here — not
// `window`/`activeWindow`. This is a one-time runtime-backend install at module
// load, not popout-window UI, so the no-global-this guideline doesn't apply.
// eslint-disable-next-line obsidianmd/no-global-this
(globalThis as Record<symbol, unknown>)[Symbol.for("onnxruntime")] = ortWeb;
