// onnxruntime-web ships type definitions in types.d.ts, but its package.json
// "exports" map doesn't surface them under moduleResolution "Bundler", so TS can't
// resolve them automatically. We only import the package as an opaque namespace and
// hand it to transformers via Symbol.for("onnxruntime") (see ort-shim.ts) — we never
// call into it from TypeScript — so a minimal ambient declaration is all that's
// needed to keep the build strict/any-free.
declare module "onnxruntime-web" {
  // The runtime exposes InferenceSession, Tensor, env, etc.; transformers reads
  // them off this object. Typed as a read-only record of unknown so there is no
  // implicit `any` and no member is accidentally treated as callable here.
  const ort: Readonly<Record<string, unknown>>;
  export = ort;
}
