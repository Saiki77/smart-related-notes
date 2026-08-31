// Reader asset management: everything the native engine needs on disk, none of
// it inside the vault (a 5 GB GGUF must never enter iCloud sync). Layout under
// ~/.cache/smart-related-notes/reader/:
//   dist/reader-bundle.mjs   copied from the plugin folder (re-copied per version)
//   llama/  bins/<target>/   extracted from the two pinned npm tarballs
//   models/<file>.gguf       downloaded from Hugging Face, resumable, sha256-verified

// Must match the node-llama-cpp devDependency the bundle was built from.
export const ENGINE_VERSION = "3.20.0";

export interface RungSpec {
  id: "large" | "mid" | "small";
  label: string;
  repo: string;
  file: string;
  sizeMb: number;
  minRamGb: number; // RAM-probe floor for the auto pick
  judge: boolean; // measured: may this rung run judgment tasks?
  noThink: boolean; // classic Qwen3 needs /no_think; 2507 instruct does not
}

export const RUNGS: RungSpec[] = [
  { id: "large", label: "Large (Qwen3 8B)", repo: "Qwen/Qwen3-8B-GGUF", file: "Qwen3-8B-Q4_K_M.gguf", sizeMb: 4800, minRamGb: 24, judge: true, noThink: true },
  { id: "mid", label: "Mid (Qwen3 4B)", repo: "unsloth/Qwen3-4B-Instruct-2507-GGUF", file: "Qwen3-4B-Instruct-2507-Q4_K_M.gguf", sizeMb: 2450, minRamGb: 12, judge: true, noThink: false },
  { id: "small", label: "Small (Qwen3 1.7B)", repo: "unsloth/Qwen3-1.7B-GGUF", file: "Qwen3-1.7B-Q4_K_M.gguf", sizeMb: 1110, minRamGb: 0, judge: false, noThink: true },
];

export function autoRung(): RungSpec {
  const os = require("node:os") as typeof import("os");
  const ramGb = os.totalmem() / 1024 ** 3;
  return RUNGS.find((r) => ramGb >= r.minRamGb) ?? RUNGS[RUNGS.length - 1];
}

// llama.cpp prebuilt targets for this process; first entry that exists in the
// registry wins at runtime inside the engine (it probes vulkan, then cpu).
function platformPackages(): string[] {
  const { platform, arch } = process;
  if (platform === "darwin") return arch === "arm64" ? ["mac-arm64-metal"] : ["mac-x64"];
  if (platform === "win32") return arch === "arm64" ? ["win-arm64"] : ["win-x64-vulkan", "win-x64"];
  return arch === "arm64" ? ["linux-arm64"] : ["linux-x64-vulkan", "linux-x64"];
}

export function assetsRoot(): string {
  const os = require("node:os") as typeof import("os");
  const path = require("node:path") as typeof import("path");
  // Test hook: the bench first-enable simulation redirects the asset root.
  const override = process.env.SRN_READER_HOME;
  return override ?? path.join(os.homedir(), ".cache", "smart-related-notes", "reader");
}

export function modelPath(rung: RungSpec): string {
  const path = require("node:path") as typeof import("path");
  return path.join(assetsRoot(), "models", rung.file);
}

export function bundlePath(): string {
  const path = require("node:path") as typeof import("path");
  return path.join(assetsRoot(), "dist", "reader-bundle.mjs");
}

export type AssetProgress = (label: string, done: number, total: number) => void;

interface FsMod {
  existsSync(p: string): boolean;
  mkdirSync(p: string, o?: { recursive: boolean }): void;
  writeFileSync(p: string, d: Uint8Array | string): void;
  readFileSync(p: string): Buffer;
  statSync(p: string): { size: number };
  renameSync(a: string, b: string): void;
  rmSync(p: string, o?: { recursive: boolean; force: boolean }): void;
  createWriteStream(p: string, o?: { flags: string }): NodeJS.WritableStream & { close(cb: () => void): void };
}
const fs = (): FsMod => require("node:fs") as FsMod;
const pathMod = (): typeof import("path") => require("node:path") as typeof import("path");

// ---------------------------------------------------------------- tar ------

// Minimal tar reader: enough for npm tarballs (ustar + GNU longname 'L').
function* tarEntries(buf: Uint8Array): Generator<{ name: string; type: string; data: Uint8Array }> {
  let off = 0;
  let longName: string | null = null;
  const text = (s: number, l: number): string => {
    let end = s;
    while (end < s + l && buf[end] !== 0) end++;
    return new TextDecoder().decode(buf.subarray(s, end));
  };
  while (off + 512 <= buf.length) {
    if (buf[off] === 0) break; // two zero blocks end the archive
    const rawName = text(off, 100);
    const prefix = text(off + 345, 155);
    const size = parseInt(text(off + 124, 12).trim() || "0", 8);
    const type = String.fromCharCode(buf[off + 156] || 48);
    const data = buf.subarray(off + 512, off + 512 + size);
    const name = longName ?? (prefix ? `${prefix}/${rawName}` : rawName);
    longName = null;
    if (type === "L") longName = new TextDecoder().decode(data).replace(/\0+$/, "");
    else yield { name, type, data };
    off += 512 + Math.ceil(size / 512) * 512;
  }
}

// Fetch one npm tarball, verify its registry sha512, extract `keep`-matching
// paths (package/ prefix stripped) under destRoot. Build-time-only payload
// (toolchains, source bundles) is skipped: the runtime needs llama/ and bins/.
async function extractNpmPackage(
  name: string,
  keep: RegExp,
  destRoot: string,
  progress: AssetProgress,
): Promise<void> {
  const crypto = require("node:crypto") as typeof import("crypto");
  const meta = await (await fetch(`https://registry.npmjs.org/${name}/${ENGINE_VERSION}`)).json() as {
    dist: { tarball: string; integrity: string };
  };
  progress(`Downloading ${name}`, 0, 1);
  const raw = new Uint8Array(await (await fetch(meta.dist.tarball)).arrayBuffer());
  const want = meta.dist.integrity;
  const got = `sha512-${crypto.createHash("sha512").update(raw).digest("base64")}`;
  if (want !== got) throw new Error(`checksum mismatch for ${name}`);
  const zlib = require("node:zlib") as typeof import("zlib");
  const tar = zlib.gunzipSync(raw);
  const f = fs();
  const path = pathMod();
  for (const e of tarEntries(tar)) {
    const rel = e.name.replace(/^package\//, "");
    if (!keep.test(rel) || e.type === "5") continue;
    if (e.type !== "0" && e.type !== "\0") continue;
    if (rel.includes("..")) continue;
    const dest = path.join(destRoot, rel);
    f.mkdirSync(path.dirname(dest), { recursive: true });
    f.writeFileSync(dest, e.data);
  }
  progress(`Downloading ${name}`, 1, 1);
}

// ---------------------------------------------------------------- gguf -----

// Resumable streaming download with an incremental sha256 that survives ONLY
// full non-resumed runs; on resume the hash is recomputed from disk first.
async function downloadGguf(rung: RungSpec, progress: AssetProgress): Promise<void> {
  const f = fs();
  const path = pathMod();
  const crypto = require("node:crypto") as typeof import("crypto");
  const dest = modelPath(rung);
  const part = `${dest}.part`;
  f.mkdirSync(path.dirname(dest), { recursive: true });
  if (f.existsSync(dest)) return;

  const base = `https://huggingface.co/${rung.repo}`;
  // The LFS pointer file carries the upstream sha256.
  const pointer = await (await fetch(`${base}/raw/main/${rung.file}`)).text();
  const wantSha = /sha256:([0-9a-f]{64})/.exec(pointer)?.[1] ?? null;

  const hash = crypto.createHash("sha256");
  let have = 0;
  if (f.existsSync(part)) {
    have = f.statSync(part).size;
    // Feed the existing bytes through the hash so verification stays valid.
    hash.update(f.readFileSync(part));
  }
  const total = rung.sizeMb * 1024 * 1024;
  const res = await fetch(`${base}/resolve/main/${rung.file}`, {
    headers: have > 0 ? { Range: `bytes=${have}-` } : {},
  });
  if (!res.ok && res.status !== 206) throw new Error(`model download failed: HTTP ${res.status}`);
  if (res.status === 200 && have > 0) {
    f.rmSync(part, { recursive: false, force: true });
    have = 0;
    hash.destroy?.();
    return downloadGguf(rung, progress);
  }
  const out = f.createWriteStream(part, { flags: have > 0 ? "a" : "w" });
  const reader = (res.body as ReadableStream<Uint8Array>).getReader();
  let sinceReport = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    hash.update(value);
    have += value.byteLength;
    sinceReport += value.byteLength;
    if (!out.write(value)) await new Promise((r) => out.once("drain", r));
    if (sinceReport > 16 * 1024 * 1024) {
      sinceReport = 0;
      progress(`Downloading ${rung.label}`, have, Math.max(total, have));
    }
  }
  await new Promise<void>((r) => out.end(r));
  const gotSha = hash.digest("hex");
  if (wantSha && gotSha !== wantSha) {
    f.rmSync(part, { recursive: false, force: true });
    throw new Error("model download corrupted (checksum mismatch); try again");
  }
  f.renameSync(part, dest);
  progress(`Downloading ${rung.label}`, 1, 1);
}

// ---------------------------------------------------------------- public ---

export function assetsReady(rung: RungSpec): boolean {
  const f = fs();
  const path = pathMod();
  const root = assetsRoot();
  return (
    f.existsSync(bundlePath()) &&
    f.existsSync(path.join(root, "llama", "binariesGithubRelease.json")) &&
    f.existsSync(path.join(root, "bins")) &&
    f.existsSync(modelPath(rung))
  );
}

// Idempotent: each stage skips itself when already present.
export async function ensureAssets(
  pluginBundleSource: () => Promise<Uint8Array>,
  rung: RungSpec,
  progress: AssetProgress,
): Promise<void> {
  const f = fs();
  const path = pathMod();
  const root = assetsRoot();
  f.mkdirSync(path.join(root, "dist"), { recursive: true });

  const stamp = path.join(root, "engine.version");
  const current = f.existsSync(stamp) ? f.readFileSync(stamp).toString() : "";
  if (current !== ENGINE_VERSION || !f.existsSync(bundlePath())) {
    progress("Staging engine", 0, 1);
    f.writeFileSync(bundlePath(), await pluginBundleSource());
    // Runtime data dir of the engine package; build-time payload excluded.
    if (current !== ENGINE_VERSION) {
      f.rmSync(path.join(root, "llama"), { recursive: true, force: true });
      f.rmSync(path.join(root, "bins"), { recursive: true, force: true });
    }
    await extractNpmPackage(
      "node-llama-cpp",
      /^(llama\/(?!gitRelease|toolchains|cmake|addon)|package\.json$)/,
      root,
      progress,
    );
    for (const target of platformPackages()) {
      try {
        await extractNpmPackage(`@node-llama-cpp/${target}`, /^bins\//, root, progress);
      } catch (e) {
        // A missing optional target (e.g. no vulkan build for this version) is
        // fine as long as at least one target lands; verified below.
        console.warn(`[related-notes] reader target ${target}:`, e);
      }
    }
    if (!f.existsSync(path.join(root, "bins"))) throw new Error("no engine binaries available for this platform");
    f.writeFileSync(stamp, ENGINE_VERSION);
  }
  await downloadGguf(rung, progress);
}

export function removeAssets(): void {
  fs().rmSync(assetsRoot(), { recursive: true, force: true });
}

export function assetSizesMb(): { engine: number; models: { file: string; mb: number }[] } {
  const f = fs();
  const path = pathMod();
  const root = assetsRoot();
  let engine = 0;
  const walk = (d: string): number => {
    const fsx = require("node:fs") as typeof import("fs");
    let s = 0;
    if (!f.existsSync(d)) return 0;
    for (const e of fsx.readdirSync(d, { withFileTypes: true })) {
      const p = path.join(d, e.name);
      s += e.isDirectory() ? walk(p) : f.statSync(p).size;
    }
    return s;
  };
  engine = (walk(path.join(root, "llama")) + walk(path.join(root, "bins")) + walk(path.join(root, "dist"))) / 1e6;
  const models: { file: string; mb: number }[] = [];
  const mdir = path.join(root, "models");
  if (f.existsSync(mdir)) {
    const fsx = require("node:fs") as typeof import("fs");
    for (const m of fsx.readdirSync(mdir)) models.push({ file: m, mb: f.statSync(path.join(mdir, m)).size / 1e6 });
  }
  return { engine, models };
}
