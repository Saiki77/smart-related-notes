import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import crypto from "node:crypto";
import zlib from "node:zlib";
import { Readable } from "node:stream";
import { requestUrl } from "obsidian";

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
  const ramGb = os.totalmem() / 1024 ** 3;
  return RUNGS.find((r) => ramGb >= r.minRamGb) ?? RUNGS[RUNGS.length - 1];
}

// Pinned checksums, so files a user downloaded in a BROWSER (the offline path
// for networks that block the plugin's own downloads) verify without any
// network access. npm integrity values are the registry's own, recorded for
// ENGINE_VERSION; GGUF sha256/size come from each repo's LFS pointer and were
// cross-checked against locally downloaded files.
const NPM_INTEGRITY: Record<string, string> = {
  "node-llama-cpp": "sha512-KnET3ttADYLCobjMnMTLkWkLt87rPRDhNzTZgGOCt8m8yTmdQW2sfLNmulGBA9laFvcyRIMOsgQST8lunPmMgw==",
  "@node-llama-cpp/mac-arm64-metal": "sha512-QeFyyTZWicxKGzyoYwR1VtBGM8R1/oHjai9DC6KSg3T8WYpZd3mqATy24GPPVxgg20yEXUCbNl4xyKXZsLD0dQ==",
  "@node-llama-cpp/mac-x64": "sha512-3/B1uT0dNkhGTVkjTpI6OlHdUsic9NWDeocO0GHeq134LmQan34rERpR7JJgyv50iXufah+mvumFD/VZvfUqqQ==",
  "@node-llama-cpp/win-x64-vulkan": "sha512-7V2SjNejon668+xmtlZ36u2FmtIT2fOfQbGjT5zJ6ydW1Xec53VsX1VwQdikx+79Y9/gEp7L2GtBcX6bc0LsCQ==",
  "@node-llama-cpp/win-x64": "sha512-Mbh9n74DCB5zTw02cme7Kp9nVg9X5Wvf+SNMWkXx5o3rGLuiSijGqRIktPOO3aHQwshB/RXC4j6I34HCPsgdgg==",
  "@node-llama-cpp/win-arm64": "sha512-UDx5NBXVRtcLaoQsF1gZiYlgXQYfxLbFVb4j4sa7Jgq/b6oq2lTAjeos7sYsMPRKa+S/BB/rDFis6FsRDJur7w==",
  "@node-llama-cpp/linux-x64-vulkan": "sha512-xTzv4cuTpsmmQgqvWWZvcvURHfPgUQdQzVXvYN1bwAEYq2DRVckEKrHPJ48824sFmdIRRJqDerniqEXiZlDPzA==",
  "@node-llama-cpp/linux-x64": "sha512-zCSTd5m4MDrLWzgUvOvuGGzHm9DiEONZt+srgHYhV4Ppu/T5TL3kENj7lHY1cmQccKNgepABiyb9KcigjZbvSQ==",
  "@node-llama-cpp/linux-arm64": "sha512-WFAffebfOLqBaZMfNsORns1G5vLMRVthxw/moDzON7TGYH6PTQN97h5YkLfRoSmZne/rtpHXH2LdYg0vFNAgnQ==",
};
const GGUF_PIN: Record<string, { sha256: string; bytes: number }> = {
  "Qwen3-8B-Q4_K_M.gguf": { sha256: "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785", bytes: 5027783488 },
  "Qwen3-4B-Instruct-2507-Q4_K_M.gguf": { sha256: "3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597", bytes: 2497281120 },
  "Qwen3-1.7B-Q4_K_M.gguf": { sha256: "b139949c5bd74937ad8ed8c8cf3d9ffb1e99c866c823204dc42c0d91fa181897", bytes: 1107409472 },
};
// Runtime payload of the main engine package; build-time payload excluded.
const MAIN_KEEP = /^(llama\/(?!gitRelease|toolchains|cmake|addon)|package\.json$)/;

function npmTarballUrl(name: string): string {
  const base = name.startsWith("@") ? name.split("/")[1] : name;
  return `https://registry.npmjs.org/${name}/-/${base}-${ENGINE_VERSION}.tgz`;
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
  // Test hook: the bench first-enable simulation redirects the asset root.
  const override = process.env.SRN_READER_HOME;
  return override ?? path.join(os.homedir(), ".cache", "smart-related-notes", "reader");
}

export function modelPath(rung: RungSpec): string {
  return path.join(assetsRoot(), "models", rung.file);
}

export function bundlePath(): string {
  return path.join(assetsRoot(), "dist", "reader-bundle.mjs");
}

export type AssetProgress = (label: string, done: number, total: number) => void;


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
  const meta = (await requestUrl({ url: `https://registry.npmjs.org/${name}/${ENGINE_VERSION}` })).json as {
    dist: { tarball: string; integrity: string };
  };
  progress(`Downloading ${name}`, 0, 1);
  // fetch, not requestUrl: this and the model download below stream through
  // Chromium's network stack (system proxy, enterprise certificates) and the
  // model one cannot be buffered whole in memory the way requestUrl does.
  const raw = new Uint8Array(await (await fetch(meta.dist.tarball)).arrayBuffer());
  const want = meta.dist.integrity;
  const got = `sha512-${crypto.createHash("sha512").update(raw).digest("base64")}`;
  if (want !== got) throw new Error(`checksum mismatch for ${name}`);
  extractTarball(raw, keep, destRoot);
  progress(`Downloading ${name}`, 1, 1);
}

function extractTarball(gz: Uint8Array, keep: RegExp, destRoot: string): void {
  const tar = zlib.gunzipSync(gz);
  for (const e of tarEntries(tar)) {
    const rel = e.name.replace(/^package\//, "");
    if (!keep.test(rel) || e.type === "5") continue;
    if (e.type !== "0" && e.type !== "\0") continue;
    if (rel.includes("..")) continue;
    const dest = path.join(destRoot, rel);
    fs.mkdirSync(path.dirname(dest), { recursive: true });
    fs.writeFileSync(dest, e.data);
  }
}

// ---------------------------------------------------------------- gguf -----

// Resumable streaming download with an incremental sha256 that survives ONLY
// full non-resumed runs; on resume the hash is recomputed from disk first.
async function downloadGguf(rung: RungSpec, progress: AssetProgress): Promise<void> {
  const dest = modelPath(rung);
  const part = `${dest}.part`;
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  if (fs.existsSync(dest)) return;

  const base = `https://huggingface.co/${rung.repo}`;
  // The LFS pointer file carries the upstream sha256.
  const pointer = (await requestUrl({ url: `${base}/raw/main/${rung.file}` })).text;
  const wantSha = /sha256:([0-9a-f]{64})/.exec(pointer)?.[1] ?? null;

  const hash = crypto.createHash("sha256");
  let have = 0;
  if (fs.existsSync(part)) {
    have = fs.statSync(part).size;
    // Feed the existing bytes through the hash so verification stays valid.
    hash.update(fs.readFileSync(part));
  }
  const total = rung.sizeMb * 1024 * 1024;
  const res = await fetch(`${base}/resolve/main/${rung.file}`, {
    headers: have > 0 ? { Range: `bytes=${have}-` } : {},
  });
  if (!res.ok && res.status !== 206) throw new Error(`model download failed: HTTP ${res.status}`);
  if (res.status === 200 && have > 0) {
    fs.rmSync(part, { recursive: false, force: true });
    have = 0;
    hash.destroy?.();
    return downloadGguf(rung, progress);
  }
  const out = fs.createWriteStream(part, { flags: have > 0 ? "a" : "w" });
  const reader = (res.body as ReadableStream<Uint8Array>).getReader();
  let sinceReport = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    hash.update(value);
    have += value.byteLength;
    sinceReport += value.byteLength;
    if (!out.write(value)) await new Promise<void>((r) => out.once("drain", () => r()));
    if (sinceReport > 16 * 1024 * 1024) {
      sinceReport = 0;
      progress(`Downloading ${rung.label}`, have, Math.max(total, have));
    }
  }
  await new Promise<void>((r) => out.end(() => r()));
  const gotSha = hash.digest("hex");
  if (wantSha && gotSha !== wantSha) {
    fs.rmSync(part, { recursive: false, force: true });
    throw new Error("model download corrupted (checksum mismatch); try again");
  }
  fs.renameSync(part, dest);
  progress(`Downloading ${rung.label}`, 1, 1);
}

// ---------------------------------------------------------------- public ---

export function assetsReady(rung: RungSpec): boolean {
  const root = assetsRoot();
  return (
    fs.existsSync(bundlePath()) &&
    fs.existsSync(path.join(root, "llama", "binariesGithubRelease.json")) &&
    fs.existsSync(path.join(root, "bins")) &&
    fs.existsSync(modelPath(rung))
  );
}

// Idempotent: each stage skips itself when already present.
export async function ensureAssets(
  pluginBundleSource: () => Promise<Uint8Array>,
  rung: RungSpec,
  progress: AssetProgress,
): Promise<void> {
  const root = assetsRoot();
  fs.mkdirSync(path.join(root, "dist"), { recursive: true });

  // The stamp is "<engine version>:<bundle hash>". The version part guards
  // the npm-fetched parts (llama/, bins/); the hash part guards the staged
  // JS bundle, so a plugin update that reworks the bundle WITHOUT a new
  // node-llama-cpp version still restages it. (4.0.5 shipped a reworked
  // bundle under the same engine version, and existing installs kept
  // loading the stale staged copy.)
  const source = await pluginBundleSource();
  const bundleHash = crypto.createHash("sha256").update(source).digest("hex").slice(0, 16);
  const stampWant = `${ENGINE_VERSION}:${bundleHash}`;
  const stamp = path.join(root, "engine.version");
  const current = fs.existsSync(stamp) ? fs.readFileSync(stamp).toString() : "";
  const currentVersion = current.split(":")[0];
  if (currentVersion !== ENGINE_VERSION && currentVersion !== "") {
    // Engine parts from another version cannot be trusted with this bundle.
    fs.rmSync(path.join(root, "llama"), { recursive: true, force: true });
    fs.rmSync(path.join(root, "bins"), { recursive: true, force: true });
    fs.rmSync(stamp, { recursive: false, force: true });
  }
  if (current !== stampWant || !fs.existsSync(bundlePath())) {
    // Local copy out of the plugin's inlined string; involves no network.
    progress("Staging engine", 0, 1);
    fs.writeFileSync(bundlePath(), source);
  }
  // Engine data + binaries: skipped entirely when an offline import (or an
  // earlier run) already put them there.
  if (!fs.existsSync(path.join(root, "llama", "binariesGithubRelease.json"))) {
    await extractNpmPackage("node-llama-cpp", MAIN_KEEP, root, progress);
  }
  if (!fs.existsSync(path.join(root, "bins"))) {
    for (const target of platformPackages()) {
      try {
        await extractNpmPackage(`@node-llama-cpp/${target}`, /^bins\//, root, progress);
      } catch (e) {
        // A missing optional target (e.g. no vulkan build for this version) is
        // fine as long as at least one target lands; verified below.
        console.warn(`[related-notes] reader target ${target}:`, e);
      }
    }
    if (!fs.existsSync(path.join(root, "bins"))) throw new Error("no engine binaries available for this platform");
  }
  fs.writeFileSync(stamp, stampWant);
  await downloadGguf(rung, progress);
}

export function removeAssets(): void {
  fs.rmSync(assetsRoot(), { recursive: true, force: true });
}

// ------------------------------------------------------------ offline ------
// For networks that block the plugin's own downloads (corporate proxies,
// domain blocklists): the user downloads these URLs in any browser — on this
// machine or another — and imports the files below. Import identifies every
// file by CONTENT (checksum), so browser-renamed files ("model (1).gguf")
// are fine and a wrong or tampered file can never be installed.

export interface OfflineItem {
  label: string;
  url: string;
  sizeLabel: string;
  present: boolean;
  optional: boolean;
}

export function offlineItems(rung: RungSpec): OfflineItem[] {
  const root = assetsRoot();
  const items: OfflineItem[] = [
    {
      label: "Engine core (node-llama-cpp)",
      url: npmTarballUrl("node-llama-cpp"),
      sizeLabel: "≈35 MB",
      present: fs.existsSync(path.join(root, "llama", "binariesGithubRelease.json")),
      optional: false,
    },
  ];
  const targets = platformPackages();
  for (const t of targets) {
    items.push({
      label: `Engine binaries (${t})`,
      url: npmTarballUrl(`@node-llama-cpp/${t}`),
      sizeLabel: "≈5-45 MB",
      present: fs.existsSync(path.join(root, "bins", t)),
      // On Windows/Linux the vulkan build is the fast path and the plain one
      // the fallback; either alone is enough to run.
      optional: targets.length > 1 && t === targets[targets.length - 1],
    });
  }
  const pin = GGUF_PIN[rung.file];
  items.push({
    label: rung.label,
    url: `https://huggingface.co/${rung.repo}/resolve/main/${rung.file}?download=true`,
    sizeLabel: `${(pin.bytes / 1e9).toFixed(1)} GB`,
    present: fs.existsSync(modelPath(rung)),
    optional: false,
  });
  return items;
}

export interface ImportCandidate {
  name: string;
  size: number;
  stream(): ReadableStream<Uint8Array>;
}

export interface ImportOutcome {
  file: string;
  ok: boolean;
  note: string;
}

async function slurp(c: ImportCandidate): Promise<Uint8Array> {
  const reader = c.stream().getReader();
  const parts: Uint8Array[] = [];
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
  }
  const out = new Uint8Array(parts.reduce((s, p) => s + p.byteLength, 0));
  let off = 0;
  for (const p of parts) {
    out.set(p, off);
    off += p.byteLength;
  }
  return out;
}

async function importGguf(c: ImportCandidate, spec: RungSpec, progress: AssetProgress): Promise<ImportOutcome> {
  const pin = GGUF_PIN[spec.file];
  const dest = modelPath(spec);
  if (fs.existsSync(dest)) return { file: c.name, ok: true, note: `${spec.label} is already installed` };
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  const part = `${dest}.import`;
  const hash = crypto.createHash("sha256");
  const out = fs.createWriteStream(part, { flags: "w" });
  const reader = c.stream().getReader();
  let have = 0;
  let first = true;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (first) {
        first = false;
        if (new TextDecoder().decode(value.subarray(0, 4)) !== "GGUF") {
          throw new Error("not a GGUF model file");
        }
      }
      hash.update(value);
      have += value.byteLength;
      if (!out.write(value)) await new Promise<void>((r) => out.once("drain", () => r()));
      progress(`Importing ${spec.label}`, have, pin.bytes);
    }
    await new Promise<void>((r) => out.end(() => r()));
    if (hash.digest("hex") !== pin.sha256) {
      throw new Error("checksum mismatch; the download is incomplete or altered, download it again");
    }
    fs.renameSync(part, dest);
    return { file: c.name, ok: true, note: `${spec.label} installed and verified` };
  } catch (e) {
    await new Promise<void>((r) => out.end(() => r()));
    fs.rmSync(part, { recursive: false, force: true });
    return { file: c.name, ok: false, note: e instanceof Error ? e.message : String(e) };
  }
}

function importEngineTarball(c: ImportCandidate, raw: Uint8Array): ImportOutcome {
  const root = assetsRoot();
  const got = `sha512-${crypto.createHash("sha512").update(raw).digest("base64")}`;
  const name = Object.keys(NPM_INTEGRITY).find((n) => NPM_INTEGRITY[n] === got);
  if (!name) {
    // Identify what it actually is, so the message can name the fix.
    try {
      for (const e of tarEntries(zlib.gunzipSync(raw))) {
        if (e.name.replace(/^package\//, "") === "package.json") {
          const pkg = JSON.parse(new TextDecoder().decode(e.data)) as { name?: string; version?: string };
          return {
            file: c.name,
            ok: false,
            note: `this is ${pkg.name ?? "an unknown package"} ${pkg.version ?? ""}; the reader needs version ${ENGINE_VERSION} from the links above`,
          };
        }
      }
    } catch {
      /* fall through */
    }
    return { file: c.name, ok: false, note: "not one of the reader's engine files" };
  }
  if (name === "node-llama-cpp") {
    fs.rmSync(path.join(root, "llama"), { recursive: true, force: true });
    extractTarball(raw, MAIN_KEEP, root);
    return { file: c.name, ok: true, note: "engine core installed and verified" };
  }
  extractTarball(raw, /^bins\//, root);
  return { file: c.name, ok: true, note: `engine binaries (${name.split("/")[1]}) installed and verified` };
}

// The browser's default download location, where the guided setup watches
// for the files it asked the browser to fetch.
export function downloadsDir(): string {
  return path.join(os.homedir(), "Downloads");
}

// Scan a folder for reader files and import what verifies. `skip` lets the
// caller remember already-tried files (keyed by path|size|mtime, so a file
// that was still downloading is retried once it changes). In-progress
// browser downloads (.crdownload/.part/.download) are ignored; a partial
// GGUF never matches its pinned byte size, so it is never picked up early.
export async function importFromFolder(
  dir: string,
  skip: (key: string) => boolean,
  progress: AssetProgress,
): Promise<{ key: string; outcome: ImportOutcome }[]> {
  if (!fs.existsSync(dir)) return [];
  const results: { key: string; outcome: ImportOutcome }[] = [];
  let names: string[];
  try {
    names = fs.readdirSync(dir);
  } catch (e) {
    // macOS gates ~/Downloads behind a per-app permission; without it the
    // scan throws EPERM. Surface that as a readable outcome instead of an
    // uncaught error, so the modal can point at the fix.
    const msg = e instanceof Error ? e.message : String(e);
    if (/EPERM|EACCES/.test(msg)) {
      throw new Error(
        `Obsidian has no permission to read ${dir}. Grant it under System Settings > Privacy and Security > Files and Folders, or use "Import them by hand" below.`,
      );
    }
    throw e;
  }
  for (const name of names) {
    if (/\.(crdownload|part|download|tmp)$/i.test(name)) continue;
    const p = path.join(dir, name);
    let st: { size: number; mtimeMs: number };
    try {
      const s = fs.statSync(p);
      if (!s.isFile()) continue;
      st = { size: s.size, mtimeMs: s.mtimeMs };
    } catch {
      continue;
    }
    const isModel = Object.values(GGUF_PIN).some((g) => g.bytes === st.size);
    const isTarball = /\.(tgz|tar\.gz|gz)$/i.test(name) && st.size > 0 && st.size <= 200 * 1024 * 1024;
    if (!isModel && !isTarball) continue;
    const key = `${p}|${st.size}|${Math.round(st.mtimeMs)}`;
    if (skip(key)) continue;
    const outcome = (
      await importAssetFiles(
        [{ name, size: st.size, stream: () => Readable.toWeb(fs.createReadStream(p)) as ReadableStream<Uint8Array> }],
        progress,
      )
    )[0];
    results.push({ key, outcome });
  }
  return results;
}

export async function importAssetFiles(files: ImportCandidate[], progress: AssetProgress): Promise<ImportOutcome[]> {
  const root = assetsRoot();
  const outcomes: ImportOutcome[] = [];
  for (const c of files) {
    const spec = RUNGS.find((r) => GGUF_PIN[r.file]?.bytes === c.size);
    if (spec) {
      outcomes.push(await importGguf(c, spec, progress));
      continue;
    }
    if (c.size > 200 * 1024 * 1024) {
      outcomes.push({ file: c.name, ok: false, note: "size matches no reader file; download the exact files linked above" });
      continue;
    }
    progress(`Importing ${c.name}`, 0, 1);
    const raw = await slurp(c);
    if (raw[0] === 0x1f && raw[1] === 0x8b) {
      outcomes.push(importEngineTarball(c, raw));
    } else if (new TextDecoder().decode(raw.subarray(0, 4)) === "GGUF") {
      outcomes.push({ file: c.name, ok: false, note: "a GGUF, but not one of the reader's models (wrong file or quantization)" });
    } else {
      outcomes.push({ file: c.name, ok: false, note: "unrecognized file; expected a .tgz engine file or a .gguf model" });
    }
  }
  // The stamp marks a complete engine; write it only when both halves exist.
  if (
    fs.existsSync(path.join(root, "llama", "binariesGithubRelease.json")) &&
    fs.existsSync(path.join(root, "bins"))
  ) {
    fs.writeFileSync(path.join(root, "engine.version"), ENGINE_VERSION);
  }
  return outcomes;
}

export function assetSizesMb(): { engine: number; models: { file: string; mb: number }[] } {
  const root = assetsRoot();
  let engine = 0;
  const walk = (d: string): number => {
    let s = 0;
    if (!fs.existsSync(d)) return 0;
    for (const e of fs.readdirSync(d, { withFileTypes: true })) {
      const p = path.join(d, e.name);
      s += e.isDirectory() ? walk(p) : fs.statSync(p).size;
    }
    return s;
  };
  engine = (walk(path.join(root, "llama")) + walk(path.join(root, "bins")) + walk(path.join(root, "dist"))) / 1e6;
  const models: { file: string; mb: number }[] = [];
  const mdir = path.join(root, "models");
  if (fs.existsSync(mdir)) {
    for (const m of fs.readdirSync(mdir)) models.push({ file: m, mb: fs.statSync(path.join(mdir, m)).size / 1e6 });
  }
  return { engine, models };
}
