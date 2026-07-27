// A minimal stand-in for the `obsidian` module so the REAL plugin sources can be
// imported and exercised in node.
//
// Why this exists: every earlier harness reimplemented the ranking pipeline, and
// that is exactly how the template de-crowding bug shipped. The harness measured
// note-level vectors and reported 9.9 -> 0.67, while the plugin ranks on
// chunk-level BiMax, which the correction never reached. Both were "right"; they
// were measuring different code. A test that does not run the shipped code path
// cannot catch a bug in the shipped code path.
//
// Node resolves this through an import alias, so `import { TFile } from "obsidian"`
// inside src/ picks this up. Only the surface index-store.ts actually touches is
// implemented; anything else throws loudly rather than silently returning
// undefined and producing a plausible wrong number.
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";

export class TFile {
  constructor(path, mtime = 0) {
    this.path = path;
    this.extension = path.split(".").pop();
    this.basename = path.slice(path.lastIndexOf("/") + 1).replace(/\.md$/, "");
    this.stat = { mtime, ctime: mtime, size: 0 };
  }
}
export class TAbstractFile {}
export class Notice {
  constructor(message) { this.message = message; }
  hide() {}
  setMessage(m) { this.message = m; return this; }
}
export function normalizePath(p) { return p.replace(/\\/g, "/").replace(/^\/+/, ""); }
export function debounce(fn) {
  const wrapped = (...args) => fn(...args);
  wrapped.cancel = () => {};
  return wrapped;
}
export function parseFrontMatterAliases() { return null; }
export function getFrontMatterInfo() { return { exists: false, frontmatter: "", contentStart: 0 }; }
// `App` and the other type-only imports are erased by the transform, but a named
// import still has to resolve at link time, so every symbol index-store.ts and its
// dependencies import must exist here even when it is only used as a type.
export class App {}
export class ItemView {}
export class Plugin {}
export class PluginSettingTab {}
export class Setting {}
export class Modal {}
export class EditorSuggest {}
export function setIcon() {}
export function prepareFuzzySearch() { return () => null; }

const HEADING_RE = /^\s{0,3}(#{1,6})\s+(.+?)\s*$/gm;
const TAG_RE = /(?:^|\s)#([\p{L}][\p{L}\p{N}_/-]*)/gu;
const LINK_RE = /\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]/g;

/**
 * Build a fake App over a real folder of markdown, with the pieces index-store.ts
 * reads: vault.getAbstractFileByPath / getMarkdownFiles / cachedRead, and a
 * metadataCache carrying headings, tags, frontmatter and resolvedLinks.
 */
export function makeApp(vaultDir, { exclude = () => false } = {}) {
  const files = [];
  const walk = (dir) => {
    for (const name of readdirSync(dir)) {
      if (name.startsWith(".")) continue;
      const abs = join(dir, name);
      const st = statSync(abs);
      if (st.isDirectory()) walk(abs);
      else if (name.endsWith(".md")) {
        const rel = relative(vaultDir, abs);
        if (!exclude(rel)) files.push(new TFile(rel, st.mtimeMs));
      }
    }
  };
  walk(vaultDir);

  const bodies = new Map();
  const read = (f) => {
    let b = bodies.get(f.path);
    if (b === undefined) {
      b = readFileSync(join(vaultDir, f.path), "utf8");
      bodies.set(f.path, b);
    }
    return b;
  };
  const byPath = new Map(files.map((f) => [f.path, f]));
  const byBasename = new Map();
  for (const f of files) {
    const k = f.basename.toLowerCase();
    if (!byBasename.has(k)) byBasename.set(k, f);
  }

  // resolvedLinks, the way Obsidian exposes it: {from: {to: count}}.
  const resolvedLinks = {};
  for (const f of files) {
    const out = {};
    for (const m of read(f).matchAll(LINK_RE)) {
      const name = m[1].trim();
      if (!name) continue;
      const target = byBasename.get(name.toLowerCase());
      if (!target || target.path === f.path) continue;
      out[target.path] = (out[target.path] ?? 0) + 1;
    }
    resolvedLinks[f.path] = out;
  }

  const caches = new Map();
  const cacheFor = (path) => {
    let c = caches.get(path);
    if (c) return c;
    const f = byPath.get(path);
    if (!f) return null;
    const raw = read(f);
    const fmMatch = raw.match(/^---\n([\s\S]*?)\n---/);
    const fmText = fmMatch ? fmMatch[1] : "";
    const body = fmMatch ? raw.slice(fmMatch[0].length) : raw;

    const headings = [];
    HEADING_RE.lastIndex = 0;
    for (const m of body.matchAll(HEADING_RE)) headings.push({ heading: m[2].trim(), level: m[1].length });

    const tags = [];
    for (const m of body.matchAll(TAG_RE)) tags.push({ tag: "#" + m[1] });

    const frontmatter = {};
    const inline = fmText.match(/^tags:\s*\[([^\]]*)\]/m);
    if (inline) frontmatter.tags = inline[1].split(",").map((t) => t.trim().replace(/["']/g, "")).filter(Boolean);
    const block = fmText.match(/^tags:\s*\n((?:\s*-\s*.+\n?)+)/m);
    if (block) frontmatter.tags = block[1].split("\n").map((l) => (l.match(/-\s*(.+)/) || [])[1]).filter(Boolean);

    c = { headings, tags, frontmatter, links: [], embeds: [] };
    caches.set(path, c);
    return c;
  };

  return {
    vault: {
      getMarkdownFiles: () => files,
      getAbstractFileByPath: (p) => byPath.get(p) ?? null,
      cachedRead: async (f) => read(f),
      read: async (f) => read(f),
      adapter: {
        exists: async () => false,
        read: async () => { throw new Error("stub adapter: no persisted index"); },
        write: async () => {},
        mkdir: async () => {},
        remove: async () => {},
        rename: async () => {},
      },
      on: () => ({}),
    },
    metadataCache: {
      resolvedLinks,
      getCache: (p) => cacheFor(p),
      getFileCache: (f) => cacheFor(f.path),
      getFirstLinkpathDest: (link) => byBasename.get(link.toLowerCase()) ?? null,
      on: () => ({}),
    },
    workspace: { on: () => ({}), getActiveFile: () => null },
  };
}

export class WorkspaceLeaf {}
export class MarkdownView {}
export class Component {}
export class Menu {}
export class Platform {}
export const apiVersion = "1.7.2";
export function requestUrl() { throw new Error("stub: no network in tests"); }
export function normalizePath2() {}
