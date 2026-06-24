import { App, TFile, parseFrontMatterAliases } from "obsidian";

// =============================================================================
// TitleIndex — the PRECISION backbone shared by the inline glow (Feature A) and
// the smart `[[` suggester (Feature B).
//
// It maps every NORMALIZED title/alias SURFACE in the vault to the file(s) that
// own it, and compiles the longest-first alternation regex the glow scans with.
// Because ONLY surfaces that resolve to a real existing note ever enter that
// alternation, ANY regex hit is, by construction, a real matching note —
// precision is paramount (we never glow a phrase that has no note).
//
// Design mirrors IndexStore's ambiguousBasenames: lazy null-and-recompute,
// invalidated on vault create/delete/rename and on metadataCache 'changed'
// (aliases change without an mtime bump — the same reason easy-links drops its
// alias cache on 'changed').
// =============================================================================

// Surfaces shorter than this never glow — a precision guard so two-letter titles
// ("KI", "ML") don't paint half the prose. Measured on the normalized form.
const MIN_SURFACE_CHARS = 3;

export interface ResolvedSurface {
  file: TFile;
  // True when 2+ files own this surface; the caller (glow) may choose to skip
  // ambiguous surfaces, but a deterministic target is still provided so an
  // explicit insertion (suggester / command) can proceed.
  ambiguous: boolean;
}

export class TitleIndex {
  private readonly app: App;

  // Normalized surface -> the files that own it. An array (not a single file) so
  // ambiguity is detectable, mirroring IndexStore.ambiguousBasenames.
  private surfaceToFiles = new Map<string, TFile[]>();

  // Lazy rebuild flag: the map is recomputed on the next access after markDirty().
  private dirty = true;

  // The compiled longest-first alternation, cached PER active-note path: a note's
  // OWN surfaces are excluded from its own alternation (no self-link), so the
  // regex is only valid for the path it was compiled for.
  private compiled: RegExp | null = null;
  private compiledForPath: string | null = null;

  constructor(app: App) {
    this.app = app;
  }

  // Normalize a surface for comparison and as a map key:
  //   NFC first so a composed "ü" and a decomposed "u+¨" compare equal,
  //   then toLocaleLowerCase for German sharp-s / dotted-I correctness,
  //   then trim + collapse internal whitespace to a single space.
  normalize(s: string): string {
    return s
      .normalize("NFC")
      .toLocaleLowerCase()
      .trim()
      .replace(/\s+/g, " ");
  }

  // Mark the index stale (cheap). The map and every cached alternation are
  // recomputed lazily on the next access. Wired by the plugin to vault
  // create/delete/rename and metadataCache 'changed'.
  markDirty(): void {
    this.dirty = true;
    this.compiled = null;
    this.compiledForPath = null;
  }

  // (Re)build surfaceToFiles from every markdown file's basename + frontmatter
  // aliases. Cheap CPU work; the plugin debounces the markDirty() that precedes
  // it to coalesce bulk-edit bursts.
  private rebuild(): void {
    const map = new Map<string, TFile[]>();
    for (const file of this.app.vault.getMarkdownFiles()) {
      this.addSurface(map, file.basename, file);
      const cache = this.app.metadataCache.getFileCache(file);
      const aliases = cache ? parseFrontMatterAliases(cache.frontmatter) ?? [] : [];
      for (const alias of aliases) {
        if (typeof alias === "string") this.addSurface(map, alias, file);
      }
    }
    this.surfaceToFiles = map;
    this.dirty = false;
    // A rebuilt map invalidates any cached alternation.
    this.compiled = null;
    this.compiledForPath = null;
  }

  private addSurface(
    map: Map<string, TFile[]>,
    surfaceRaw: string,
    file: TFile,
  ): void {
    const surface = this.normalize(surfaceRaw);
    if (surface.length < MIN_SURFACE_CHARS) return;
    const existing = map.get(surface);
    if (existing) {
      if (!existing.includes(file)) existing.push(file);
    } else {
      map.set(surface, [file]);
    }
  }

  private ensureFresh(): void {
    if (this.dirty) this.rebuild();
  }

  // Resolve a raw surface to its owning file (+ ambiguity flag), or null when no
  // note owns it. When 2+ files own the surface a DETERMINISTIC target is chosen
  // (shortest path, then localeCompare) so an explicit insertion still works.
  resolve(surfaceRaw: string): ResolvedSurface | null {
    this.ensureFresh();
    const files = this.surfaceToFiles.get(this.normalize(surfaceRaw));
    if (!files || files.length === 0) return null;
    if (files.length === 1) return { file: files[0], ambiguous: false };
    const target = files
      .slice()
      .sort(
        (a, b) =>
          a.path.length - b.path.length || a.path.localeCompare(b.path),
      )[0];
    return { file: target, ambiguous: true };
  }

  // True when the normalized surface differs from the file's normalized basename —
  // i.e. the surface is an ALIAS, which drives the alias display-text branch in
  // buildWikiLink (so `[[Note|Alias]]` keeps the alias text the user wrote).
  isAlias(surfaceRaw: string, file: TFile): boolean {
    return this.normalize(surfaceRaw) !== this.normalize(file.basename);
  }

  // True when the surface resolves to ANY existing note. Used by the suggester to
  // reject a "create new note" concept that already exists.
  hasSurface(surfaceRaw: string): boolean {
    this.ensureFresh();
    return this.surfaceToFiles.has(this.normalize(surfaceRaw));
  }

  // Build (and cache) the longest-first alternation over every surface EXCEPT the
  // active note's own surfaces. Returns null when there is nothing to match.
  //
  // The boundaries are Unicode-aware lookbehind/lookahead asserting the char
  // before and after the match is NOT a letter or number (\p{L}/\p{N}) — NOT the
  // ASCII word boundary \b, which mis-handles umlauts and sharp-s. Flags: global
  // + ignorecase + unicode. (Surfaces are matched against the RAW document text,
  // so we rely on the 'i' flag rather than pre-lowercasing — the alternation
  // members are the normalized lower-cased surfaces, which 'i' matches in either
  // case for the ASCII range; non-ASCII letters in titles are matched verbatim
  // because they were NFC-normalized identically on both sides.)
  compiledFor(activePath: string): RegExp | null {
    this.ensureFresh();
    if (this.compiled && this.compiledForPath === activePath) {
      return this.compiled;
    }

    // The active note's own surfaces (basename + aliases) — excluded so a note
    // never glows a mention of itself.
    const ownSurfaces = new Set<string>();
    const activeFile = this.app.vault.getAbstractFileByPath(activePath);
    if (activeFile instanceof TFile) {
      ownSurfaces.add(this.normalize(activeFile.basename));
      const cache = this.app.metadataCache.getFileCache(activeFile);
      const aliases = cache ? parseFrontMatterAliases(cache.frontmatter) ?? [] : [];
      for (const a of aliases) {
        if (typeof a === "string") ownSurfaces.add(this.normalize(a));
      }
    }

    const surfaces: string[] = [];
    for (const surface of this.surfaceToFiles.keys()) {
      if (ownSurfaces.has(surface)) continue;
      surfaces.push(surface);
    }
    if (surfaces.length === 0) {
      this.compiled = null;
      this.compiledForPath = activePath;
      return null;
    }

    // LONGEST surface first so the alternation prefers the most specific match
    // ("Theory of mind" over "mind"). Each member is RegExp-escaped, then its
    // (already-collapsed) single spaces become `\s+` so a title written with
    // multiple internal spaces / a newline in the source still matches the
    // single-space normalized surface (precision: don't silently miss a real
    // mention). NFC-normalize the scanned text too (done in detectMentions /
    // buildGlow callers via the document text — Obsidian text is NFC in practice).
    surfaces.sort((a, b) => b.length - a.length || a.localeCompare(b));
    const alternation = surfaces
      .map((s) => escapeRegExp(s).replace(/ /g, "\\s+"))
      .join("|");
    // Lookbehind/lookahead assert a NON-letter/number boundary on both sides.
    const pattern = `(?<![\\p{L}\\p{N}])(?:${alternation})(?![\\p{L}\\p{N}])`;

    let regex: RegExp | null;
    try {
      regex = new RegExp(pattern, "giu");
    } catch {
      // A pathological surface set could in theory exceed an engine limit; fail
      // closed (no glow) rather than throw out of the ViewPlugin.
      regex = null;
    }
    this.compiled = regex;
    this.compiledForPath = activePath;
    return regex;
  }
}

// Escape a string for literal use inside a RegExp.
function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
