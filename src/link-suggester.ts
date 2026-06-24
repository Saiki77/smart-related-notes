import {
  EditorSuggest,
  TFile,
  setIcon,
  prepareFuzzySearch,
  parseFrontMatterAliases,
  normalizePath,
  Notice,
  debounce,
  type Editor,
  type EditorPosition,
  type EditorSuggestContext,
  type EditorSuggestTriggerInfo,
  type SearchResult,
  type Debouncer,
} from "obsidian";
import type RelatedNotesPlugin from "./main";
import { buildWikiLink } from "./link-glow";

// =============================================================================
// Feature B — SMARTER `[[` SUGGESTER (Obsidian EditorSuggest).
//
// On `[[`, suggest EXISTING notes ranked by SEMANTIC relevance to the current
// note/cursor context (reusing the embedding index), AND propose "create NEW
// note: <concept>" rows for strongly-relevant phrases that aren't notes yet.
//
// Fast: onTrigger is sync + cheap; getSuggestions serves from a CACHED context
// vector (debounced recompute, ~200ms) so keystrokes stay CPU-only. Falls back to
// recency order when no vector is ready (model loading / cold start).
//
// Coexists with easy-links: it also hooks `[[` via EditorSuggest. The plugin's
// applySuggesterPrecedence() reorders ours appropriately; see main.ts. This
// suggester works fully WITHOUT easy-links too.
// =============================================================================

// Triggers ONLY on the plain `[[query` case — no `]`, `#`, `^`, `|`, or newline in
// the query. Heading/block/alias completions belong to the native suggester. This
// is the same proven regex easy-links uses.
const PLAIN_LINK_RE = /\[\[([^\]\n#^|]*)$/;

// Popup cap.
const LIMIT = 40;
// How many semantic candidates to pull from the index before fuzzy-filtering.
const SEMANTIC_POOL = 80;
// Blend weights when the user has typed a query: mostly fuzzy, nudged by semantics.
const FUZZY_WEIGHT = 0.6;
const SEMANTIC_WEIGHT = 0.4;
// Up to this many "create new note" rows.
const MAX_NEW_ROWS = 3;
// Context-embed debounce (ms): the vector is usually ready by the time the popup
// opens, and a fast typist never triggers a recompute storm.
const CONTEXT_DEBOUNCE = 200;
// How much text around the cursor to embed as the context.
const CONTEXT_RADIUS = 600;

type Item =
  | { kind: "note"; file: TFile; alias?: string; score: number; semantic: number }
  | { kind: "new"; concept: string };

// Narrow shape of the active context captured at trigger time, so getSuggestions
// can embed the surrounding paragraph and the selection knows the source path.
interface BlockContext {
  path: string;
  text: string;
  hash: string;
}

export class SmartLinkSuggester extends EditorSuggest<Item> {
  private readonly plugin: RelatedNotesPlugin;

  // Cached context vector, keyed by active path + a cheap hash of the surrounding
  // block. Recomputed (debounced) only when the block changes.
  private contextVec: Float32Array | null = null;
  private contextKey: string | null = null;
  private pendingKey: string | null = null;
  private readonly scheduleEmbed: Debouncer<[BlockContext], void>;

  // Frontmatter aliases per file, validated by mtime, so a keystroke doesn't
  // re-parse every note's frontmatter. Dropped on metadata/vault events.
  private readonly aliasCache = new Map<
    string,
    { mtime: number; aliases: string[] }
  >();

  constructor(plugin: RelatedNotesPlugin) {
    super(plugin.app);
    this.plugin = plugin;
    this.scheduleEmbed = debounce(
      (ctx: BlockContext) => void this.computeContextVec(ctx),
      CONTEXT_DEBOUNCE,
      false,
    );
  }

  // Drop one file's cached aliases (on rename/delete/metadata change).
  invalidateAliasCache(path: string): void {
    this.aliasCache.delete(path);
  }

  private aliasesOf(file: TFile): string[] {
    const cached = this.aliasCache.get(file.path);
    if (cached !== undefined && cached.mtime === file.stat.mtime) {
      return cached.aliases;
    }
    const cache = this.plugin.app.metadataCache.getFileCache(file);
    const aliases = cache ? parseFrontMatterAliases(cache.frontmatter) ?? [] : [];
    this.aliasCache.set(file.path, { mtime: file.stat.mtime, aliases });
    return aliases;
  }

  onTrigger(
    cursor: EditorPosition,
    editor: Editor,
    file: TFile | null,
  ): EditorSuggestTriggerInfo | null {
    // Disabled => let native / easy-links handle everything.
    if (!this.plugin.settings.suggesterEnabled) return null;

    const line = editor.getLine(cursor.line);
    const before = line.slice(0, cursor.ch);
    const m = PLAIN_LINK_RE.exec(before);
    if (m === null) return null;

    // Stash the cursor-block context and kick the debounced embed so the vector is
    // usually ready by the time getSuggestions runs.
    if (file) {
      const block = this.captureBlock(editor, cursor, file);
      if (block.hash !== this.contextKey && block.hash !== this.pendingKey) {
        this.pendingKey = block.hash;
        this.scheduleEmbed(block);
      }
    }

    const start: EditorPosition = { line: cursor.line, ch: m.index };
    return { start, end: cursor, query: m[1] };
  }

  // Grab ~CONTEXT_RADIUS chars of text around the cursor (excluding the open `[[`)
  // as the semantic context, plus a cheap stable hash so we don't re-embed an
  // unchanged block.
  private captureBlock(
    editor: Editor,
    cursor: EditorPosition,
    file: TFile,
  ): BlockContext {
    const offset = editor.posToOffset(cursor);
    const full = editor.getValue();
    const from = Math.max(0, offset - CONTEXT_RADIUS);
    const to = Math.min(full.length, offset + CONTEXT_RADIUS);
    const text = full.slice(from, to).replace(/\[\[[^\]]*$/, " ");
    return { path: file.path, text, hash: `${file.path}:${cheapHash(text)}` };
  }

  private async computeContextVec(ctx: BlockContext): Promise<void> {
    try {
      const trimmed = ctx.text.trim();
      if (trimmed.length === 0) return;
      const vec = await this.plugin.store.embedQuery(trimmed);
      this.contextVec = vec;
      this.contextKey = ctx.hash;
    } catch (e) {
      // Model not ready / failed: leave the vector null so we fall back to recency.
      console.warn("[related-notes] context embed failed", e);
    } finally {
      if (this.pendingKey === ctx.hash) this.pendingKey = null;
    }
  }

  getSuggestions(context: EditorSuggestContext): Item[] {
    const query = context.query;
    // The active note's path: it must NEVER appear as a suggestion (a self-link),
    // and a context vector embedded from text inside it ranks it #1, so we exclude
    // it from both the semantic pool and the fuzzy/recency branches below.
    const activePath = context.file?.path ?? this.context?.file?.path ?? null;
    const semantic = this.semanticCandidates(activePath);

    const items: Item[] = [];
    if (query.length === 0) {
      // Pure semantic ordering — the key value-add over folder-only ordering.
      for (const { file, semantic: s } of semantic) {
        items.push({ kind: "note", file, score: s, semantic: s });
      }
    } else {
      const semanticByPath = new Map<string, number>();
      for (const { file, semantic: s } of semantic) {
        semanticByPath.set(file.path, s);
      }
      const fuzzy = prepareFuzzySearch(query);
      // Per-query running maximum of |score| across CANDIDATES, used to normalize
      // each fuzzy magnitude into [0,1] (see fuzzyMagnitude). prepareFuzzySearch
      // returns NEGATIVE scores (0 = best), so we collect |score| first, then
      // normalize in a second pass once the strongest candidate magnitude is known.
      let maxMag = 0;
      const raw: {
        file: TFile;
        alias?: string;
        mag: number;
        sem: number;
      }[] = [];
      for (const file of this.plugin.app.vault.getMarkdownFiles()) {
        if (activePath !== null && file.path === activePath) continue;
        let best: SearchResult | null = fuzzy(file.basename);
        let bestAlias: string | undefined;
        for (const alias of this.aliasesOf(file)) {
          const r = fuzzy(alias);
          if (r !== null && (best === null || r.score > best.score)) {
            best = r;
            bestAlias = alias;
          }
        }
        if (best === null) continue;
        const mag = fuzzyMagnitude(best.score);
        if (mag > maxMag) maxMag = mag;
        raw.push({
          file,
          alias: bestAlias,
          mag,
          sem: semanticByPath.get(file.path) ?? 0,
        });
      }
      for (const r of raw) {
        // Normalize this candidate's magnitude against the strongest candidate
        // magnitude this query produced, so normFuzzy ∈ [0,1] and the 60/40 blend
        // with the [0,1] semantic term is meaningful (and never sign-flipped or
        // collapsed when a perfect self-match would have scored 0).
        const normFuzzy = maxMag > 0 ? r.mag / maxMag : 0;
        const score = FUZZY_WEIGHT * normFuzzy + SEMANTIC_WEIGHT * r.sem;
        items.push({
          kind: "note",
          file: r.file,
          alias: r.alias,
          score,
          semantic: r.sem,
        });
      }
      items.sort((a, b) => scoreOf(b) - scoreOf(a));
    }

    const notes = items.slice(0, LIMIT - MAX_NEW_ROWS);
    const newRows = this.newNoteRows(query, semantic, notes);
    return [...notes, ...newRows].slice(0, LIMIT);
  }

  // Semantic candidate list from the cached context vector, or [] when no vector
  // is ready yet (the query path then falls back to fuzzy-only; the empty-query
  // path falls back to recency below). The active note is excluded so it never
  // surfaces as a self-link.
  private semanticCandidates(
    activePath: string | null,
  ): { file: TFile; semantic: number }[] {
    if (this.contextVec) {
      return this.plugin.store.rankForContext(
        this.contextVec,
        SEMANTIC_POOL,
        activePath ?? undefined,
      );
    }
    // Recency fallback: newest notes first.
    return this.plugin.app.vault
      .getMarkdownFiles()
      .filter((file) => activePath === null || file.path !== activePath)
      .sort((a, b) => b.stat.mtime - a.stat.mtime)
      .slice(0, SEMANTIC_POOL)
      .map((file) => ({ file, semantic: 0 }));
  }

  // Propose up to MAX_NEW_ROWS "create new note: <concept>" rows for relevant
  // concepts that don't already resolve to a note. With a typed query the single
  // candidate concept is the query text — like native `[[` create-by-name, this
  // row is always offered (the user clearly intends the name), so
  // newNoteMinSimilarity does NOT gate it. With no query we MINE the top semantic
  // candidates' topic labels and there newNoteMinSimilarity IS the confidence
  // floor (their basenames are existing notes, so we never propose those). Each
  // concept is rejected if it already resolves via the TitleIndex or appears among
  // the shown note rows.
  private newNoteRows(
    query: string,
    semantic: { file: TFile; semantic: number }[],
    shown: Item[],
  ): Item[] {
    if (!this.plugin.settings.suggestNewNotes) return [];
    const out: Item[] = [];
    const seen = new Set<string>();
    const titleIndex = this.plugin.titleIndex;

    const consider = (raw: string, confidence: number): void => {
      const concept = raw.trim();
      if (concept.length < 3) return;
      if (confidence < this.plugin.settings.newNoteMinSimilarity) return;
      const key = concept.toLocaleLowerCase();
      if (seen.has(key)) return;
      // Reject concepts that already exist as a note/alias.
      if (titleIndex.hasSurface(concept)) return;
      // Reject ones already shown as a note row.
      if (
        shown.some(
          (it) =>
            it.kind === "note" &&
            it.file.basename.toLocaleLowerCase() === key,
        )
      ) {
        return;
      }
      seen.add(key);
      out.push({ kind: "new", concept });
    };

    if (query.trim().length >= 3) {
      // A typed query the user clearly intends — always offered (parity with the
      // native create-by-name row); newNoteMinSimilarity intentionally does not
      // gate it. Confidence 1 makes consider()'s floor a no-op for this row.
      consider(query, 1);
    } else {
      // Mine concept labels from the most-relevant notes' own topic labels.
      for (const { file, semantic: s } of semantic.slice(0, 8)) {
        const label = this.plugin.store.getSummary(file);
        if (label) {
          for (const part of label.split("·")) consider(part, s);
        }
        if (out.length >= MAX_NEW_ROWS) break;
      }
    }
    return out.slice(0, MAX_NEW_ROWS);
  }

  renderSuggestion(item: Item, el: HTMLElement): void {
    el.addClass("mod-complex");
    const content = el.createDiv({ cls: "suggestion-content" });

    if (item.kind === "new") {
      el.addClass("srn-new-suggestion");
      const titleRow = content.createDiv({ cls: "suggestion-title" });
      setIcon(titleRow.createSpan({ cls: "srn-new-icon" }), "file-plus");
      titleRow.createSpan({ text: item.concept });
      content.createDiv({ cls: "suggestion-note", text: "Create new note" });
      el.createDiv({ cls: "suggestion-aux", text: "New" });
      return;
    }

    const title = item.alias ?? item.file.basename;
    content.createDiv({ cls: "suggestion-title", text: title });
    const parentPath = item.file.parent?.path ?? "";
    if (parentPath.length > 0 && parentPath !== "/") {
      content.createDiv({ cls: "suggestion-note", text: parentPath });
    }
    // A small semantic badge when there is a meaningful signal.
    if (item.semantic > 0.01) {
      el.createDiv({
        cls: "suggestion-aux srn-semantic-badge",
        text: `${Math.round(item.semantic * 100)}%`,
      });
    }
  }

  selectSuggestion(item: Item, _evt: MouseEvent | KeyboardEvent): void {
    const context = this.context;
    if (context === null) return;
    const editor = context.editor;

    if (item.kind === "new") {
      void this.createAndLink(item.concept, context);
      return;
    }

    const link = buildWikiLink(
      item.file,
      item.alias ?? item.file.basename,
      this.plugin.titleIndex,
    );
    this.insert(editor, context, link);
    this.close();
  }

  // Create a new note for the concept, then link it. Respects the user's
  // new-note folder; guards name collisions; surfaces failures via a Notice.
  private async createAndLink(
    concept: string,
    context: EditorSuggestContext,
  ): Promise<void> {
    try {
      const folder = this.newNoteFolder();
      const safeName = concept.replace(/[\\/:*?"<>|#^[\]]/g, "").trim();
      if (safeName.length === 0) {
        new Notice("Related notes: can't create a note with that name.");
        return;
      }
      let path = normalizePath(
        folder ? `${folder}/${safeName}.md` : `${safeName}.md`,
      );
      // Avoid clobbering an existing file (race / case-folding): suffix a counter.
      let n = 2;
      while (this.plugin.app.vault.getAbstractFileByPath(path)) {
        path = normalizePath(
          folder ? `${folder}/${safeName} ${n}.md` : `${safeName} ${n}.md`,
        );
        n++;
      }
      const file = await this.plugin.app.vault.create(path, "");
      const link = buildWikiLink(file, concept, this.plugin.titleIndex);
      this.insert(context.editor, context, link);
      this.close();
    } catch (e) {
      console.warn("[related-notes] create new note failed", e);
      new Notice("Related notes: could not create the note. See the console.");
    }
  }

  // The folder a new note should be created in, honouring Obsidian's "Default
  // location for new notes" setting where it points at a specific folder.
  private newNoteFolder(): string {
    const active = this.plugin.app.workspace.getActiveFile();
    try {
      const dest = this.plugin.app.fileManager.getNewFileParent(
        active?.path ?? "",
      );
      return dest.path === "/" ? "" : dest.path;
    } catch {
      return "";
    }
  }

  // Insert the link text, swallowing an auto-inserted trailing `]]` so we never
  // produce `[[Note]]]]` (the same guard easy-links uses).
  private insert(
    editor: Editor,
    context: EditorSuggestContext,
    link: string,
  ): void {
    let end: EditorPosition = context.end;
    const tail = editor.getLine(end.line).slice(end.ch);
    if (tail.startsWith("]]")) {
      end = { line: end.line, ch: end.ch + 2 };
    }
    editor.replaceRange(link, context.start, end);
    editor.setCursor({
      line: context.start.line,
      ch: context.start.ch + link.length,
    });
  }
}

// --- helpers ----------------------------------------------------------------

function scoreOf(item: Item): number {
  return item.kind === "note" ? item.score : 0;
}

// Map a prepareFuzzySearch score (negative, 0 = best, more negative = worse) to a
// non-negative magnitude that is MONOTONIC in match quality and sign-independent.
// 1/(1+|score|) ∈ (0,1]: a perfect match (score 0) → 1, weaker matches → smaller.
// The caller further normalizes against the strongest candidate so the blended
// 60/40 fuzzy/semantic weighting stays meaningful.
function fuzzyMagnitude(score: number): number {
  return 1 / (1 + Math.abs(score));
}

// A cheap, allocation-light string hash (djb2). Only used to detect block changes
// for the context-embed cache, so collisions just cost an occasional re-embed.
function cheapHash(s: string): string {
  let h = 5381;
  for (let i = 0; i < s.length; i++) {
    h = ((h << 5) + h + s.charCodeAt(i)) | 0;
  }
  return (h >>> 0).toString(36);
}
