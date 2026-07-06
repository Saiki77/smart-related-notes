// NOTE: transformers.js/onnxruntime-web are no longer part of this bundle — the
// embedding engine lives in a terminable Web Worker (see worker/embed-worker.ts,
// which imports ort-shim.ts first for the same reason main.ts used to).
import {
  Plugin,
  PluginSettingTab,
  Setting,
  App,
  TFile,
  TAbstractFile,
  WorkspaceLeaf,
  MarkdownView,
  Notice,
  normalizePath,
  requestUrl,
  debounce,
  type Editor,
  type Debouncer,
} from "obsidian";
import {
  EmbeddingEngine,
  setOrtAssetLoader,
  setEmbedThreads,
  type OrtAssets,
  type DevicePref,
} from "./embeddings";
import {
  ORT_WEB_CDN,
  ORT_WEB_VERSION,
  ORT_GLUE_BYTES,
  ORT_WASM_BYTES,
} from "./ort-version";

// "Indexing speed" presets → WASM worker-thread count. Light is single-threaded
// (slowest); Fast uses a capped slice of the cores (fastest). A CPU/speed knob —
// the engine's memory is dominated by the loaded model, not the thread count.
export type IndexSpeed = "light" | "balanced" | "fast";
function threadsForSpeed(speed: IndexSpeed): number {
  const cores =
    typeof navigator !== "undefined" ? navigator.hardwareConcurrency || 4 : 4;
  if (speed === "light") return 1;
  if (speed === "fast") return Math.max(1, Math.min(cores - 2, 8));
  return Math.max(1, Math.min(Math.ceil(cores / 3), 4)); // balanced
}
import {
  IndexStore,
  stripMarkdown,
  type IndexStoreOptions,
  type VaultInsights,
} from "./index-store";
import { RelatedNotesView, VIEW_TYPE_RELATED } from "./view";
import { TitleIndex } from "./title-index";
import {
  glowPlugin,
  glowBridge,
  detectMentions,
  applyMentions,
  buildWikiLink,
} from "./link-glow";
import { SmartLinkSuggester } from "./link-suggester";

// The internal, undocumented suggester registry. Both our suggester and
// easy-links' unshift to the FRONT of this array to win the plain `[[` case. We
// touch it only through these narrow interfaces, wrapped in try/catch, and fall
// back to ordinary registration if the shape ever changes.
interface EditorSuggestManager {
  suggests: unknown[];
}
interface WorkspaceWithSuggest {
  editorSuggest?: EditorSuggestManager;
}

// --- settings ---------------------------------------------------------------

export interface RelatedNotesSettings {
  modelId: string;
  modelChosen: boolean; // first-run gate: no indexing until the user picks a model
  device: DevicePref;
  // WASM indexing speed/memory trade-off (worker-thread count). See threadsForSpeed.
  indexSpeed: IndexSpeed;
  // Minutes without any embedding work before the engine is unloaded to free its
  // memory (0 = never; fractional allowed — 0.5 is the "right after use" option).
  // It re-initialises transparently on the next demand.
  idleUnloadMinutes: number;
  topK: number;
  minSimilarity: number;
  // One-time flag: when false, onload lowers a pre-1.8.0 minSimilarity onto the new
  // mean-centered (floor-free) score scale, then sets it true.
  centeredScaleMigrated: boolean;
  excludeFolders: string; // comma- or newline-separated folder paths (index + links)
  excludeFoldersLinks: string; // additionally excluded from link suggestions only
  showSnippet: boolean;
  // --- multi-vector / ranking ---
  chunking: boolean; // master toggle for the chunk-level path
  structureInfluence: number; // B_MAX for the hybrid structural boost (0..0.3)
  showSummary: boolean; // keyphrase topic-label line (supersedes snippet when on)
  showRecency: boolean; // muted "edited Nd ago" line
  maxChunks: number; // body-chunk cap (advanced)
  shortlistSize: number; // Stage-1 -> Stage-2 funnel width (advanced)
  headingContext: boolean; // prefix each section's first chunk with a heading breadcrumb
  ideaInfluence: number; // 0..0.6 rank-time blend of idea-level similarity (0 = off; live, no re-embed)
  isolatedAreas: string; // activated tag namespaces (comma/newline) that form self-contained areas
  // --- linking (Features A + B) ---
  glowEnabled: boolean; // inline glow + 1-click link (Feature A)
  glowRestrictToLivePreview: boolean; // only decorate live preview
  glowAmbiguous: boolean; // glow a surface owned by 2+ notes (off = precision)
  autoLinkSubsequent: boolean; // idle auto-link of 2nd..Nth mentions (opt-in)
  suggesterEnabled: boolean; // smart `[[` suggester (Feature B)
  suggesterTakeOver: boolean; // move our suggester to the front of the `[[` popup
  // True once the user has EXPLICITLY toggled suggesterTakeOver. While false the
  // effective default is auto-derived against easy-links at layout-ready; once
  // true the stored value is honoured verbatim (never silently flipped).
  suggesterTakeOverUserSet: boolean;
  suggestNewNotes: boolean; // propose "create new note" rows
  newNoteMinSimilarity: number; // confidence floor for a new-note proposal
}

export const DEFAULT_SETTINGS: RelatedNotesSettings = {
  // A SYMMETRIC sentence-similarity model — the right tool for "which notes are
  // alike". (multilingual-e5-* are RETRIEVAL models, tuned for short-query →
  // document search, and rank note-to-note similarity poorly.)
  modelId: "Xenova/paraphrase-multilingual-MiniLM-L12-v2",
  modelChosen: false, // fresh installs start gated; migrated to true for existing users
  device: "auto",
  indexSpeed: "balanced",
  idleUnloadMinutes: 15,
  topK: 12,
  // Mean-centering (1.8.0) removes the anisotropy noise floor, so scores sit on a
  // lower, floor-free scale where ~0.2 cleanly separates topically-related notes from
  // the now-near-zero unrelated ones. (Pre-1.8.0 vaults are migrated down on load.)
  minSimilarity: 0.2,
  centeredScaleMigrated: false,
  excludeFolders: "",
  excludeFoldersLinks: "",
  showSnippet: true,
  chunking: true,
  structureInfluence: 0.15,
  showSummary: true,
  showRecency: false,
  maxChunks: 48,
  shortlistSize: 60,
  headingContext: true,
  ideaInfluence: 0.3,
  isolatedAreas: "",
  // Precision-first, low-risk behaviors ON; riskier ones OFF. suggesterTakeOver's
  // effective default is computed against easy-links at layout-ready (see
  // resolveSuggesterTakeOver) so we don't fight it; the stored value is the
  // user's explicit override once they toggle it.
  glowEnabled: true,
  glowRestrictToLivePreview: true,
  glowAmbiguous: false,
  autoLinkSubsequent: false,
  suggesterEnabled: true,
  suggesterTakeOver: true,
  suggesterTakeOverUserSet: false,
  suggestNewNotes: true,
  newNoteMinSimilarity: 0.45,
};

// A few vetted model ids surfaced as a dropdown so users don't have to memorise
// HF repo paths. Any other id can still be typed in the text field below.
// Paraphrase (symmetric) models first — they judge note-to-note similarity far
// better than retrieval models for this use case.
const MODEL_OPTIONS: Record<string, string> = {
  "Xenova/paraphrase-multilingual-MiniLM-L12-v2":
    "MiniLM-L12 multilingual — best for related notes, fast (default)",
  "Xenova/paraphrase-multilingual-mpnet-base-v2":
    "mpnet-base multilingual — strongest matches, larger & slower",
  "Xenova/multilingual-e5-small":
    "e5-small: retrieval/search model, weaker for note similarity",
  "jinaai/jina-embeddings-v5-text-nano-text-matching":
    "jina-v5-nano: best quality (whole-note), ~250MB download, non-commercial",
};

// One-click presets. "Balanced" is light and fast; "Best quality" uses a larger
// model and more context for the strongest matches. Each applies to the relevant
// settings; the index rebuilds automatically if the model changes.
type ProfileName = "balanced" | "best";
const PROFILES: Record<ProfileName, Partial<RelatedNotesSettings>> = {
  balanced: {
    modelId: "Xenova/paraphrase-multilingual-MiniLM-L12-v2",
    device: "auto",
    topK: 8,
    minSimilarity: 0.2,
    showSnippet: true,
    chunking: true,
    showSummary: true,
    structureInfluence: 0.15,
    maxChunks: 48,
    headingContext: true,
    ideaInfluence: 0.3,
  },
  best: {
    modelId: "Xenova/paraphrase-multilingual-mpnet-base-v2",
    device: "auto",
    topK: 20,
    minSimilarity: 0.2,
    showSnippet: true,
    chunking: true,
    showSummary: true,
    structureInfluence: 0.2,
    maxChunks: 64,
    headingContext: true,
    ideaInfluence: 0.35,
  },
};

// Length of the muted snippet shown on each card.
const SNIPPET_CHARS = 160;
// Max cached card snippets (LRU). ~500 × 160 chars ≈ 0.2 MB worst case.
const SNIPPET_CACHE_CAP = 500;

// The ORT runtime pair every worker spawn needs. Present locally in dev builds
// (gen-ort.mjs copies them next to main.js) and in legacy full-zip installs;
// community-directory installs get only main.js/manifest/styles, so the plugin
// downloads the pair ONCE from a pinned CDN into its own ort/ folder and serves
// it locally from then on (offline after first run). Any pair — local or just
// downloaded — is validated against the EXACT byte sizes gen-ort.mjs pinned
// (ORT_GLUE_BYTES/ORT_WASM_BYTES) before use: the pair must be the same build
// the bundled transformers glue expects, or ORT init fails with cryptic
// mixed-build errors. Size mismatch (torn cache write, files from an install
// carrying a different onnxruntime-web, CDN/proxy garbage) → re-download.
const ORT_GLUE_FILE = "ort-wasm-simd-threaded.asyncify.mjs";
const ORT_WASM_FILE = "ort-wasm-simd-threaded.asyncify.wasm";
// CDN sources tried in order; both serve the pinned npm package.
const ORT_CDNS = [
  ORT_WEB_CDN,
  `https://unpkg.com/onnxruntime-web@${ORT_WEB_VERSION}/dist/`,
];

// ---------------------------------------------------------------------------

export default class RelatedNotesPlugin extends Plugin {
  declare settings: RelatedNotesSettings;
  store!: IndexStore;
  // Held by the plugin and shared by BOTH link features (glow + suggester).
  titleIndex!: TitleIndex;
  private engine!: EmbeddingEngine;

  // The smart `[[` suggester instance (Feature B), so settings changes can
  // re-assert/remove its precedence in the popup.
  private suggester: SmartLinkSuggester | null = null;
  // True while our suggester sits at the FRONT of the manager's suggests array.
  private suggesterPrioritised = false;

  // Coalesces TitleIndex rebuilds across bulk-edit bursts (aliases/titles change
  // without a vector re-embed, so this is independent of debouncedUpdate).
  private debouncedTitleRefresh!: Debouncer<[], void>;
  // Idle auto-link-subsequent pass for the active file (opt-in; ~3s after the
  // last edit). Lighter + separate from the 20s re-embed debounce.
  private debouncedAutoLink!: Debouncer<[string], void>;

  // The model id / device preference the current engine was built for. Compared
  // against settings on save to decide whether a re-embed is actually needed —
  // critically NOT against engine.device (the RESOLVED device), which would never
  // equal the "auto" preference and would rebuild on every unrelated save.
  private appliedModelId!: string;
  private appliedDevicePref!: DevicePref;
  private appliedIndexSpeed!: IndexSpeed;
  // The embedding-shape settings the current index was built for. Changing any of
  // them alters WHAT is embedded per note, so they trigger a rebuild like a model
  // change. (showSummary also changes whether chunk text is persisted.)
  private appliedChunking!: boolean;
  private appliedMaxChunks!: number;
  private appliedShowSummary!: boolean;
  private appliedHeadingContext!: boolean;
  // Guards the engine-swap + rebuild path against re-entrancy.
  private swapping = false;

  // Coalesce vault-change bursts (bulk edits, sync) into batched re-embeds.
  private debouncedUpdate!: Debouncer<[TFile], void>;
  // Pending changed files, drained by the debounced updater above.
  private dirty = new Set<string>();

  async onload(): Promise<void> {
    const saved = (await this.loadData()) as Partial<RelatedNotesSettings> | null;
    this.settings = Object.assign({}, DEFAULT_SETTINGS, saved);

    // First-run gate migration: an install with settings already saved is treated as
    // having chosen a model, so only a truly fresh install (no data.json) starts gated.
    if (saved && saved.modelChosen === undefined) this.settings.modelChosen = true;

    // One-time recalibration for 1.8.0 mean-centering: the old scores carried an
    // anisotropy "noise floor" (~0.4 for unrelated notes), so users had minSimilarity
    // tuned up around there. Centering removes that floor, so a 0.4 cutoff now hides
    // almost everything — lower it onto the new scale once (only ever lowers it).
    if (!this.settings.centeredScaleMigrated) {
      this.settings.minSimilarity = Math.min(this.settings.minSimilarity, 0.2);
      this.settings.centeredScaleMigrated = true;
      await this.saveData(this.settings);
    }

    // How the engine obtains the ORT wasm runtime for each worker spawn: read
    // from the plugin's ort/ folder, downloading it there once when absent.
    // Lazy — nothing is read or downloaded until the first embed needs it.
    setOrtAssetLoader(() => this.loadOrtAssets());

    // Apply the WASM thread count BEFORE the engine's first init (configureEnv reads it).
    setEmbedThreads(threadsForSpeed(this.settings.indexSpeed));
    this.engine = new EmbeddingEngine(this.settings.modelId, this.settings.device);
    this.appliedModelId = this.settings.modelId;
    this.appliedDevicePref = this.settings.device;
    this.appliedIndexSpeed = this.settings.indexSpeed;
    this.appliedChunking = this.settings.chunking;
    this.appliedMaxChunks = this.settings.maxChunks;
    this.appliedShowSummary = this.settings.showSummary;
    this.appliedHeadingContext = this.settings.headingContext;
    this.store = new IndexStore(
      this.app,
      this.engine,
      this.pluginDir(),
      this.storeOptions(),
    );
    // Lazy keyphrase labels: when a label finishes computing on first demand, the
    // store asks the view to re-render the affected card(s). requestRender is itself
    // debounced (300ms) in the view, so a batch of resolving labels collapses into
    // one render pass — mirroring getSnippet's coalescing.
    this.store.setRenderHook(() => this.getView()?.requestRender());
    // Idle auto-unload: free the engine's memory after a quiet period.
    this.applyEngineIdlePolicy();

    // The precision backbone for both link features.
    this.titleIndex = new TitleIndex(this.app, () => this.linkExcludedFolders());

    this.registerView(VIEW_TYPE_RELATED, (leaf) => new RelatedNotesView(leaf, this));

    this.addRibbonIcon("sparkles", "Smart related notes", () => {
      void this.activateView();
    });

    this.addCommand({
      id: "open-view",
      name: "Open the panel",
      callback: () => {
        void this.activateView();
      },
    });

    this.addCommand({
      id: "rebuild-index",
      name: "Rebuild the index",
      callback: () => {
        void this.rebuildIndex();
      },
    });

    this.addCommand({
      id: "vault-insights",
      name: "Vault insights (suggested links, orphans, duplicates)",
      callback: () => {
        void this.generateVaultInsights();
      },
    });

    // --- Feature A: inline glow + 1-click link -------------------------------
    // Seed the module-scoped bridge the ViewPlugin reads, then register the CM6
    // extension. @codemirror/* is externalized so this uses Obsidian's singleton.
    this.syncGlowBridge();
    glowBridge.insert = (range) => this.insertLinkAtRange(range);
    this.registerEditorExtension([glowPlugin]);

    this.addCommand({
      id: "link-all-mentions",
      name: "Link all unlinked mentions in this note",
      editorCallback: (editor, view) => {
        this.linkAllMentions(editor, view.file);
      },
    });

    // --- Feature B: smart `[[` suggester -------------------------------------
    this.suggester = new SmartLinkSuggester(this);
    this.registerEditorSuggest(this.suggester);

    this.addSettingTab(new RelatedNotesSettingTab(this.app, this));

    // Re-rank the panel when the active note changes (the view debounces
    // internally), and stamp the active file path onto the glow bridge so the CM6
    // ViewPlugin excludes self-surfaces + scopes its skip ranges to this note.
    this.registerEvent(
      this.app.workspace.on("active-leaf-change", () => {
        this.getView()?.requestRender();
        this.syncActivePath();
        // Force every glow ViewPlugin to rebuild now, so the switched-to note
        // deterministically uses its OWN alternation/self-exclusion instead of
        // waiting for an incidental viewport/doc update to refresh it.
        this.app.workspace.updateOptions();
      }),
    );
    this.syncActivePath();

    // --- TitleIndex invalidation (independent of the embedding index) --------
    // Titles/aliases change the glow alternation + suggester surfaces but NOT the
    // embeddings, so this is its own debounced refresh. Coalesces bulk bursts.
    this.debouncedTitleRefresh = debounce(() => this.refreshTitleIndex(), 1500, false);

    // --- idle auto-link-subsequent (opt-in) ----------------------------------
    this.debouncedAutoLink = debounce(
      (path: string) => this.autoLinkSubsequent(path),
      3000,
      false,
    );

    // --- incremental index maintenance ---------------------------------------
    // 20s idle before re-embedding a changed note: typing (and the short pauses
    // while typing) never kicks off embeddings — only a real edit pause does.
    this.debouncedUpdate = debounce(() => void this.flushDirty(), 20000, false);

    this.registerEvent(
      this.app.vault.on("modify", (file) => {
        if (file instanceof TFile && file.extension === "md") {
          this.dirty.add(file.path);
          this.debouncedUpdate(file);
          // Title text can change on a modify; refresh the (cheap) title index too.
          this.debouncedTitleRefresh();
          // Idle auto-link the active note's later mentions (opt-in; cursor-aware,
          // re-validating) only when this is the active file.
          if (
            this.settings.autoLinkSubsequent &&
            file.path === this.app.workspace.getActiveFile()?.path
          ) {
            this.debouncedAutoLink(file.path);
          }
        }
      }),
    );
    this.registerEvent(
      this.app.vault.on("create", (file) => {
        if (file instanceof TFile && file.extension === "md") {
          this.dirty.add(file.path);
          this.debouncedUpdate(file);
          this.titleIndex.markDirty();
          this.debouncedTitleRefresh();
        }
      }),
    );
    this.registerEvent(
      this.app.vault.on("delete", (file: TAbstractFile) => {
        this.store.removeFile(file.path);
        this.snippetCache.delete(file.path);
        this.titleIndex.markDirty();
        this.suggester?.invalidateAliasCache(file.path);
        this.debouncedTitleRefresh();
        this.getView()?.requestRender();
      }),
    );
    this.registerEvent(
      this.app.vault.on("rename", (file, oldPath) => {
        this.snippetCache.delete(oldPath);
        if (file instanceof TFile && file.extension === "md") {
          this.store.renameFile(oldPath, file);
        } else {
          this.store.removeFile(oldPath);
        }
        this.titleIndex.markDirty();
        this.suggester?.invalidateAliasCache(oldPath);
        if (file instanceof TFile) this.suggester?.invalidateAliasCache(file.path);
        this.debouncedTitleRefresh();
        this.getView()?.requestRender();
      }),
    );
    // Aliases change without bumping mtime, so drop caches on the metadata
    // 'changed' event (the exact reason easy-links does too).
    this.registerEvent(
      this.app.metadataCache.on("changed", (file) => {
        this.titleIndex.markDirty();
        this.suggester?.invalidateAliasCache(file.path);
        this.debouncedTitleRefresh();
      }),
    );

    // Load (or build) the index once the layout is ready, so the vault file list
    // and metadata cache are fully populated first. Also resolve the suggester
    // take-over default against easy-links and apply precedence now that the core
    // suggester is registered.
    this.app.workspace.onLayoutReady(() => {
      void this.bootstrapIndex();
      this.refreshTitleIndex();
      this.resolveSuggesterTakeOver();
      this.applySuggesterPrecedence();
    });
  }

  onunload(): void {
    // Cancel armed debouncers so a pending timer (the 20s re-embed flush
    // especially) can't fire into the unloaded plugin.
    this.debouncedUpdate?.cancel();
    this.debouncedTitleRefresh?.cancel();
    this.debouncedAutoLink?.cancel();
    // Stop any in-flight build/flush/label loop, then terminate the engine's
    // worker realm (terminal: post-dispose calls reject instead of respawning).
    this.store?.close();
    void this.engine?.dispose();
    // The registered view + editor extension + suggest registration are torn down
    // by Obsidian; we only undo the manual precedence reorder we made.
    this.removeSuggesterPrecedence();
  }

  // Push the idle-unload policy (and the cache-trim hook) onto the CURRENT
  // engine. Called at load, after every engine swap, and on settings saves.
  private applyEngineIdlePolicy(): void {
    this.engine.onIdleUnload = () => {
      // The engine just unloaded — drop the warm ranking caches too, so an idle
      // plugin holds as little as possible. Both repopulate on the next rank
      // (the dequant re-derivation is bit-identical).
      this.store.trimIdleCaches();
    };
    const min = this.settings.idleUnloadMinutes;
    this.engine.setIdleUnload(min > 0 ? min * 60_000 : null);
  }

  // --- ORT runtime assets ------------------------------------------------------
  // Produce the glue+wasm pair a worker spawn runs, called by the engine per
  // spawn (the wasmBinary is transferred into the worker, so every call returns
  // fresh buffers). Resolution order:
  //   1) The plugin's local ort/ folder (dev builds, legacy full-zip installs,
  //      cached downloads) when BOTH files match the pinned byte sizes.
  //   2) A one-time download (~24 MB) from a version-pinned CDN, size-validated
  //      the same way, then cached into ort/ (best effort) so every later spawn
  //      — and every later session — is served locally and works offline.
  private async loadOrtAssets(): Promise<OrtAssets> {
    const adapter = this.app.vault.adapter;
    const dir = normalizePath(`${this.pluginDir()}/ort`);
    const gluePath = normalizePath(`${dir}/${ORT_GLUE_FILE}`);
    const wasmPath = normalizePath(`${dir}/${ORT_WASM_FILE}`);

    // 1) Local pair, size-validated against the pinned build.
    try {
      if ((await adapter.exists(gluePath)) && (await adapter.exists(wasmPath))) {
        const [glueBuf, wasmBinary] = await Promise.all([
          adapter.readBinary(gluePath),
          adapter.readBinary(wasmPath),
        ]);
        if (
          glueBuf.byteLength === ORT_GLUE_BYTES &&
          wasmBinary.byteLength === ORT_WASM_BYTES
        ) {
          return { glueText: new TextDecoder().decode(glueBuf), wasmBinary };
        }
        // Wrong sizes: a torn/interrupted cache write, or files from an install
        // carrying a different onnxruntime-web build — unusable with the glue
        // bundled into THIS version. Fall through and replace them.
        console.warn(
          `[related-notes] local ORT runtime does not match the bundled build (glue ${glueBuf.byteLength}/${ORT_GLUE_BYTES} B, wasm ${wasmBinary.byteLength}/${ORT_WASM_BYTES} B); re-downloading`,
        );
      }
    } catch (e) {
      console.warn("[related-notes] local ORT runtime unreadable; downloading", e);
    }

    // 2) One-time pinned-CDN download, cached for next time.
    const failures: string[] = [];
    for (const base of ORT_CDNS) {
      try {
        const [glue, wasm] = await Promise.all([
          requestUrl({ url: base + ORT_GLUE_FILE }),
          requestUrl({ url: base + ORT_WASM_FILE }),
        ]);
        const glueBinary = glue.arrayBuffer;
        const wasmBinary = wasm.arrayBuffer;
        // Validate BEFORE caching or serving: a proxy/captive-portal can return
        // 200 with garbage, which must not poison the cache or reach the worker.
        if (
          glueBinary.byteLength !== ORT_GLUE_BYTES ||
          wasmBinary.byteLength !== ORT_WASM_BYTES
        ) {
          failures.push(
            `${base}: unexpected sizes (glue ${glueBinary.byteLength}/${ORT_GLUE_BYTES} B, wasm ${wasmBinary.byteLength}/${ORT_WASM_BYTES} B)`,
          );
          continue;
        }
        const glueText = new TextDecoder().decode(glueBinary);
        try {
          if (!(await adapter.exists(dir))) await adapter.mkdir(dir);
          await adapter.writeBinary(gluePath, glueBinary);
          await adapter.writeBinary(wasmPath, wasmBinary);
        } catch (e) {
          // Cache write failed (read-only dir?): still usable this session, and
          // a torn write is caught by the size check on the next launch.
          console.warn("[related-notes] could not cache the ORT runtime", e);
        }
        return { glueText, wasmBinary };
      } catch (e) {
        failures.push(`${base}: ${e instanceof Error ? e.message : String(e)}`);
      }
    }
    // All sources failed: name them (this is the ONNX runtime, a separate
    // download from the model weights, so it can fail even when huggingface.co
    // is reachable — and vice versa).
    throw new Error(
      `could not obtain the ONNX runtime (${ORT_WASM_FILE}) — ${failures.join("; ")}. ` +
        "Offline, or a firewall/proxy/antivirus blocks these URLs.",
    );
  }

  private pluginDir(): string {
    return (
      this.manifest.dir ??
      `${this.app.vault.configDir}/plugins/smart-related-notes`
    );
  }

  // --- index lifecycle -------------------------------------------------------
  private async bootstrapIndex(): Promise<void> {
    // First-run gate: do not download a model or build the index until the user has
    // explicitly picked one in settings (so a fresh install never auto-downloads).
    if (!this.settings.modelChosen) {
      new Notice(
        "Smart Related Notes: open Settings → Smart Related Notes and pick an embedding model to start indexing.",
        10000,
      );
      this.getView()?.requestRender();
      return;
    }
    const loaded = await this.store.load();
    this.getView()?.requestRender();
    if (!loaded) {
      await this.store.build();
      this.getView()?.requestRender();
    }
  }

  // Manual "Rebuild index" command/button: FORCE a full re-embed of every note
  // (not the cheap mtime-reuse build), so it always reflects the current model and
  // settings instead of finishing instantly with the old vectors.
  async rebuildIndex(): Promise<void> {
    if (!this.settings.modelChosen) {
      new Notice("Smart Related Notes: pick an embedding model in Settings first.", 8000);
      return;
    }
    await this.store.build(undefined, true);
    this.getView()?.requestRender();
  }

  // Generate (or refresh) a "Vault Insights" report note: the strongest related-but-
  // unlinked pairs to connect, orphan notes (with their closest relative), near-
  // duplicates, and the oldest-edited notes. Computed from the live semantic index.
  private async generateVaultInsights(): Promise<void> {
    const path = "Vault Insights (Smart Related Notes).md";
    new Notice("Smart Related Notes: computing vault insights...");
    let insights: VaultInsights;
    try {
      insights = this.store.computeInsights(new Set([path]));
    } catch (e) {
      new Notice(`Vault insights failed: ${(e as Error).message}`);
      return;
    }
    const md = this.renderInsights(insights);
    const existing = this.app.vault.getAbstractFileByPath(path);
    let file: TFile;
    if (existing instanceof TFile) {
      await this.app.vault.modify(existing, md);
      file = existing;
    } else {
      file = await this.app.vault.create(path, md);
    }
    await this.app.workspace.getLeaf(false).openFile(file);
    new Notice(
      `Vault insights: ${insights.suggestedLinks.length} link suggestions, ${insights.orphans.length} orphans.`,
    );
  }

  private renderInsights(ins: VaultInsights): string {
    const wl = (p: string): string => `[[${p.replace(/\.md$/i, "")}]]`;
    const pct = (s: number | undefined): string => `${Math.round((s ?? 0) * 100)}%`;
    const ymd = (ms: number): string => {
      const d = new Date(ms);
      const p2 = (n: number): string => String(n).padStart(2, "0");
      return `${d.getFullYear()}-${p2(d.getMonth() + 1)}-${p2(d.getDate())}`;
    };
    const L: string[] = [];
    L.push("# Vault Insights", "");
    L.push(
      `Generated by Smart Related Notes across ${ins.total} indexed notes. Re-run the "Vault insights" command to refresh.`,
      "",
    );

    L.push(`## Suggested links (${ins.suggestedLinks.length})`, "");
    L.push("Strongly related notes that are not linked yet. Consider connecting them.", "");
    if (ins.suggestedLinks.length === 0) L.push("None above threshold.");
    for (const s of ins.suggestedLinks) {
      L.push(`- ${wl(s.from)} and ${wl(s.to)}  (${pct(s.score)})`);
    }
    L.push("");

    L.push(`## Suggested tags (${ins.suggestedTags.length})`, "");
    L.push(
      "Notes whose semantic neighbours share a tag they are missing. Inferred from similarity, not rules.",
      "",
    );
    if (ins.suggestedTags.length === 0) L.push("None.");
    for (const s of ins.suggestedTags) {
      L.push(`- ${wl(s.path)} -> #${s.tag}  (${s.support}/${s.neighbors} neighbours)`);
    }
    L.push("");

    L.push(`## Orphan notes (${ins.orphans.length})`, "");
    L.push("Notes with no links in or out. Each shows its closest relative to start from.", "");
    if (ins.orphans.length === 0) L.push("None. Every note is connected.");
    for (const o of ins.orphans) {
      L.push(
        `- ${wl(o.path)}${o.closest ? `  (closest: ${wl(o.closest)} ${pct(o.closestScore)})` : ""}`,
      );
    }
    L.push("");

    L.push(`## Possibly duplicate or very similar (${ins.nearDuplicates.length})`, "");
    L.push("Near-identical pairs. Consider merging or cross-linking.", "");
    if (ins.nearDuplicates.length === 0) L.push("None above threshold.");
    for (const d of ins.nearDuplicates) {
      L.push(`- ${wl(d.a)} and ${wl(d.b)}  (${pct(d.score)})`);
    }
    L.push("");

    L.push("## Stale notes (oldest edited)", "");
    for (const s of ins.stale) L.push(`- ${wl(s.path)}  (${ymd(s.mtime)})`);
    L.push("");
    return L.join("\n");
  }

  private async flushDirty(): Promise<void> {
    // First-run gate: never embed (which would download the default model) before the
    // user has picked a model. Picking one runs a full build over all notes anyway.
    if (!this.settings.modelChosen) {
      this.dirty.clear();
      return;
    }
    if (this.dirty.size === 0) return;
    const paths = Array.from(this.dirty);
    this.dirty.clear();
    for (const path of paths) {
      const file = this.app.vault.getAbstractFileByPath(path);
      if (file instanceof TFile) await this.store.updateFile(file);
    }
    this.getView()?.requestRender();
  }

  // --- link features: glow bridge + insertion --------------------------------
  // Push the current settings + titleIndex onto the module-scoped bridge the CM6
  // ViewPlugin reads. Called on load and on every settings change.
  private syncGlowBridge(): void {
    glowBridge.enabled = this.settings.glowEnabled;
    glowBridge.restrictToLivePreview = this.settings.glowRestrictToLivePreview;
    glowBridge.glowAmbiguous = this.settings.glowAmbiguous;
    // When auto-link-subsequent is on we glow every occurrence as a preview of what
    // will be linked; otherwise only the first unlinked occurrence glows.
    glowBridge.glowAll = this.settings.autoLinkSubsequent;
    glowBridge.titleIndex = this.titleIndex;
    // Semantic context gate: only glow a mention whose target fits the note's topic
    // (a common word like "analysis" no longer glows the math note in a security note).
    glowBridge.contextGate = (activePath, targetPath) =>
      this.store.glowAllowed(activePath, targetPath);
    // The app handle lets buildGlow resolve each EditorView's OWN file (split-pane
    // correct) instead of relying on the single global active path.
    glowBridge.app = this.app;
  }

  // Stamp the active markdown file's path onto the glow bridge so the ViewPlugin
  // scopes self-exclusion + skip ranges to the right note.
  private syncActivePath(): void {
    glowBridge.activePath = this.app.workspace.getActiveFile()?.path ?? null;
  }

  // Rebuild the title index now (its rebuild is lazy, so we just mark dirty and
  // force a recompute through a resolve-style touch), then force every editor to
  // re-derive its glow decorations with the fresh alternation.
  private refreshTitleIndex(): void {
    this.titleIndex.markDirty();
    // Force the CM6 ViewPlugin(s) to rebuild so the glow reflects the new titles.
    this.app.workspace.updateOptions();
  }

  // Insert a wikilink for a clicked glow range. Resolved through the TitleIndex
  // (so an alias keeps its display text) and applied via the active editor's
  // replaceRange, which joins Obsidian undo + link bookkeeping and re-fires the
  // metadataCache 'changed' event (refreshing both indexes).
  private insertLinkAtRange(range: {
    from: number;
    to: number;
    surface: string;
  }): void {
    const view = this.app.workspace.getActiveViewOfType(MarkdownView);
    const editor = view?.editor;
    if (!editor) return;
    try {
      const resolved = this.titleIndex.resolve(range.surface);
      if (!resolved) return;
      const fromPos = editor.offsetToPos(range.from);
      const toPos = editor.offsetToPos(range.to);
      // Re-validate the live text against the surface before mutating.
      if (editor.getRange(fromPos, toPos) !== range.surface) return;
      const link = buildWikiLink(resolved.file, range.surface, this.titleIndex);
      editor.replaceRange(link, fromPos, toPos);
    } catch (e) {
      console.warn("[related-notes] glow link insertion failed", e);
    }
  }

  // Command: link EVERY surviving unlinked mention in the active note, across all
  // target notes, in one undo step (descending offsets, re-validated per range).
  private linkAllMentions(editor: Editor, file: TFile | null): void {
    if (!file) return;
    try {
      const mentions = detectMentions(
        editor.getValue(),
        this.titleIndex,
        file.path,
        { all: true, allowAmbiguous: this.settings.glowAmbiguous },
      );
      const n = applyMentions(editor, mentions, this.titleIndex);
      new Notice(
        n === 0
          ? "Related notes: no unlinked mentions to link."
          : `Related notes: linked ${n} mention${n === 1 ? "" : "s"}.`,
      );
    } catch (e) {
      console.warn("[related-notes] link-all failed", e);
      new Notice("Related notes: linking failed. See the console.");
    }
  }

  // Idle auto-link of the 2nd..Nth mentions (opt-in). For each target that ALREADY
  // has at least one existing `[[link]]` in the note, link its remaining surviving
  // occurrences. Re-validates each range against the CURRENT text and never
  // touches the range the cursor occupies, so it can't race the typist.
  private autoLinkSubsequent(path: string): void {
    if (!this.settings.autoLinkSubsequent) return;
    const active = this.app.workspace.getActiveFile();
    if (!active || active.path !== path) return;
    const view = this.app.workspace.getActiveViewOfType(MarkdownView);
    const editor = view?.editor;
    if (!editor || view.file?.path !== path) return;

    try {
      // Targets the note already links to — only THOSE get their later mentions
      // auto-linked (so we never invent a first link silently).
      const cache = this.app.metadataCache.getFileCache(active);
      const linkedTargets = new Set<string>();
      for (const l of cache?.links ?? []) {
        const dest = this.app.metadataCache.getFirstLinkpathDest(
          l.link,
          active.path,
        );
        if (dest) linkedTargets.add(dest.path);
      }
      if (linkedTargets.size === 0) return;

      const cursorOffset = editor.posToOffset(editor.getCursor());
      const all = detectMentions(editor.getValue(), this.titleIndex, active.path, {
        all: true,
        allowAmbiguous: this.settings.glowAmbiguous,
      });
      const toLink = all.filter((m) => {
        const resolved = this.titleIndex.resolve(m.surface);
        if (!resolved || !linkedTargets.has(resolved.file.path)) return false;
        // Never auto-link the range the cursor is currently inside.
        if (cursorOffset >= m.from && cursorOffset <= m.to) return false;
        return true;
      });
      applyMentions(editor, toLink, this.titleIndex);
    } catch (e) {
      console.warn("[related-notes] auto-link-subsequent failed", e);
    }
  }

  // --- suggester precedence (coexist with easy-links) ------------------------
  // Resolve the EFFECTIVE take-over default ONCE, before the user has expressed an
  // intent: when easy-links' smart suggester is active we default to NOT taking
  // over (so we don't fight it); otherwise we take over. The moment the user
  // toggles the setting (suggesterTakeOverUserSet=true) we never re-derive — their
  // explicit choice is honoured on every launch, even with easy-links present.
  private resolveSuggesterTakeOver(): void {
    if (this.settings.suggesterTakeOverUserSet) return;
    if (this.easyLinksSmartSuggesterActive()) {
      // Defer to easy-links by default; the user can still flip the toggle.
      this.settings.suggesterTakeOver = false;
    }
  }

  // Detect whether easy-links is installed AND its smart `[[` suggester is on, via
  // its public plugin instance settings. Read defensively (undocumented shape).
  private easyLinksSmartSuggesterActive(): boolean {
    try {
      const plugins = (
        this.app as unknown as {
          plugins?: {
            plugins?: Record<string, { settings?: { smartSuggester?: boolean } }>;
          };
        }
      ).plugins?.plugins;
      const easy = plugins?.["easy-links"];
      return easy?.settings?.smartSuggester === true;
    } catch {
      return false;
    }
  }

  // Move our suggester to the FRONT of the manager's suggests array so it wins the
  // plain `[[` case. Undocumented internal API: narrow typed interface + try/catch.
  private applySuggesterPrecedence(): void {
    if (!this.settings.suggesterEnabled || !this.settings.suggesterTakeOver) return;
    if (this.suggesterPrioritised) return;
    const suggester = this.suggester;
    if (!suggester) return;
    try {
      const manager = (this.app.workspace as unknown as WorkspaceWithSuggest)
        .editorSuggest;
      const list = manager?.suggests;
      if (!Array.isArray(list)) return;
      const idx = list.indexOf(suggester);
      if (idx !== -1) list.splice(idx, 1);
      list.unshift(suggester);
      this.suggesterPrioritised = true;
    } catch {
      // Internal shape changed — fall back to ordinary registration order.
    }
  }

  // Reverse applySuggesterPrecedence: move our instance from the front back to the
  // END (keeping it REGISTERED and inert when disabled — onTrigger returns null —
  // rather than unregistering it).
  private removeSuggesterPrecedence(): void {
    if (!this.suggesterPrioritised) return;
    const suggester = this.suggester;
    if (!suggester) {
      this.suggesterPrioritised = false;
      return;
    }
    try {
      const manager = (this.app.workspace as unknown as WorkspaceWithSuggest)
        .editorSuggest;
      const list = manager?.suggests;
      if (Array.isArray(list)) {
        const idx = list.indexOf(suggester);
        if (idx !== -1) {
          list.splice(idx, 1);
          list.push(suggester);
        }
      }
    } catch {
      // Nothing to restore if the internal shape changed.
    }
    this.suggesterPrioritised = false;
  }

  // --- view plumbing ---------------------------------------------------------
  getView(): RelatedNotesView | null {
    const leaves = this.app.workspace.getLeavesOfType(VIEW_TYPE_RELATED);
    const view = leaves[0]?.view;
    return view instanceof RelatedNotesView ? view : null;
  }

  // Open the view in the LEFT sidebar (reuse an existing one if present), then
  // reveal it so it is the active tab.
  async activateView(): Promise<void> {
    const { workspace } = this.app;
    let leaf: WorkspaceLeaf | null = workspace.getLeavesOfType(VIEW_TYPE_RELATED)[0] ?? null;
    if (!leaf) {
      leaf = workspace.getLeftLeaf(false);
      if (!leaf) return;
      await leaf.setViewState({ type: VIEW_TYPE_RELATED, active: true });
    }
    await workspace.revealLeaf(leaf);
  }

  // --- snippet ---------------------------------------------------------------
  // The 1–2 line muted preview shown on each card. A synchronous read of file
  // content isn't possible here, so we cache by mtime and kick off an async read
  // when missing, returning any stale value meanwhile so the card isn't blank.
  // The async resolution schedules ONE debounced re-render (via the view), so N
  // cards resolving on first paint collapse into a single extra render pass
  // instead of a render storm.
  getSnippet(file: TFile): string {
    const cached = this.snippetCache.get(file.path);
    if (cached && cached.mtime === file.stat.mtime) {
      // Refresh to MRU (Map insertion order is the LRU order, as in DequantCache).
      this.snippetCache.delete(file.path);
      this.snippetCache.set(file.path, cached);
      return cached.text;
    }
    void this.app.vault.cachedRead(file).then((content) => {
      const text = stripMarkdown(content).slice(0, SNIPPET_CHARS);
      this.snippetCache.set(file.path, { mtime: file.stat.mtime, text });
      // Bound the cache: it used to grow one entry per note EVER shown on a card
      // (only trimmed on delete/rename), i.e. unbounded over a long session.
      while (this.snippetCache.size > SNIPPET_CACHE_CAP) {
        const lru = this.snippetCache.keys().next().value;
        if (lru === undefined) break;
        this.snippetCache.delete(lru);
      }
      this.getView()?.requestRender();
    });
    return cached?.text ?? "";
  }

  private snippetCache = new Map<string, { mtime: number; text: string }>();

  // --- settings glue ---------------------------------------------------------
  private storeOptions(): IndexStoreOptions {
    return {
      excludeFolders: this.parseExcludeFolders(),
      topK: this.settings.topK,
      minSimilarity: this.settings.minSimilarity,
      chunking: this.settings.chunking,
      structureInfluence: this.settings.structureInfluence,
      maxChunks: this.settings.maxChunks,
      shortlistSize: this.settings.shortlistSize,
      showSummary: this.settings.showSummary,
      headingContext: this.settings.headingContext,
      ideaInfluence: this.settings.ideaInfluence,
      isolatedAreas: this.parseIsolatedAreas(),
    };
  }

  // Activated isolated-area tag namespaces, normalized (lowercased, leading # and any
  // sub-path dropped so "#goa/" or "goa/character" all activate the "goa" area).
  private parseIsolatedAreas(): string[] {
    return [
      ...new Set(
        this.settings.isolatedAreas
          .split(/[\n,]/)
          .map((s) => s.trim().replace(/^#/, "").split("/")[0].toLowerCase())
          .filter((s) => s.length > 0),
      ),
    ];
  }

  private parseExcludeFolders(): string[] {
    return this.settings.excludeFolders
      .split(/[\n,]/)
      .map((s) => s.trim().replace(/\/+$/, ""))
      .filter((s) => s.length > 0);
  }

  private parseLinkExcludeFolders(): string[] {
    return this.settings.excludeFoldersLinks
      .split(/[\n,]/)
      .map((s) => s.trim().replace(/\/+$/, ""))
      .filter((s) => s.length > 0);
  }

  // Folders the TitleIndex must skip: anything excluded from the index (so it is
  // never suggested as a link either) plus the link-only exclusions.
  private linkExcludedFolders(): string[] {
    return [...this.parseExcludeFolders(), ...this.parseLinkExcludeFolders()];
  }

  async saveSettings(): Promise<void> {
    await this.saveData(this.settings);
    this.store.updateOptions(this.storeOptions());
    // Idle-unload is a live knob: apply it without any engine rebuild. (The swap
    // path below re-applies it to the replacement engine.)
    this.applyEngineIdlePolicy();

    // --- linking toggles: pure UI/extension state, NEVER an embedding rebuild ---
    // These change what is glowed/suggested, not what is embedded, so they must
    // bypass the model/shape rebuild branch below. Push them to the bridge,
    // re-assert/remove suggester precedence, and force the glow ViewPlugin to
    // rebuild via updateOptions() (also called by refreshTitleIndex paths).
    this.syncGlowBridge();
    if (this.settings.suggesterEnabled && this.settings.suggesterTakeOver) {
      this.applySuggesterPrecedence();
    } else {
      this.removeSuggesterPrecedence();
    }
    this.app.workspace.updateOptions();

    // Only a MODEL or DEVICE-PREFERENCE change invalidates the stored vectors.
    // Compare against the last-APPLIED preferences (not the engine's resolved
    // device, which is "webgpu"/"wasm"/null and would never equal an "auto"
    // preference — the bug that made every slider drag rebuild the vault).
    const modelChanged = this.appliedModelId !== this.settings.modelId;
    const deviceChanged = this.appliedDevicePref !== this.settings.device;
    // Changing the thread count needs a fresh engine so configureEnv re-applies it.
    const indexSpeedChanged = this.appliedIndexSpeed !== this.settings.indexSpeed;
    // Embedding-SHAPE changes alter WHAT is embedded (chunking on/off, the chunk
    // cap) or whether chunk text is persisted (showSummary), so they too need a
    // full re-embed — but they keep the SAME engine.
    const shapeChanged =
      this.appliedChunking !== this.settings.chunking ||
      this.appliedMaxChunks !== this.settings.maxChunks ||
      this.appliedShowSummary !== this.settings.showSummary ||
      this.appliedHeadingContext !== this.settings.headingContext;

    if ((modelChanged || deviceChanged || indexSpeedChanged) && !this.swapping) {
      this.swapping = true;
      try {
        // Dispose the engine being REPLACED. Without this every model/device/
        // speed change orphaned the old ONNX session inside the wasm heap —
        // which can never shrink — so repeated settings changes ratcheted the
        // resident memory upward by a full model each time. (On the explicit
        // WebGPU pin this also releases the accumulated GPU memory.)
        const oldEngine = this.engine;
        oldEngine.onIdleUnload = null;
        void oldEngine.dispose();
        // Re-apply the thread count before the new engine's init reads it.
        setEmbedThreads(threadsForSpeed(this.settings.indexSpeed));
        this.engine = new EmbeddingEngine(
          this.settings.modelId,
          this.settings.device,
        );
        this.appliedModelId = this.settings.modelId;
        this.appliedDevicePref = this.settings.device;
        this.appliedIndexSpeed = this.settings.indexSpeed;
        this.appliedChunking = this.settings.chunking;
        this.appliedMaxChunks = this.settings.maxChunks;
        this.appliedShowSummary = this.settings.showSummary;
        this.appliedHeadingContext = this.settings.headingContext;
        // Swap the engine IN PLACE: the store (and the view's progress
        // subscription) stay valid, so the rebuild's status line stays live.
        this.store.setEngine(this.engine);
        this.applyEngineIdlePolicy();
        new Notice("Related notes: model changed, rebuilding index…");
        await this.store.build();
      } finally {
        this.swapping = false;
      }
    } else if (shapeChanged && !this.swapping) {
      this.swapping = true;
      try {
        this.appliedChunking = this.settings.chunking;
        this.appliedMaxChunks = this.settings.maxChunks;
        this.appliedShowSummary = this.settings.showSummary;
        this.appliedHeadingContext = this.settings.headingContext;
        new Notice("Related notes: chunking settings changed, rebuilding index…");
        // Same engine, but force a full re-embed so every note's chunk set matches
        // the new shape.
        await this.store.build(undefined, true);
      } finally {
        this.swapping = false;
      }
    }
    this.getView()?.requestRender();
  }
}

export class RelatedNotesSettingTab extends PluginSettingTab {
  plugin: RelatedNotesPlugin;
  private readonly debouncedSave: Debouncer<[], void>;

  constructor(app: App, plugin: RelatedNotesPlugin) {
    super(app, plugin);
    this.plugin = plugin;
    this.debouncedSave = debounce(() => void this.plugin.saveSettings(), 500, false);
  }

  // Apply a one-click preset, persist (the index rebuilds if the model changed),
  // then re-render so every control reflects the new values.
  private async applyProfile(name: ProfileName): Promise<void> {
    Object.assign(this.plugin.settings, PROFILES[name]);
    await this.plugin.saveSettings();
    this.render();
    new Notice(
      `Related notes: applied the ${name === "best" ? "Best quality" : "Balanced"} profile.`,
    );
  }

  display(): void {
    this.render();
  }

  // The tab body, callable directly (e.g. after applying a profile) without the
  // deprecated display() entry point.
  // Model cache manager: list the models transformers.js has downloaded into the
  // browser Cache ("transformers-cache"), with a Remove button to free space after
  // switching. Async-populates `host` (re-render after a removal).
  private renderModelCache(host: HTMLElement): void {
    host.empty();
    void (async () => {
      let ids: string[];
      try {
        const cache = await caches.open("transformers-cache");
        const keys = await cache.keys();
        const set = new Set<string>();
        for (const req of keys) {
          const m = /huggingface\.co\/([^/]+\/[^/]+)\/resolve\//.exec(req.url);
          if (m) set.add(m[1]);
        }
        ids = [...set].sort();
      } catch {
        host.createEl("div", {
          cls: "setting-item-description",
          text: "Could not read the model cache.",
        });
        return;
      }
      if (ids.length === 0) {
        host.createEl("div", {
          cls: "setting-item-description",
          text: "No models downloaded yet.",
        });
        return;
      }
      for (const id of ids) {
        const inUse = this.plugin.settings.modelId === id;
        new Setting(host).setName(inUse ? `${id}  (in use)` : id).addButton((b) =>
          b
            .setButtonText("Remove")
            .setClass("mod-warning")
            .onClick(async () => {
              try {
                const cache = await caches.open("transformers-cache");
                for (const req of await cache.keys()) {
                  if (req.url.includes(`/${id}/`)) await cache.delete(req);
                }
                new Notice(`Removed cached model: ${id}`);
              } catch {
                new Notice("Could not remove the cached model.");
              }
              this.renderModelCache(host);
            }),
        );
      }
    })();
  }

  private render(): void {
    const { containerEl } = this;
    containerEl.empty();

    // Short orientation blurb at the top.
    const intro = containerEl.createEl("div", {
      cls: "setting-item-description rn-settings-intro",
    });
    intro.createEl("strong", { text: "Smart Related Notes" });
    intro.appendText(
      " embeds your notes locally (offline, nothing leaves your vault) and ranks the rest by meaning. " +
        "Pick a model below to start indexing: MiniLM (default) is fast and light; jina-v5-nano is the " +
        "highest quality but a larger, non-commercial download. The first index build runs once after you " +
        "choose; switching models re-embeds the vault.",
    );
    if (!this.plugin.settings.modelChosen) {
      containerEl.createEl("div", {
        cls: "setting-item-description rn-settings-gate",
        text: "No model chosen yet. Indexing is paused until you select one in Model below.",
      });
    }

    new Setting(containerEl)
      .setName("Performance profile")
      .setDesc(
        "Quick presets. Balanced is lighter and faster; Best quality uses a larger model and more context for the strongest matches.",
      )
      .addButton((b) =>
        b
          .setButtonText("Balanced")
          .onClick(() => void this.applyProfile("balanced")),
      )
      .addButton((b) =>
        b
          .setButtonText("Best quality")
          .onClick(() => void this.applyProfile("best")),
      );

    new Setting(containerEl)
      .setName("Model")
      .setDesc(
        "The embedding model. Paraphrase (symmetric) models judge note-to-note similarity best: MiniLM-L12 (default) is fast, mpnet-base is stronger. jina-v5-nano is the highest quality here (it embeds whole notes plus your ideas), but it is a larger ~250MB download and non-commercial (CC-BY-NC), so it is opt-in. e5 is a retrieval model and ranks similarity poorly here. Weights download once and are cached; changing the model rebuilds the index.",
      )
      .addDropdown((d) => {
        for (const [id, label] of Object.entries(MODEL_OPTIONS)) d.addOption(id, label);
        // Allow a custom id the dropdown doesn't list.
        if (!(this.plugin.settings.modelId in MODEL_OPTIONS)) {
          d.addOption(this.plugin.settings.modelId, `${this.plugin.settings.modelId} (custom)`);
        }
        d.setValue(this.plugin.settings.modelId).onChange(async (v) => {
          // Picking a model satisfies the first-run gate. saveSettings rebuilds when the
          // model id changes; if it didn't change (a gated user re-picking the default),
          // kick off the initial build here.
          const firstChoice = !this.plugin.settings.modelChosen;
          const modelChanged = this.plugin.settings.modelId !== v;
          this.plugin.settings.modelId = v;
          this.plugin.settings.modelChosen = true;
          await this.plugin.saveSettings();
          if (firstChoice && !modelChanged) await this.plugin.rebuildIndex();
        });
      });

    new Setting(containerEl).setName("Downloaded models").setHeading();
    containerEl.createEl("div", {
      cls: "setting-item-description",
      text: "Models cached on this device. Remove one to free disk space after switching models.",
    });
    this.renderModelCache(containerEl.createDiv());

    new Setting(containerEl)
      .setName("Compute device")
      .setDesc(
        "Auto runs on the CPU (WASM) — memory-stable and recommended. WebGPU is faster " +
          "per reindex but its GPU backend accumulates memory until Obsidian crashes " +
          "(seen on large vaults), so it's an explicit opt-in only. Changing this " +
          "re-downloads the model for the new backend.",
      )
      .addDropdown((d) =>
        d
          .addOption("auto", "Auto · WASM/CPU (recommended)")
          .addOption("webgpu", "WebGPU/GPU (faster, but can spike memory)")
          .addOption("wasm", "WASM/CPU")
          .setValue(this.plugin.settings.device)
          .onChange(async (v) => {
            this.plugin.settings.device = v as DevicePref;
            await this.plugin.saveSettings();
          }),
      );

    new Setting(containerEl)
      .setName("Indexing speed")
      .setDesc(
        "How many CPU threads embed notes (WASM only). Fast uses the most cores " +
          "and is quickest; Light is single-threaded and slowest. This is a CPU/speed " +
          "knob — the engine's memory is dominated by the loaded model itself, not " +
          "the thread count (use 'Unload model when idle' below to free it). Editing " +
          "a note re-indexes just that note and stays fast at any setting. Changing " +
          "this rebuilds the index.",
      )
      .addDropdown((d) =>
        d
          .addOption("light", "Light · 1 thread (slowest)")
          .addOption("balanced", "Balanced (recommended)")
          .addOption("fast", "Fast · all cores")
          .setValue(this.plugin.settings.indexSpeed)
          .onChange(async (v) => {
            this.plugin.settings.indexSpeed = v as IndexSpeed;
            await this.plugin.saveSettings();
          }),
      );

    new Setting(containerEl)
      .setName("Unload model when idle")
      .setDesc(
        "After this long without indexing or searching, the embedding engine is " +
          "shut down to free its memory. It reloads automatically in a few seconds " +
          "the next time it is needed — no re-download. Related-notes ranking when " +
          "switching notes never needs the engine and is unaffected.",
      )
      .addDropdown((d) => {
        // 30s is the aggressive floor, not "instant": a respawn costs ~2-5s of
        // background CPU and note edits are 20s-debounced, so anything shorter
        // would tear the engine down BETWEEN consecutive edits/label batches
        // and respawn it seconds later — more churn than the RAM is worth.
        d.addOption("0", "Never")
          .addOption("0.5", "Right after use (30 seconds)")
          .addOption("1", "After 1 minute")
          .addOption("5", "After 5 minutes")
          .addOption("15", "After 15 minutes (recommended)")
          .addOption("60", "After 1 hour");
        // Honour a hand-edited data.json value the list doesn't carry (mirrors
        // the model dropdown) — otherwise the control would DISPLAY "Never"
        // while the real timeout stays active.
        const cur = String(this.plugin.settings.idleUnloadMinutes);
        if (!["0", "0.5", "1", "5", "15", "60"].includes(cur)) {
          d.addOption(cur, `After ${cur} minutes (custom)`);
        }
        d.setValue(cur).onChange(async (v) => {
          this.plugin.settings.idleUnloadMinutes = Number(v) || 0;
          await this.plugin.saveSettings();
        });
      });

    {
      const setting = new Setting(containerEl)
        .setName("Number of results")
        .setDesc("How many related notes to show in the card stack.");
      const valueEl = setting.controlEl.createSpan({
        cls: "related-notes-slider-value",
        text: String(this.plugin.settings.topK),
      });
      setting.addSlider((s) =>
        s
          .setLimits(4, 30, 1)
          .setValue(this.plugin.settings.topK)
          .onChange((v) => {
            this.plugin.settings.topK = v;
            valueEl.setText(String(v));
            this.debouncedSave();
          }),
      );
    }

    {
      const setting = new Setting(containerEl)
        .setName("Minimum similarity")
        .setDesc(
          "Hide notes below this topical-similarity score (0–1). Scores are mean-centered to remove the embedding noise floor, so unrelated notes sit near 0 and only genuinely on-topic notes score high — about 0.2 cleanly separates them. Raise it for a tighter, more focused list; lower it to show weaker matches.",
        );
      const fmt = (v: number) => v.toFixed(2);
      const valueEl = setting.controlEl.createSpan({
        cls: "related-notes-slider-value",
        text: fmt(this.plugin.settings.minSimilarity),
      });
      setting.addSlider((s) =>
        s
          .setLimits(0, 0.9, 0.05)
          .setValue(this.plugin.settings.minSimilarity)
          .onChange((v) => {
            this.plugin.settings.minSimilarity = v;
            valueEl.setText(fmt(v));
            this.debouncedSave();
          }),
      );
    }

    {
      const setting = new Setting(containerEl)
        .setName("Idea influence")
        .setDesc(
          "How much idea-level matching blends into the score. Notes are grouped into coherent ideas (~200-500 words); this weights whether two notes share a whole idea, not just one lucky passage. 0 = passage-only (the prior behavior). Live ranking knob — changing it re-ranks instantly, no re-index, so you can compare on your own notes.",
        );
      const fmt = (v: number) => v.toFixed(2);
      const valueEl = setting.controlEl.createSpan({
        cls: "related-notes-slider-value",
        text: fmt(this.plugin.settings.ideaInfluence),
      });
      setting.addSlider((s) =>
        s
          .setLimits(0, 0.6, 0.05)
          .setValue(this.plugin.settings.ideaInfluence)
          .onChange((v) => {
            this.plugin.settings.ideaInfluence = v;
            valueEl.setText(fmt(v));
            this.debouncedSave();
          }),
      );
    }

    new Setting(containerEl)
      .setName("Isolated areas (tag namespaces)")
      .setDesc(
        "Self-contained areas, one tag namespace per line. A note tagged with an activated namespace (e.g. goa, which matches goa and goa/character) only relates to, and takes tag suggestions from, other notes in that area, and never appears in any other note's cards. Notes in no activated area share one pool. Live ranking knob, no re-index.",
      )
      .addTextArea((t) =>
        t
          .setPlaceholder("goa")
          .setValue(this.plugin.settings.isolatedAreas)
          .onChange((v) => {
            this.plugin.settings.isolatedAreas = v;
            this.debouncedSave();
          }),
      );

    new Setting(containerEl)
      .setName("Excluded folders")
      .setDesc(
        "Folders left out of the index entirely — not ranked and not suggested as links. One per line (or comma-separated); matches a folder and everything beneath it.",
      )
      .addTextArea((t) =>
        t
          .setPlaceholder("Templates\nArchive/2023")
          .setValue(this.plugin.settings.excludeFolders)
          .onChange((v) => {
            this.plugin.settings.excludeFolders = v;
            this.plugin.titleIndex.markDirty();
            this.debouncedSave();
          }),
      );

    new Setting(containerEl)
      .setName("Folders excluded from link suggestions")
      .setDesc(
        "Folders whose notes stay indexed and ranked in the panel, but are never suggested as inline [[links]] (no glow). One per line (or comma-separated).",
      )
      .addTextArea((t) =>
        t
          .setPlaceholder("Daily\nAttachments/templates")
          .setValue(this.plugin.settings.excludeFoldersLinks)
          .onChange((v) => {
            this.plugin.settings.excludeFoldersLinks = v;
            this.plugin.titleIndex.markDirty();
            this.debouncedSave();
          }),
      );

    new Setting(containerEl)
      .setName("Show summary line")
      .setDesc(
        "Show a concise 3–7-word topic label on each card (extracted locally from the note's own chunks — no extra model or download). Falls back to a plain preview when off. Toggling this rebuilds the index so the labels are available, and a full rebuild takes a little longer with labels on.",
      )
      .addToggle((t) =>
        t.setValue(this.plugin.settings.showSummary).onChange(async (v) => {
          this.plugin.settings.showSummary = v;
          await this.plugin.saveSettings();
        }),
      );

    new Setting(containerEl)
      .setName("Show snippet")
      .setDesc(
        "Show a one- to two-line text preview on each card. Used as the fallback when the summary line is off.",
      )
      .addToggle((t) =>
        t.setValue(this.plugin.settings.showSnippet).onChange(async (v) => {
          this.plugin.settings.showSnippet = v;
          await this.plugin.saveSettings();
        }),
      );

    new Setting(containerEl)
      .setName("Show last-edited time")
      .setDesc("Add a muted “edited Nd ago” line to each card.")
      .addToggle((t) =>
        t.setValue(this.plugin.settings.showRecency).onChange(async (v) => {
          this.plugin.settings.showRecency = v;
          await this.plugin.saveSettings();
        }),
      );

    {
      const setting = new Setting(containerEl)
        .setName("Structure influence")
        .setDesc(
          "How much shared tags, links, co-citations, and frontmatter nudge the ranking. Bounded so it only re-orders near-ties and never promotes an unrelated note. 0 disables it.",
        );
      const fmt = (v: number) => v.toFixed(2);
      const valueEl = setting.controlEl.createSpan({
        cls: "related-notes-slider-value",
        text: fmt(this.plugin.settings.structureInfluence),
      });
      setting.addSlider((s) =>
        s
          .setLimits(0, 0.3, 0.01)
          .setValue(this.plugin.settings.structureInfluence)
          .onChange((v) => {
            this.plugin.settings.structureInfluence = v;
            valueEl.setText(fmt(v));
            this.debouncedSave();
          }),
      );
    }

    new Setting(containerEl)
      .setName("Chunk-level matching")
      .setDesc(
        "Embed each note as several sentence-level chunks instead of one whole-note vector — far more accurate for long notes. Turning this off reverts to single-vector matching. Changing it rebuilds the index.",
      )
      .addToggle((t) =>
        t.setValue(this.plugin.settings.chunking).onChange(async (v) => {
          this.plugin.settings.chunking = v;
          await this.plugin.saveSettings();
        }),
      );

    {
      const setting = new Setting(containerEl)
        .setName("Max chunks per note")
        .setDesc(
          "Advanced. Ceiling on idea-chunks embedded per note (the title is extra). The whole note is covered up to this cap; only very long notes approach it. Higher captures more breadth but grows the index and slows ranking. Changing it rebuilds the index.",
        );
      const valueEl = setting.controlEl.createSpan({
        cls: "related-notes-slider-value",
        text: String(this.plugin.settings.maxChunks),
      });
      setting.addSlider((s) =>
        s
          .setLimits(8, 64, 1)
          .setValue(this.plugin.settings.maxChunks)
          .onChange((v) => {
            this.plugin.settings.maxChunks = v;
            valueEl.setText(String(v));
            this.debouncedSave();
          }),
      );
    }

    new Setting(containerEl)
      .setName("Heading context")
      .setDesc(
        "Prefix each section's first chunk with its note + heading breadcrumb when embedding, so a section embeds with the context of where it sits. Improves matching; turn off to compare. Changing it rebuilds the index.",
      )
      .addToggle((t) =>
        t.setValue(this.plugin.settings.headingContext).onChange(async (v) => {
          this.plugin.settings.headingContext = v;
          await this.plugin.saveSettings();
        }),
      );

    {
      const setting = new Setting(containerEl)
        .setName("Shortlist size")
        .setDesc(
          "Advanced. How many coarse candidates are re-ranked with the precise chunk comparison on each note switch. Higher is slightly more thorough but slower (kept at least 4× the result count).",
        );
      const valueEl = setting.controlEl.createSpan({
        cls: "related-notes-slider-value",
        text: String(this.plugin.settings.shortlistSize),
      });
      setting.addSlider((s) =>
        s
          .setLimits(20, 150, 10)
          .setValue(this.plugin.settings.shortlistSize)
          .onChange((v) => {
            this.plugin.settings.shortlistSize = v;
            valueEl.setText(String(v));
            this.debouncedSave();
          }),
      );
    }

    // --- Linking (Features A + B) --------------------------------------------
    new Setting(containerEl).setName("Linking").setHeading();

    new Setting(containerEl)
      .setName("Highlight linkable mentions")
      .setDesc(
        "Glow the first time a concept that already has a note is named in the current note. Click the glow to turn it into a [[wikilink]]. Matches exact titles and aliases only (precise — it never glows a phrase with no matching note).",
      )
      .addToggle((t) =>
        t.setValue(this.plugin.settings.glowEnabled).onChange(async (v) => {
          this.plugin.settings.glowEnabled = v;
          await this.plugin.saveSettings();
        }),
      );

    new Setting(containerEl)
      .setName("Highlight in live preview only")
      .setDesc(
        "Only show the glow in live preview (the normal reading/editing view), not in raw source mode.",
      )
      .addToggle((t) =>
        t
          .setValue(this.plugin.settings.glowRestrictToLivePreview)
          .onChange(async (v) => {
            this.plugin.settings.glowRestrictToLivePreview = v;
            await this.plugin.saveSettings();
          }),
      );

    new Setting(containerEl)
      .setName("Highlight ambiguous mentions")
      .setDesc(
        "Also glow a phrase owned by two or more notes. Off by default for precision (an ambiguous mention can't be attributed to one note).",
      )
      .addToggle((t) =>
        t.setValue(this.plugin.settings.glowAmbiguous).onChange(async (v) => {
          this.plugin.settings.glowAmbiguous = v;
          await this.plugin.saveSettings();
        }),
      );

    new Setting(containerEl)
      .setName("Auto-link later mentions")
      .setDesc(
        "After a note is linked once (by click or the command), automatically link its remaining mentions in this note while you're idle. Opt-in: cursor-aware and re-validating, so it never clobbers what you're typing.",
      )
      .addToggle((t) =>
        t
          .setValue(this.plugin.settings.autoLinkSubsequent)
          .onChange(async (v) => {
            this.plugin.settings.autoLinkSubsequent = v;
            await this.plugin.saveSettings();
          }),
      );

    new Setting(containerEl)
      .setName("Smart [[ suggestions")
      .setDesc(
        "When you type [[, rank existing notes by semantic relevance to what you're writing (reusing the index), not just by name.",
      )
      .addToggle((t) =>
        t.setValue(this.plugin.settings.suggesterEnabled).onChange(async (v) => {
          this.plugin.settings.suggesterEnabled = v;
          await this.plugin.saveSettings();
        }),
      );

    new Setting(containerEl)
      .setName("Take over the [[ popup")
      .setDesc(
        "Put these suggestions at the top of the [[ popup. Off by default when the Easy Links smart suggester is active, so the two don't fight; turn on to prefer these.",
      )
      .addToggle((t) =>
        t.setValue(this.plugin.settings.suggesterTakeOver).onChange(async (v) => {
          this.plugin.settings.suggesterTakeOver = v;
          // Mark the value as an explicit user choice so the easy-links-aware
          // auto-default never re-derives (and silently overrides) it again.
          this.plugin.settings.suggesterTakeOverUserSet = true;
          await this.plugin.saveSettings();
        }),
      );

    new Setting(containerEl)
      .setName("Suggest new notes")
      .setDesc(
        "Offer to create a brand-new note for a strongly-relevant concept that doesn't have one yet, labelled clearly as new.",
      )
      .addToggle((t) =>
        t.setValue(this.plugin.settings.suggestNewNotes).onChange(async (v) => {
          this.plugin.settings.suggestNewNotes = v;
          await this.plugin.saveSettings();
        }),
      );

    {
      const setting = new Setting(containerEl)
        .setName("New-note confidence")
        .setDesc(
          "How relevant a concept must be before a “create new note” row is offered (0–1). Higher proposes fewer, more confident new notes.",
        );
      const fmt = (v: number) => v.toFixed(2);
      const valueEl = setting.controlEl.createSpan({
        cls: "related-notes-slider-value",
        text: fmt(this.plugin.settings.newNoteMinSimilarity),
      });
      setting.addSlider((s) =>
        s
          .setLimits(0, 0.9, 0.05)
          .setValue(this.plugin.settings.newNoteMinSimilarity)
          .onChange((v) => {
            this.plugin.settings.newNoteMinSimilarity = v;
            valueEl.setText(fmt(v));
            this.debouncedSave();
          }),
      );
    }

    // --- index status + manual rebuild ---------------------------------------
    const progress = this.plugin.store.getProgress();
    new Setting(containerEl)
      .setName("Index")
      .setDesc(
        progress.status === "building"
          ? `Indexing… ${progress.done}/${progress.total}`
          : `${this.plugin.store.count} notes embedded.`,
      )
      .addButton((b) =>
        b
          .setButtonText("Rebuild index")
          .setCta()
          .onClick(() => {
            void this.plugin.rebuildIndex();
          }),
      );
  }

  hide(): void {
    // Persist any value typed/dragged right before the pane closed.
    this.debouncedSave.run();
  }
}
