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
  Modal,
  Notice,
  normalizePath,
  requestUrl,
  debounce,
  getAllTags,
  type Editor,
  type Debouncer,
  type SettingDefinition,
  type SettingDefinitionItem,
} from "obsidian";
import { ReaderService, type ReaderHost, type ReaderPace } from "./reader/reader-service";
import { WhatsNewModal, WHATS_NEW_ID } from "./whats-new";
import readerEngineSource from "virtual:reader-engine";
import pluginStyles from "virtual:plugin-styles";
import {
  RUNGS,
  removeAssets,
  assetSizesMb,
  offlineItems,
  importAssetFiles,
  importFromFolder,
  downloadsDir,
  type RungSpec,
} from "./reader/reader-assets";
import {
  EmbeddingEngine,
  setOrtAssetLoader,
  setEmbedThreads,
  type OrtAssets,
  type OrtRuntimeBuild,
  type DevicePref,
} from "./embeddings";
import { ORT_WEB_CDN, ORT_WEB_VERSION, ORT_RUNTIMES, type OrtRuntimeFiles } from "./ort-version";

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
import { VaultMapView, VIEW_TYPE_MAP } from "./map-view";
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
  showFolder: boolean; // folder path on cards
  showWhyPills: boolean; // why-related reason pill (Co-cited, Similar text, via ...)
  showTagPills: boolean; // the shared-tag reason pill (#tag) specifically
  linkedFirst: boolean; // sort directly linked notes above the rest
  openInNewTab: boolean; // plain card click opens a new tab (mod-click then opens here)
  pinPersist: boolean; // keep the panel's pinned note across restarts
  pinnedPath: string; // internal: last pinned note path (only meaningful with pinPersist)
  showWhatsNew: boolean; // one-time highlights popup after a feature update
  whatsNewShownId: string; // internal: WHATS_NEW_ID the user has already seen
  maxChunks: number; // body-chunk cap (advanced)
  shortlistSize: number; // Stage-1 -> Stage-2 funnel width (advanced)
  headingContext: boolean; // prefix each section's first chunk with a heading breadcrumb
  ideaInfluence: number; // 0..0.6 rank-time blend of idea-level similarity (0 = off; live, no re-embed)
  isolatedAreas: string; // activated tag namespaces (comma/newline) that form self-contained areas
  graphInfluence: number; // 0..1 weight of the link-graph channel in the fused rank (0 = content only)
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
  // --- reader (4.0, beta) ---
  readerEnabled: boolean; // master toggle; enabling starts the asset download
  readerRung: "auto" | RungSpec["id"]; // model size; auto = RAM probe
  readerPace: ReaderPace; // idle reading cadence
  readerOneLiners: boolean; // reader one-liner replaces the keyphrase label line
  readerSuggestTags: boolean; // ghost tag chips on the active note
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
  showFolder: true,
  showWhyPills: true,
  showTagPills: true,
  linkedFirst: false,
  openInNewTab: false,
  pinPersist: false,
  pinnedPath: "",
  showWhatsNew: true,
  whatsNewShownId: "",
  maxChunks: 48,
  shortlistSize: 60,
  headingContext: true,
  ideaInfluence: 0.3,
  isolatedAreas: "",
  graphInfluence: 1,
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
  readerEnabled: false,
  readerRung: "auto",
  readerPace: "balanced",
  readerOneLiners: true,
  readerSuggestTags: true,
};

// A few vetted model ids surfaced as a dropdown so users don't have to memorise
// HF repo paths. Any other id can still be typed in the text field below.
// Paraphrase (symmetric) models first — they judge note-to-note similarity far
// better than retrieval models for this use case.
const MODEL_OPTIONS: Record<string, string> = {
  "Xenova/paraphrase-multilingual-MiniLM-L12-v2":
    "MiniLM (default, fast)",
  "Xenova/paraphrase-multilingual-mpnet-base-v2":
    "mpnet (stronger, slower)",
  "Xenova/multilingual-e5-small":
    "e5-small (search-tuned)",
  "jinaai/jina-embeddings-v5-text-nano-text-matching":
    "jina-v5-nano (best, ~250 MB)",
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

// The ORT runtime pair every worker spawn needs. WHICH pair depends on the
// device: the engine asks for "wasm" (CPU-only build) or "webgpu" (asyncify
// build, the only one with the WebGPU EP) — see embeddings.ts runtimeBuildFor.
// They are not interchangeable, so each is named, sized and cached separately.
// Present locally in dev builds (gen-ort.mjs copies BOTH next to main.js) and in
// legacy full-zip installs; community-directory installs get only
// main.js/manifest/styles, so the plugin downloads the pair it needs ONCE from a
// pinned CDN into its own ort/ folder and serves it locally from then on
// (offline after first run) — only the pair actually used is ever fetched. Any
// pair — local or just downloaded — is validated against the EXACT byte sizes and
// SHA-256 digests gen-ort.mjs pinned (ORT_RUNTIMES[build]) before use: the pair
// must be the same build the bundled transformers glue expects, or ORT init fails
// with cryptic mixed-build errors, and it is executable code, so it must be the
// exact file npm installed at build time whatever CDN or proxy served it. A
// mismatch (torn cache write, files from an install carrying a different
// onnxruntime-web, CDN/proxy garbage, tampering) → re-download.
// CDN sources tried in order; both serve the pinned npm package.
const ORT_CDNS = [
  ORT_WEB_CDN,
  `https://unpkg.com/onnxruntime-web@${ORT_WEB_VERSION}/dist/`,
];

// Hex SHA-256 of a buffer, with the Web Crypto every Obsidian window has.
async function sha256Hex(buf: ArrayBuffer): Promise<string> {
  const digest = await window.crypto.subtle.digest("SHA-256", buf);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

// Why a glue+wasm pair is unusable for this build, or null when it matches the
// pinned sizes and digests. Sizes first: they are free and catch most damage.
async function ortPairProblem(
  glue: ArrayBuffer,
  wasm: ArrayBuffer,
  files: OrtRuntimeFiles,
): Promise<string | null> {
  if (glue.byteLength !== files.glueBytes || wasm.byteLength !== files.wasmBytes) {
    return `unexpected sizes (glue ${glue.byteLength}/${files.glueBytes} B, wasm ${wasm.byteLength}/${files.wasmBytes} B)`;
  }
  const [glueSha, wasmSha] = await Promise.all([sha256Hex(glue), sha256Hex(wasm)]);
  if (glueSha !== files.glueSha256 || wasmSha !== files.wasmSha256) {
    return "content does not match the pinned SHA-256 digests";
  }
  return null;
}

// ---------------------------------------------------------------------------

export default class RelatedNotesPlugin extends Plugin {
  declare settings: RelatedNotesSettings;
  store!: IndexStore;
  reader: ReaderService | null = null;
  private settingsTab: RelatedNotesSettingTab | null = null;
  private lastActivityAt = Date.now();

  requestPanelRender(): void {
    this.getView()?.requestRender();
  }
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
    // Self-heal styles.css: a manual update that replaces only main.js leaves
    // the stylesheet stale, silently losing every CSS fix. Obsidian loads the
    // file itself (injecting CSS is against plugin guidelines), so when the
    // on-disk copy differs from the one this main.js was built with, rewrite
    // it and ask for one reload.
    void (async () => {
      try {
        const cssPath = normalizePath(`${this.pluginDir()}/styles.css`);
        const onDisk = (await this.app.vault.adapter.exists(cssPath))
          ? await this.app.vault.adapter.read(cssPath)
          : "";
        if (onDisk !== pluginStyles) {
          await this.app.vault.adapter.write(cssPath, pluginStyles);
          new Notice(
            "Smart Related Notes: styles.css was from another version and has been refreshed. Reload Obsidian once to apply.",
            10000,
          );
        }
      } catch (e) {
        console.warn("[related-notes] styles self-heal:", e);
      }
    })();

    const saved = (await this.loadData()) as Partial<RelatedNotesSettings> | null;
    this.settings = Object.assign({}, DEFAULT_SETTINGS, saved);

    // First-run gate migration: an install with settings already saved is treated as
    // having chosen a model, so only a truly fresh install (no data.json) starts gated.
    if (saved && saved.modelChosen === undefined) this.settings.modelChosen = true;

    // The what's-new popup is for UPDATES. A fresh install has nothing "new",
    // so it is marked as seen up front and never shown the current highlights.
    if (!saved) this.settings.whatsNewShownId = WHATS_NEW_ID;

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
    setOrtAssetLoader((build) => this.loadOrtAssets(build));

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

    // Reader (4.0): a local reader model working only in idle gaps. Activity
    // timestamps gate its scheduler; everything else lives in ReaderService.
    this.registerDomEvent(activeDocument, "keydown", () => (this.lastActivityAt = Date.now()));
    this.registerDomEvent(activeDocument, "pointerdown", () => (this.lastActivityAt = Date.now()));
    this.reader = new ReaderService(this.readerHost(), {
      rung: this.settings.readerRung,
      pace: this.settings.readerPace,
      idleUnloadMinutes: this.settings.idleUnloadMinutes,
    });
    this.reader.onStatus = () => {
      this.getView()?.requestRender();
      this.settingsTab?.updateReaderStatus();
    };
    if (this.settings.readerEnabled) void this.reader.enable();

    this.registerView(VIEW_TYPE_RELATED, (leaf) => new RelatedNotesView(leaf, this));
    this.registerView(VIEW_TYPE_MAP, (leaf) => new VaultMapView(leaf, this));

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
      id: "reader-offline-setup",
      name: "Reader: offline setup (restricted networks)",
      callback: () => {
        new ReaderOfflineModal(this).open();
      },
    });

    this.addCommand({
      id: "open-map",
      name: "Open the vault map",
      callback: () => {
        void this.activateMap();
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

    this.settingsTab = new RelatedNotesSettingTab(this.app, this);
    this.addSettingTab(this.settingsTab);

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
        // Editing a note can add or remove wikilinks, which changes the structural
        // channel's adjacency. Rebuilding it is a single pass over resolvedLinks
        // with no model work, so invalidating eagerly is cheaper than tracking
        // which links actually moved.
        this.store.invalidateGraph();
      }),
    );
    // resolvedLinks is only authoritative once Obsidian has resolved the whole
    // vault; before that the structural channel would see a partial graph.
    this.registerEvent(
      this.app.metadataCache.on("resolved", () => {
        this.store.invalidateGraph();
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
      this.maybeShowWhatsNew();
    });
  }

  // After a feature update, show the bundled highlights once. Slightly delayed
  // past layout-ready so it never competes with the workspace restoring.
  private maybeShowWhatsNew(): void {
    if (!this.settings.showWhatsNew) return;
    if (this.settings.whatsNewShownId === WHATS_NEW_ID) return;
    this.settings.whatsNewShownId = WHATS_NEW_ID;
    void this.saveData(this.settings);
    window.setTimeout(() => {
      new WhatsNewModal(this.app, () => this.openReaderSettingsTab()).open();
    }, 900);
  }

  private openReaderSettingsTab(): void {
    const setting = (
      this.app as App & { setting?: { open(): void; openTabById(id: string): void } }
    ).setting;
    setting?.open();
    setting?.openTabById(this.manifest.id);
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
    void this.reader?.disable();
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
  private async loadOrtAssets(build: OrtRuntimeBuild): Promise<OrtAssets> {
    const files = ORT_RUNTIMES[build];
    const adapter = this.app.vault.adapter;
    const dir = normalizePath(`${this.pluginDir()}/ort`);
    const gluePath = normalizePath(`${dir}/${files.glue}`);
    const wasmPath = normalizePath(`${dir}/${files.wasm}`);

    // 1) Local pair, validated against the pinned build.
    try {
      if ((await adapter.exists(gluePath)) && (await adapter.exists(wasmPath))) {
        const [glueBuf, wasmBinary] = await Promise.all([
          adapter.readBinary(gluePath),
          adapter.readBinary(wasmPath),
        ]);
        const problem = await ortPairProblem(glueBuf, wasmBinary, files);
        if (problem === null) {
          return { glueText: new TextDecoder().decode(glueBuf), wasmBinary };
        }
        // A torn/interrupted cache write, files from an install carrying a
        // different onnxruntime-web build, or altered files: unusable with the
        // glue bundled into THIS version. Fall through and replace them.
        console.warn(
          `[related-notes] local ORT ${build} runtime does not match the bundled build (${problem}); re-downloading`,
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
          requestUrl({ url: base + files.glue }),
          requestUrl({ url: base + files.wasm }),
        ]);
        const glueBinary = glue.arrayBuffer;
        const wasmBinary = wasm.arrayBuffer;
        // Validate BEFORE caching or serving: a proxy/captive-portal can return
        // 200 with garbage, and nothing but the pinned build may reach the worker
        // or the cache.
        const problem = await ortPairProblem(glueBinary, wasmBinary, files);
        if (problem !== null) {
          failures.push(`${base}: ${problem}`);
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
      `could not obtain the ONNX runtime (${files.wasm}) — ${failures.join("; ")}. ` +
        "Offline, or a firewall/proxy/antivirus blocks these URLs.",
    );
  }

  // Everything the reader service may touch, in one narrow surface. It gets no
  // direct store or vault access beyond this.
  private readerHost(): ReaderHost {
    return {
      vaultRead: (file) => this.app.vault.cachedRead(file),
      markdownFiles: () =>
        this.app.vault.getMarkdownFiles().filter((f) => !this.store.isExcluded(f.path)),
      fileByPath: (path) => {
        const f = this.app.vault.getAbstractFileByPath(path);
        return f instanceof TFile ? f : null;
      },
      fileTags: (path) => {
        const f = this.app.vault.getAbstractFileByPath(path);
        if (!(f instanceof TFile)) return [];
        const cache = this.app.metadataCache.getFileCache(f);
        return (cache ? getAllTags(cache) ?? [] : []).map((t) => t.replace(/^#/, "").toLowerCase());
      },
      neighborPaths: (path, k) => {
        const f = this.app.vault.getAbstractFileByPath(path);
        if (!(f instanceof TFile)) return Promise.resolve([]);
        return Promise.resolve(this.store.rank(f).slice(0, k).map((r) => r.file.path));
      },
      recentPaths: () => this.app.workspace.getLastOpenFiles(),
      indexBusy: () => {
        const s = this.store.getProgress().status;
        return s === "building" || s === "loading";
      },
      lastUserActivity: () => this.lastActivityAt,
      requestRender: () => this.getView()?.requestRender(),
      loadArtifacts: async () => {
        const p = normalizePath(`${this.pluginDir()}/reader-artifacts.json`);
        return (await this.app.vault.adapter.exists(p)) ? this.app.vault.adapter.read(p) : null;
      },
      saveArtifacts: (json) =>
        this.app.vault.adapter.write(normalizePath(`${this.pluginDir()}/reader-artifacts.json`), json),
      // The engine bundle ships inlined in main.js (see esbuild.config.mjs),
      // so a manual or store install can never be missing it.
      readEngineBundle: () => Promise.resolve(new TextEncoder().encode(readerEngineSource)),
    };
  }

  // Settings-tab hook: push the current reader settings into the service and
  // start/stop it on the master toggle.
  applyReaderSettings(): void {
    if (!this.reader) return;
    this.reader.configure({
      rung: this.settings.readerRung,
      pace: this.settings.readerPace,
      idleUnloadMinutes: this.settings.idleUnloadMinutes,
    });
    if (this.settings.readerEnabled && this.reader.status.state === "off") void this.reader.enable();
    if (!this.settings.readerEnabled && this.reader.status.state !== "off") void this.reader.disable();
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
    // Asking to build IS choosing. The dropdown already shows a model (the default
    // is preselected), so a user who opens settings and presses Rebuild has made a
    // deliberate choice — but re-picking the value already displayed fires no
    // change event, so the gate would never lift and the button would do nothing
    // at all. Treat the press as the choice and get on with it.
    if (!this.settings.modelChosen) {
      this.settings.modelChosen = true;
      await this.saveSettings();
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
    // Written as Obsidian callouts with tables inside, not flat bullet lists. Every
    // section is foldable, so a report with six sections and two hundred rows opens
    // as something you can scan rather than something you scroll past. Callout rows
    // need a "> " prefix on every line, including table rows and blanks.
    const cal = (kind: string, title: string, lines: string[], open = true): void => {
      L.push(`> [!${kind}]${open ? "+" : "-"} ${title}`);
      for (const line of lines) L.push(`> ${line}`);
      L.push("");
    };
    const table = (head: string[], rows: string[][]): string[] => {
      if (rows.length === 0) return [];
      return [
        `| ${head.join(" | ")} |`,
        `|${head.map(() => "---").join("|")}|`,
        ...rows.map((r) => `| ${r.join(" | ")} |`),
      ];
    };

    L.push("# Vault Insights", "");
    L.push(
      `> [!abstract] ${ins.total} notes indexed`,
      "> " + [
        `**${ins.suggestedLinks.length}** links to make`,
        `**${ins.orphans.length}** orphans`,
        `**${ins.nearDuplicates.length}** possible duplicates`,
      ].join(" · "),
      ">",
      `> <sub>Re-run the **Vault insights** command to refresh. Nothing in your notes is changed.</sub>`,
      "",
    );

    cal("tip", `Links to make (${ins.suggestedLinks.length})`,
      ins.suggestedLinks.length === 0
        ? ["Nothing above the threshold. Either the vault is well connected, or the similarity floor is set high."]
        : ["Strongly related, not linked yet.", "",
           ...table(["Note", "Relates to", "Match"],
             ins.suggestedLinks.map((x) => [wl(x.from), wl(x.to), pct(x.score)]))]);

    cal("hint", `Suggested tags (${ins.suggestedTags.length})`,
      ins.suggestedTags.length === 0
        ? ["None. No note is missing a tag that its neighbours agree on."]
        : ["Notes whose neighbours share a tag they are missing. Inferred from similarity, not from rules.", "",
           ...table(["Note", "Tag", "Neighbours agreeing"],
             ins.suggestedTags.map((x) => [wl(x.path), `#${x.tag}`, `${x.support} of ${x.neighbors}`]))], false);

    cal("warning", `Orphans (${ins.orphans.length})`,
      ins.orphans.length === 0
        ? ["None. Every note is connected to something."]
        : ["No links in or out. The closest relative is where to start.", "",
           ...table(["Orphan", "Closest relative", "Match"],
             ins.orphans.map((x) => [wl(x.path), x.closest ? wl(x.closest) : "-", x.closest ? pct(x.closestScore) : "-"]))], false);

    cal("danger", `Possible duplicates (${ins.nearDuplicates.length})`,
      ins.nearDuplicates.length === 0
        ? ["None above the threshold."]
        : ["Near-identical. Worth merging, or cross-linking if the overlap is deliberate.", "",
           ...table(["Note", "Note", "Match"],
             ins.nearDuplicates.map((x) => [wl(x.a), wl(x.b), pct(x.score)]))], false);

    cal("calendar", `Not touched in a while (${ins.stale.length})`,
      ins.stale.length === 0
        ? ["Nothing stale."]
        : ["The oldest edits in the vault. Not a problem, just the part you have stopped looking at.", "",
           ...table(["Note", "Last edited"], ins.stale.map((x) => [wl(x.path), ymd(x.mtime)]))], false);

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

  // The map wants room, so it opens as a main-area tab rather than in the
  // sidebar. Reopening focuses the existing tab and redraws it, since the index
  // may have moved on since it was last built.
  async activateMap(): Promise<void> {
    const { workspace } = this.app;
    let leaf: WorkspaceLeaf | null = workspace.getLeavesOfType(VIEW_TYPE_MAP)[0] ?? null;
    if (!leaf) {
      leaf = workspace.getLeaf(true);
      await leaf.setViewState({ type: VIEW_TYPE_MAP, active: true });
    } else if (leaf.view instanceof VaultMapView) {
      leaf.view.rebuild();
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
      graphInfluence: this.settings.graphInfluence,
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

// Settings keys a control can bind to. The definitions are typed on these, so a
// renamed setting breaks the build instead of rendering an empty control.
type SettingKey = Extract<keyof RelatedNotesSettings, string>;

// Continuous controls (sliders, text areas) save after a short pause instead of
// on every step: several of them rebuild the index when they change.
const SAVE_LATER_KEYS = new Set<string>([
  "topK",
  "minSimilarity",
  "newNoteMinSimilarity",
  "ideaInfluence",
  "graphInfluence",
  "structureInfluence",
  "maxChunks",
  "shortlistSize",
  "excludeFolders",
  "excludeFoldersLinks",
  "isolatedAreas",
]);
// Applied to the running reader at once, on every change.
const READER_KEYS = new Set<string>([
  "readerEnabled",
  "readerRung",
  "readerPace",
  "readerOneLiners",
  "readerSuggestTags",
]);
// Change what a card shows: the panel re-renders at once.
const PANEL_KEYS = new Set<string>([
  "showFolder",
  "showTagPills",
  "showWhyPills",
  "linkedFirst",
  "readerOneLiners",
  "readerSuggestTags",
]);
// Folder lists that change which mentions may become links.
const TITLE_INDEX_KEYS = new Set<string>(["excludeFolders", "excludeFoldersLinks"]);
// Shown in a sub-page's summary line, which only update() refreshes. The text
// areas feeding the Scope summary refresh when the pane closes instead, so
// typing never loses focus to a rebuild.
const SUMMARY_KEYS = new Set<string>(["readerEnabled", "readerRung", "glowEnabled", "device"]);

// 30 s is the aggressive floor, not "instant": a respawn costs ~2-5 s of
// background CPU and note edits are 20 s-debounced, so anything shorter would
// tear the engine down BETWEEN consecutive edits and label batches and respawn
// it seconds later, which is more churn than the memory is worth.
const IDLE_UNLOAD_OPTIONS: Record<string, string> = {
  "0": "Never",
  "0.5": "Right after use (30 seconds)",
  "1": "After 1 minute",
  "5": "After 5 minutes",
  "15": "After 15 minutes (recommended)",
  "60": "After 1 hour",
};

const DEVICE_OPTIONS: Record<DevicePref, string> = {
  auto: "Auto · WASM/CPU (recommended)",
  webgpu: "WebGPU/GPU (faster, but can spike memory)",
  wasm: "WASM/CPU",
};

// Entries of a comma- or newline-separated setting, for the page summaries.
function listCount(value: string): number {
  return value.split(/[\n,]/).filter((v) => v.trim().length > 0).length;
}

export class RelatedNotesSettingTab extends PluginSettingTab {
  plugin: RelatedNotesPlugin;
  private readonly debouncedSave: Debouncer<[], void>;
  // The reader's status row while it is on screen, and whether the engine
  // error was present at the last status update (a change rebuilds the tab,
  // so the error row and the page's warning mark follow it).
  private readerStatusEl: HTMLElement | null = null;
  private readerErrorShown = false;

  constructor(app: App, plugin: RelatedNotesPlugin) {
    super(app, plugin);
    this.plugin = plugin;
    this.debouncedSave = debounce(() => void this.plugin.saveSettings(), 500, false);
  }

  // Obsidian renders the tab, its sub-pages and the settings search from this
  // list. Nothing here reads files or the network: rows that need to do that
  // work in their render callback.
  getSettingDefinitions(): SettingDefinitionItem<SettingKey>[] {
    return [
      {
        name: "Pick a model to start",
        desc: "Indexing is paused until you choose a model under Setup.",
        visible: () => !this.plugin.settings.modelChosen,
        searchable: false,
      },
      { type: "group", heading: "Setup", items: this.setupItems() },
      { type: "group", heading: "Results panel", items: this.resultsItems() },
      {
        type: "group",
        heading: "More",
        items: [
          {
            type: "page",
            name: "Reader",
            desc: "A small local model reads your notes while Obsidian is idle, so the panel can label and judge connections. Notes never leave this machine.",
            displayValue: () => this.readerSummary(),
            status: () => (this.plugin.reader?.engineError() ? "warning" : null),
            items: this.readerItems(),
          },
          {
            type: "page",
            name: "Linking",
            desc: "Glowing mentions you can turn into links, and smarter [[ suggestions.",
            displayValue: () => (this.plugin.settings.glowEnabled ? "Highlights on" : "Highlights off"),
            items: this.linkingItems(),
          },
          {
            type: "page",
            name: "Scope",
            desc: "Folders the plugin leaves out, and tag areas it keeps apart.",
            displayValue: () => this.scopeSummary(),
            items: this.scopeItems(),
          },
          {
            type: "page",
            name: "Engine",
            desc: "Where the embedding model runs, and the models stored on this device.",
            displayValue: () => DEVICE_OPTIONS[this.plugin.settings.device] ?? this.plugin.settings.device,
            items: this.engineItems(),
          },
          {
            type: "page",
            name: "Advanced matching",
            desc: "Ranking settings most vaults never need.",
            items: this.advancedItems(),
          },
          {
            name: "Show what's new after updates",
            desc: "A one-time popup with the highlights of a feature release. Never shown twice, never for small fixes.",
            control: { type: "toggle", key: "showWhatsNew" },
          },
        ],
      },
    ];
  }

  getControlValue(key: string): unknown {
    const value = (this.plugin.settings as unknown as Record<string, unknown>)[key];
    // The idle-unload dropdown works with strings; the setting is in minutes.
    return key === "idleUnloadMinutes" ? String(value) : value;
  }

  // Every control change ends here. saveSettings() applies the change (and
  // rebuilds the index when the model or the embedding shape changed), so it is
  // started, not awaited: a rebuild can take minutes and the control must not
  // wait for it. Sliders and text areas save after a pause.
  setControlValue(key: string, value: unknown): Promise<void> {
    if (key === "modelId") {
      this.chooseModel(String(value));
      return Promise.resolve();
    }
    const settings = this.plugin.settings as unknown as Record<string, unknown>;
    settings[key] = key === "idleUnloadMinutes" ? Number(value) || 0 : value;
    // An explicit choice: the easy-links-aware default never overrides it again.
    if (key === "suggesterTakeOver") this.plugin.settings.suggesterTakeOverUserSet = true;
    if (key === "pinPersist" && value === false) this.plugin.settings.pinnedPath = "";
    if (TITLE_INDEX_KEYS.has(key)) this.plugin.titleIndex.markDirty();

    if (SAVE_LATER_KEYS.has(key) || READER_KEYS.has(key)) this.debouncedSave();
    else void this.plugin.saveSettings();

    if (READER_KEYS.has(key)) this.plugin.applyReaderSettings();
    if (PANEL_KEYS.has(key)) this.plugin.requestPanelRender();
    // With persistence on, the next panel render stores a pin that is set now.
    if (key === "pinPersist") this.plugin.getView()?.requestRender();

    if (SUMMARY_KEYS.has(key)) this.update();
    else this.refreshDomState();
    return Promise.resolve();
  }

  // Picking a model satisfies the first-run gate. saveSettings() rebuilds when
  // the model changes; when it did not (a gated user picking the default), the
  // first build starts here.
  private chooseModel(id: string): void {
    const firstChoice = !this.plugin.settings.modelChosen;
    const modelChanged = this.plugin.settings.modelId !== id;
    this.plugin.settings.modelId = id;
    this.plugin.settings.modelChosen = true;
    if (firstChoice) this.update();
    void this.plugin.saveSettings().then(() => {
      if (firstChoice && !modelChanged) return this.plugin.rebuildIndex();
    });
  }

  // Apply a one-click preset, persist it (the index rebuilds if the model
  // changed), and show the new values.
  private applyProfile(name: ProfileName): void {
    Object.assign(this.plugin.settings, PROFILES[name]);
    void this.plugin.saveSettings();
    this.update();
    new Notice(
      `Related notes: applied the ${name === "best" ? "Best quality" : "Balanced"} profile.`,
    );
  }

  // The model list, plus a custom id the list does not carry.
  private modelOptions(): Record<string, string> {
    const options = { ...MODEL_OPTIONS };
    const id = this.plugin.settings.modelId;
    if (!(id in options)) options[id] = `${id} (custom)`;
    return options;
  }

  private indexDescription(): string {
    const progress = this.plugin.store.getProgress();
    if (progress.status === "building") return `Indexing… ${progress.done}/${progress.total}`;
    const count = this.plugin.store.count;
    return progress.skipped
      ? `${count} notes embedded, ${progress.skipped} skipped. The developer console names each skipped note.`
      : `${count} notes embedded.`;
  }

  private readerSummary(): string {
    if (!this.plugin.settings.readerEnabled) return "Off";
    const rung = this.plugin.settings.readerRung;
    return rung === "auto" ? "On" : `On · ${RUNGS.find((r) => r.id === rung)?.label ?? rung}`;
  }

  private scopeSummary(): string {
    const folders = listCount(this.plugin.settings.excludeFolders);
    const areas = listCount(this.plugin.settings.isolatedAreas);
    const parts = [
      folders > 0 ? `${folders} folder${folders === 1 ? "" : "s"} left out` : "",
      areas > 0 ? `${areas} isolated area${areas === 1 ? "" : "s"}` : "",
    ].filter(Boolean);
    return parts.length > 0 ? parts.join(" · ") : "Whole vault";
  }

  // --- Setup -----------------------------------------------------------------
  private setupItems(): SettingDefinition<SettingKey>[] {
    return [
      {
        name: "Model",
        desc: "MiniLM is the fast default. jina-v5-nano is the strongest here (a ~250 MB, non-commercial download). Changing this re-embeds the vault.",
        aliases: ["embedding model", "MiniLM", "mpnet", "e5", "jina"],
        control: { type: "dropdown", key: "modelId", options: this.modelOptions() },
      },
      {
        name: "Index",
        desc: this.indexDescription(),
        aliases: ["rebuild", "reindex", "embedded notes"],
        render: (setting) => {
          setting.setDesc(this.indexDescription());
          setting.addButton((b) =>
            b
              .setButtonText("Rebuild index")
              .setCta()
              .onClick(() => {
                void this.plugin.rebuildIndex();
              }),
          );
        },
      },
      {
        name: "Performance profile",
        desc: "Balanced is fast and light. Best quality is slower and stronger.",
        aliases: ["preset", "quality"],
        render: (setting) => {
          setting
            .addButton((b) => b.setButtonText("Balanced").onClick(() => this.applyProfile("balanced")))
            .addButton((b) => b.setButtonText("Best quality").onClick(() => this.applyProfile("best")));
        },
      },
    ];
  }

  // --- Results panel: what the card stack shows ------------------------------
  private resultsItems(): SettingDefinition<SettingKey>[] {
    return [
      {
        name: "Number of results",
        desc: "How many related notes the card stack shows.",
        control: { type: "slider", key: "topK", min: 4, max: 30, step: 1 },
      },
      {
        name: "Minimum similarity",
        desc: "Hide notes scoring below this. Around 0.20 separates on-topic from unrelated.",
        aliases: ["threshold", "cutoff"],
        control: {
          type: "slider",
          key: "minSimilarity",
          min: 0,
          max: 0.9,
          step: 0.05,
          displayFormat: (v) => v.toFixed(2),
        },
      },
      {
        name: "Show summary line",
        desc: "Show a short topic label on each card, extracted locally. Rebuilds the index.",
        control: { type: "toggle", key: "showSummary" },
      },
      {
        name: "Show snippet",
        desc: "Show a one- to two-line preview. Used when the summary line is off.",
        control: {
          type: "toggle",
          key: "showSnippet",
          disabled: () => this.plugin.settings.showSummary,
        },
      },
      {
        name: "Show folder on cards",
        desc: "The note's folder, right-aligned on the pills row.",
        control: { type: "toggle", key: "showFolder" },
      },
      {
        name: "Show tag pills",
        desc: "The shared-tag pill on cards, e.g. #personal.",
        control: { type: "toggle", key: "showTagPills" },
      },
      {
        name: "Show why-related pills",
        desc: "The other reason pills on cards: co-cited, similar text, or the note that bridges two others. The Linked and Related chips stay either way.",
        control: { type: "toggle", key: "showWhyPills" },
      },
      {
        name: "Put linked notes on top",
        desc: "Cards you already link to or from sort above everything else; order inside each group is unchanged.",
        control: { type: "toggle", key: "linkedFirst" },
      },
      {
        name: "Open cards in a new tab",
        desc: "Where a plain click on a card opens the note. Cmd/ctrl click does the opposite of this setting; middle click always opens a new tab, and right click offers all choices.",
        control: { type: "toggle", key: "openInNewTab" },
      },
      {
        name: "Show last-edited time",
        desc: "Add a muted “edited Nd ago” line to each card.",
        control: { type: "toggle", key: "showRecency" },
      },
      {
        name: "Remember the pin across restarts",
        desc: "Keep the panel's pinned note when Obsidian restarts. Off: the pin clears on startup.",
        control: { type: "toggle", key: "pinPersist" },
      },
    ];
  }

  // --- Reader (4.0) ----------------------------------------------------------
  private readerItems(): SettingDefinition<SettingKey>[] {
    const rungOptions: Record<string, string> = { auto: "Auto (by this machine's memory)" };
    for (const r of RUNGS) rungOptions[r.id] = r.label;
    return [
      {
        name: "Status",
        searchable: false,
        render: (setting) => {
          const el = setting.descEl;
          this.readerStatusEl = el;
          // This render already reflects the current error state.
          this.readerErrorShown = Boolean(this.plugin.reader?.engineError());
          this.drawReaderStatus();
          return () => {
            if (this.readerStatusEl === el) this.readerStatusEl = null;
          };
        },
      },
      {
        name: "Enable the reader",
        desc: "Turning this on downloads the engine (~50 MB) and the selected model below, then reads notes only while Obsidian is idle. On networks that block downloads, run the command \"Reader: offline setup\".",
        control: { type: "toggle", key: "readerEnabled" },
      },
      {
        name: "Model",
        desc: "Auto picks by memory: large on big machines (~4.8 GB download, needs ~6 GB free), mid on 16 GB (~2.5 GB), small below that (~1.3 GB). The small model labels notes but cannot judge; judgment features stay off on it.",
        aliases: ["reader model", "Qwen"],
        control: { type: "dropdown", key: "readerRung", options: rungOptions },
      },
      {
        name: "Reading pace",
        desc: "How often the reader picks up the next note during idle time. It always pauses while you type and while indexing runs.",
        control: {
          type: "dropdown",
          key: "readerPace",
          options: { light: "Light", balanced: "Balanced", fast: "Fast" },
        },
      },
      {
        name: "One-liners on cards",
        desc: "Replace the keyphrase label under each card title with the reader's one-line summary; the fuller summary appears on hover.",
        control: { type: "toggle", key: "readerOneLiners" },
      },
      {
        name: "Suggest tags that likely fit",
        desc: "Ghost chips above the panel for the active note. One click adds the tag to frontmatter; suggestions come only from tags your vault already uses.",
        control: { type: "toggle", key: "readerSuggestTags" },
      },
      {
        name: "Engine failed to load",
        searchable: false,
        visible: () => Boolean(this.plugin.reader?.engineError()),
        render: (setting) => {
          const err = this.plugin.reader?.engineError() ?? "";
          setting.setDesc(
            /block|fetch|network|certificat|offline/i.test(err)
              ? "This network blocks the downloads. Offline setup fetches the files through your browser instead and does the rest automatically."
              : "The error above decides the fix; retry after checking memory, or report it with the exact message.",
          );
          setting
            .addButton((b) =>
              b.setButtonText("Offline setup").onClick(() => {
                new ReaderOfflineModal(this.plugin).open();
              }),
            )
            .addButton((b) =>
              b.setButtonText("Retry").onClick(() => {
                void this.plugin.reader?.enable();
              }),
            );
        },
      },
      {
        name: "On this device",
        desc: "Reader files, stored outside the vault and never synced.",
        aliases: ["remove downloads", "disk space"],
        render: (setting) => {
          const sizes = assetSizesMb();
          const parts = [
            sizes.engine > 0 ? `engine ${Math.round(sizes.engine)} MB` : "",
            ...sizes.models.map((m) => `${m.file.replace(/\.gguf$/, "")} ${(m.mb / 1000).toFixed(1)} GB`),
          ].filter(Boolean);
          if (parts.length === 0) {
            setting.setDesc("Nothing downloaded.");
            return;
          }
          setting.setDesc(`${parts.join(" · ")}. Stored outside the vault, never synced.`);
          setting.addButton((b) =>
            b
              .setButtonText("Remove downloads")
              .setDestructive()
              .onClick(() => this.removeReaderDownloads()),
          );
        },
      },
    ];
  }

  private removeReaderDownloads(): void {
    this.plugin.settings.readerEnabled = false;
    this.debouncedSave();
    this.plugin.applyReaderSettings();
    const stopped = this.plugin.reader ? this.plugin.reader.disable() : Promise.resolve();
    void stopped.then(() => {
      removeAssets();
      this.update();
    });
  }

  // Called by the plugin on every reader status change. A change in the engine
  // error rebuilds the tab (the error row and the page's warning mark follow
  // it); anything else redraws the status row in place.
  updateReaderStatus(): void {
    const r = this.plugin.reader;
    if (!r) return;
    const errorNow = Boolean(r.engineError());
    if (errorNow !== this.readerErrorShown) {
      this.readerErrorShown = errorNow;
      this.update();
      return;
    }
    this.drawReaderStatus();
  }

  // The reader's one visible state row: what it is doing right now, with a
  // real progress bar while a download runs and a live counter while it reads.
  private drawReaderStatus(): void {
    const r = this.plugin.reader;
    if (!r || !this.readerStatusEl) return;
    const s = r.status;
    const host = this.readerStatusEl;
    host.empty();
    const line = host.createDiv({ cls: "rn-reader-state" });
    if (s.state === "off") {
      line.setText("Off. Nothing runs and nothing is downloaded until you enable it.");
      return;
    }
    if (s.state === "error") {
      line.setText(`Engine error: ${s.detail}`);
      line.addClass("rn-reader-state-error");
      return;
    }
    if (s.state === "downloading") {
      const pctText = s.pct !== null ? ` ${Math.round(s.pct * 100)}%` : "";
      line.setText(`${s.detail}${pctText}`);
      const bar = host.createDiv({ cls: "rn-reader-bar" });
      const fill = bar.createDiv({ cls: "rn-reader-bar-fill" });
      if (s.pct !== null) fill.setCssProps({ "--rn-reader-pct": `${Math.round(s.pct * 100)}%` });
      else fill.addClass("rn-reader-bar-busy");
      return;
    }
    // ready
    if (s.total > 0 && s.read >= s.total) {
      line.setText(`All ${s.total} notes read. Watching for changes.`);
    } else {
      line.setText(`Reading in the background: ${s.read} of ${s.total} notes read.`);
      const bar = host.createDiv({ cls: "rn-reader-bar" });
      const fill = bar.createDiv({ cls: "rn-reader-bar-fill rn-reader-bar-soft" });
      const pct = s.total > 0 ? Math.round((s.read / s.total) * 100) : 0;
      fill.setCssProps({ "--rn-reader-pct": `${pct}%` });
    }
  }

  // --- Linking: glow + the [[ suggester --------------------------------------
  private linkingItems(): SettingDefinition<SettingKey>[] {
    const glowOff = (): boolean => !this.plugin.settings.glowEnabled;
    const suggesterOff = (): boolean => !this.plugin.settings.suggesterEnabled;
    return [
      {
        name: "Highlight linkable mentions",
        desc: "Glow the first mention of a concept that already has a note. Click the glow to make it a [[wikilink]].",
        aliases: ["glow"],
        control: { type: "toggle", key: "glowEnabled" },
      },
      {
        name: "Highlight in live preview only",
        desc: "Skip the glow in raw source mode.",
        control: {
          type: "toggle",
          key: "glowRestrictToLivePreview",
          disabled: glowOff,
        },
      },
      {
        name: "Highlight ambiguous mentions",
        desc: "Also glow a phrase owned by more than one note. Off for precision.",
        control: {
          type: "toggle",
          key: "glowAmbiguous",
          disabled: glowOff,
        },
      },
      {
        name: "Auto-link later mentions",
        desc: "Once a note is linked here, link its remaining mentions in this note while you're idle.",
        control: { type: "toggle", key: "autoLinkSubsequent" },
      },
      {
        name: "Smart [[ suggestions",
        desc: "Rank [[ completions by meaning, not just by name.",
        aliases: ["suggester", "autocomplete"],
        control: { type: "toggle", key: "suggesterEnabled" },
      },
      {
        name: "Take over the [[ popup",
        desc: "Put these suggestions at the top of the popup. Off while the Easy Links suggester is active, so the two don't fight.",
        control: {
          type: "toggle",
          key: "suggesterTakeOver",
          disabled: suggesterOff,
        },
      },
      {
        name: "Suggest new notes",
        desc: "Offer to create a note for a strongly-relevant concept that doesn't have one yet.",
        control: {
          type: "toggle",
          key: "suggestNewNotes",
          disabled: suggesterOff,
        },
      },
      {
        name: "New-note confidence",
        desc: "How relevant a concept must be before a “create new note” row is offered.",
        control: {
          type: "slider",
          key: "newNoteMinSimilarity",
          min: 0,
          max: 0.9,
          step: 0.05,
          displayFormat: (v) => v.toFixed(2),
          disabled: () => suggesterOff() || !this.plugin.settings.suggestNewNotes,
        },
      },
    ];
  }

  // --- Scope: which notes take part ------------------------------------------
  private scopeItems(): SettingDefinition<SettingKey>[] {
    return [
      {
        name: "Excluded folders",
        desc: "Folders left out of the index entirely. One per line; includes everything beneath.",
        control: { type: "textarea", key: "excludeFolders", placeholder: "Templates\nArchive/2023", rows: 3 },
      },
      {
        name: "Folders excluded from link suggestions",
        desc: "Still indexed and ranked, but never suggested as inline [[links]]. One per line.",
        control: { type: "textarea", key: "excludeFoldersLinks", placeholder: "Daily\nAttachments/templates", rows: 3 },
      },
      {
        name: "Isolated areas (tag namespaces)",
        desc: "One tag namespace per line. Notes in an area only relate to each other, and never appear elsewhere.",
        control: { type: "textarea", key: "isolatedAreas", placeholder: "goa", rows: 3 },
      },
    ];
  }

  // --- Engine: how and where the model runs ----------------------------------
  private engineItems(): SettingDefinition<SettingKey>[] {
    const idleOptions = { ...IDLE_UNLOAD_OPTIONS };
    // Honour a hand-edited data.json value the list does not carry; otherwise
    // the control would show "Never" while the real timeout stays active.
    const current = String(this.plugin.settings.idleUnloadMinutes);
    if (!(current in idleOptions)) idleOptions[current] = `After ${current} minutes (custom)`;
    return [
      {
        name: "Compute device",
        desc: "Auto runs on the CPU and is memory-stable. WebGPU is faster per reindex but can grow memory until Obsidian crashes.",
        aliases: ["GPU", "WebGPU", "WASM"],
        control: { type: "dropdown", key: "device", options: DEVICE_OPTIONS },
      },
      {
        name: "Indexing speed",
        desc: "CPU threads used while indexing. Ignored by jina-v5-nano, which is faster on one. Changing this rebuilds the index.",
        aliases: ["threads"],
        control: {
          type: "dropdown",
          key: "indexSpeed",
          options: {
            light: "Light · 1 thread (slowest)",
            balanced: "Balanced (recommended)",
            fast: "Fast · all cores",
          },
        },
      },
      {
        name: "Unload model when idle",
        desc: "Free the model's memory after this long unused. It reloads in a few seconds, with no re-download.",
        aliases: ["memory"],
        control: { type: "dropdown", key: "idleUnloadMinutes", options: idleOptions },
      },
      {
        name: "Cached models",
        desc: "Models cached on this device. Remove one to free disk space after switching.",
        aliases: ["downloaded models", "disk space"],
        render: (setting) => this.renderModelCache(setting),
      },
    ];
  }

  // Models transformers.js has downloaded into the browser cache
  // ("transformers-cache"), with a remove action to free space after switching.
  private renderModelCache(setting: Setting): void {
    void (async () => {
      let ids: string[];
      try {
        const cache = await caches.open("transformers-cache");
        const set = new Set<string>();
        for (const req of await cache.keys()) {
          const m = /huggingface\.co\/([^/]+\/[^/]+)\/resolve\//.exec(req.url);
          if (m) set.add(m[1]);
        }
        ids = [...set].sort();
      } catch {
        if (setting.settingEl.isConnected) setting.setDesc("Could not read the model cache.");
        return;
      }
      if (!setting.settingEl.isConnected) return;
      if (ids.length === 0) {
        setting.setDesc("No models downloaded yet.");
        return;
      }
      const inUse = this.plugin.settings.modelId;
      let selected = ids.find((id) => id !== inUse) ?? ids[0];
      setting.addDropdown((d) => {
        for (const id of ids) d.addOption(id, id === inUse ? `${id} (in use)` : id);
        d.setValue(selected).onChange((v) => {
          selected = v;
        });
      });
      setting.addButton((b) =>
        b
          .setButtonText("Remove")
          .setDestructive()
          .onClick(async () => {
            try {
              const cache = await caches.open("transformers-cache");
              for (const req of await cache.keys()) {
                if (req.url.includes(`/${selected}/`)) await cache.delete(req);
              }
              new Notice(`Removed cached model: ${selected}`);
            } catch {
              new Notice("Could not remove the cached model.");
            }
            this.update();
          }),
      );
    })();
  }

  // --- Advanced matching: knobs most vaults never need -----------------------
  private advancedItems(): SettingDefinition<SettingKey>[] {
    const chunkingOff = (): boolean => !this.plugin.settings.chunking;
    return [
      {
        name: "Idea influence",
        desc: "How much whole-idea overlap counts against a single matching passage. Re-ranks live, no re-index.",
        control: {
          type: "slider",
          key: "ideaInfluence",
          min: 0,
          max: 0.6,
          step: 0.05,
          displayFormat: (v) => v.toFixed(2),
          disabled: chunkingOff,
        },
      },
      {
        name: "Link-graph influence",
        desc: "Also rank by how your links connect notes, not just by what they say. Surfaces related notes the wording alone would miss. 0 disables it.",
        control: {
          type: "slider",
          key: "graphInfluence",
          min: 0,
          max: 1,
          step: 0.1,
          displayFormat: (v) => v.toFixed(1),
        },
      },
      {
        name: "Structure influence",
        desc: "How much shared tags, links, and frontmatter nudge near-ties. 0 disables it.",
        control: {
          type: "slider",
          key: "structureInfluence",
          min: 0,
          max: 0.3,
          step: 0.01,
          displayFormat: (v) => v.toFixed(2),
        },
      },
      {
        name: "Chunk-level matching",
        desc: "Embed each note as several chunks instead of one vector, far better on long notes. Rebuilds the index.",
        control: { type: "toggle", key: "chunking" },
      },
      {
        name: "Max chunks per note",
        desc: "Ceiling on chunks embedded per note. Only very long notes reach it. Rebuilds the index.",
        control: {
          type: "slider",
          key: "maxChunks",
          min: 8,
          max: 64,
          step: 1,
          disabled: chunkingOff,
        },
      },
      {
        name: "Heading context",
        desc: "Prefix each section with its heading breadcrumb when embedding. Rebuilds the index.",
        control: {
          type: "toggle",
          key: "headingContext",
          disabled: chunkingOff,
        },
      },
      {
        name: "Shortlist size",
        desc: "How many candidates get the precise chunk comparison on each note switch.",
        control: { type: "slider", key: "shortlistSize", min: 20, max: 150, step: 10 },
      },
    ];
  }

  hide(): void {
    // Persist any value typed or dragged right before the pane closed, and
    // refresh the summaries that text input does not rebuild.
    this.debouncedSave.run();
    super.hide();
    this.update();
  }
}

// Guided offline setup for networks that block the plugin's own downloads
// (corporate proxies, domain blocklists). One button opens every missing
// file in the system browser, which usually gets through where the plugin
// cannot; the modal then watches the Downloads folder and imports each file
// the moment it finishes, verified against pinned checksums. Reached from
// the command palette or from a failed-download error; the settings pane
// itself stays clean.
class ReaderOfflineModal extends Modal {
  private timer: number | null = null;
  private tried = new Set<string>();
  private itemsEl: HTMLElement | null = null;
  private watchEl: HTMLElement | null = null;
  private logEl: HTMLElement | null = null;
  private opening = false;

  constructor(private plugin: RelatedNotesPlugin) {
    super(plugin.app);
  }

  private rung(): RungSpec {
    return this.plugin.reader?.rung() ?? RUNGS[RUNGS.length - 1];
  }

  private renderItems(): void {
    if (!this.itemsEl) return;
    this.itemsEl.empty();
    for (const it of offlineItems(this.rung())) {
      const li = this.itemsEl.createEl("li");
      li.appendText(it.present ? "✓ " : "· ");
      li.createEl("a", { text: it.label, href: it.url });
      li.appendText(` (${it.sizeLabel})`);
      if (it.present) li.appendText(" (installed)");
      else if (it.optional) li.appendText(" (optional)");
    }
  }

  private allRequiredPresent(): boolean {
    return offlineItems(this.rung())
      .filter((i) => !i.optional)
      .every((i) => i.present);
  }

  private log(line: string): void {
    this.logEl?.createDiv({ text: line });
  }

  private finishIfComplete(): void {
    if (!this.allRequiredPresent()) return;
    if (this.timer !== null) window.clearInterval(this.timer);
    this.timer = null;
    this.watchEl?.setText("Everything is in place.");
    if (this.plugin.settings.readerEnabled) {
      new Notice("Reader files verified. Starting the engine.");
      void this.plugin.reader?.enable();
    } else {
      new Notice("Reader files verified. Enable the reader in settings to start.");
    }
    window.setTimeout(() => this.close(), 1200);
  }

  // One import at a time: a 5 GB model takes minutes, and the 3 s poll must
  // not start a second copy of the same file mid-import.
  private busy = false;

  private async scan(dir: string): Promise<void> {
    if (this.busy) return;
    this.busy = true;
    try {
      const results = await importFromFolder(
        dir,
        (key) => this.tried.has(key),
        (label, done, total) => {
          this.watchEl?.setText(total > 1 ? `${label} ${Math.round((done / total) * 100)}%` : `${label}…`);
        },
      );
      for (const r of results) {
        this.tried.add(r.key);
        this.log(`${r.outcome.ok ? "✓" : "✗"} ${r.outcome.file}: ${r.outcome.note}`);
      }
      if (results.length > 0) this.renderItems();
    } catch (e) {
      // Typically the macOS Downloads permission; stop polling and say so.
      if (this.timer !== null) window.clearInterval(this.timer);
      this.timer = null;
      this.watchEl?.setText(e instanceof Error ? e.message : String(e));
      this.busy = false;
      return;
    } finally {
      this.busy = false;
    }
    if (this.timer !== null) this.watchEl?.setText(`Watching ${dir}. Files import automatically as they finish.`);
    this.finishIfComplete();
  }

  onOpen(): void {
    this.titleEl.setText("Reader offline setup");
    const c = this.contentEl;
    c.createEl("p", {
      cls: "setting-item-description",
      text:
        "For networks that block the plugin's downloads. Your browser fetches the files instead (it usually gets through), and everything after that is automatic: each file is verified and installed the moment the download finishes.",
    });

    new Setting(c)
      .setName("1. Fetch the files")
      .setDesc("Opens every missing file in your browser. You can also open the links yourself, or download them on another computer and drop them into your Downloads folder here.")
      .addButton((b) =>
        b
          .setCta()
          .setButtonText("Open downloads in browser")
          .onClick(() => {
            if (this.opening) return;
            const missing = offlineItems(this.rung()).filter((i) => !i.present);
            if (missing.length === 0) {
              new Notice("Nothing is missing.");
              return;
            }
            this.opening = true;
            missing.forEach((it, i) => {
              window.setTimeout(() => {
                window.open(it.url);
                if (i === missing.length - 1) this.opening = false;
              }, i * 500);
            });
          }),
      )
      .addButton((b) =>
        b.setButtonText("Copy links").onClick(() => {
          const missing = offlineItems(this.rung()).filter((i) => !i.present);
          const urls = (missing.length > 0 ? missing : offlineItems(this.rung())).map((i) => i.url);
          void navigator.clipboard.writeText(urls.join("\n"));
          new Notice("Download links copied.");
        }),
      );
    this.itemsEl = c.createEl("ul", { cls: "setting-item-description" });
    this.renderItems();

    new Setting(c)
      .setName("2. Wait")
      .setDesc("Imports run by themselves; nothing else to click. Renamed downloads are fine, and wrong files are refused.");
    this.watchEl = c.createEl("p", { cls: "setting-item-description" });
    this.logEl = c.createDiv({ cls: "setting-item-description" });

    const picker = c.createEl("input", { type: "file" });
    picker.multiple = true;
    picker.accept = ".tgz,.gz,.gguf";
    picker.hidden = true;
    picker.onchange = async () => {
      const files = Array.from(picker.files ?? []);
      picker.value = "";
      if (files.length === 0) return;
      if (this.busy) {
        new Notice("An import is already running; it finishes first.");
        return;
      }
      this.busy = true;
      const results = await importAssetFiles(
        files.map((f) => ({ name: f.name, size: f.size, stream: () => f.stream() })),
        (label, done, total) => {
          this.watchEl?.setText(total > 1 ? `${label} ${Math.round((done / total) * 100)}%` : `${label}…`);
        },
      );
      this.busy = false;
      for (const r of results) this.log(`${r.ok ? "✓" : "✗"} ${r.file}: ${r.note}`);
      this.renderItems();
      this.finishIfComplete();
    };
    const alt = c.createEl("p", { cls: "setting-item-description" });
    alt.appendText("Files somewhere else? ");
    const altBtn = alt.createEl("a", { text: "Import them by hand" });
    altBtn.onclick = () => picker.click();

    const dir = downloadsDir();
    this.watchEl.setText(`Watching ${dir} — files import automatically as they finish.`);
    void this.scan(dir);
    this.timer = window.setInterval(() => void this.scan(dir), 3000);
  }

  onClose(): void {
    if (this.timer !== null) window.clearInterval(this.timer);
    this.timer = null;
    this.contentEl.empty();
  }
}
