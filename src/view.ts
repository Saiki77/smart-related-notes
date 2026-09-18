import {
  ItemView,
  WorkspaceLeaf,
  TFile,
  Keymap,
  Menu,
  Notice,
  setIcon,
  debounce,
  type Debouncer,
} from "obsidian";
import type RelatedNotesPlugin from "./main";
import type {
  IndexProgress,
  RankedNote,
  WhyReason,
} from "./index-store";

export const VIEW_TYPE_RELATED = "smart-related-notes";

// The left-sidebar card stack. Subscribes to the index store's progress so its
// status line tracks indexing live, and re-ranks (debounced) whenever the active
// note changes. Because the plugin keeps ONE stable IndexStore (swapping the
// engine in place on a model change rather than replacing the store), this
// subscription stays valid for the lifetime of the view.
export class RelatedNotesView extends ItemView {
  private readonly plugin: RelatedNotesPlugin;
  private listEl!: HTMLElement;
  private subtitleEl!: HTMLElement;
  private statusEl!: HTMLElement;
  private unsubscribe: (() => void) | null = null;

  // Re-rendering is debounced so a flurry of active-leaf-change events (e.g. fast
  // tab switching) collapses into one ranking pass.
  private readonly scheduleRender: Debouncer<[], void>;

  // Search box state (toggled by the header search icon).
  private searchRowEl!: HTMLElement;
  private searchInputEl!: HTMLInputElement;
  private searchQuery = "";
  private searchSeq = 0; // guards against stale async query results
  private readonly scheduleSearch: Debouncer<[], void>;

  // "Linked notes" mode (toggled by the header link icon): instead of related notes,
  // show the notes the active note LINKS TO — i.e. a MOC's members — as cards.
  private linksMode = false;
  private linksToggleEl!: HTMLElement;

  // Pinned ranking source (toggled by the header pin icon): freeze the panel to
  // one note so opening results does not re-anchor it. Holds the TFile instance,
  // which Obsidian mutates in place on rename — so a pin survives renames, and
  // is dropped when the file is deleted (validated on every render).
  private pinnedFile: TFile | null = null;
  private pinToggleEl!: HTMLElement;

  // True while a search is re-initialising a cold (idle-unloaded) engine; owns
  // the status line so concurrent progress events can't clobber the hint.
  private modelWarmup = false;

  constructor(leaf: WorkspaceLeaf, plugin: RelatedNotesPlugin) {
    super(leaf);
    this.plugin = plugin;
    this.scheduleRender = debounce(() => this.render(), 300, false);
    this.scheduleSearch = debounce(() => void this.runSearch(), 250, false);
  }

  getViewType(): string {
    return VIEW_TYPE_RELATED;
  }

  getDisplayText(): string {
    return "Smart related notes";
  }

  getIcon(): string {
    return "sparkles";
  }

  protected async onOpen(): Promise<void> {
    const root = this.contentEl;
    root.empty();
    root.addClass("related-notes-view");

    const header = root.createDiv({ cls: "rn-header" });
    const titleRow = header.createDiv({ cls: "rn-title-row" });
    titleRow.createDiv({ cls: "rn-heading", text: "Smart related notes" });
    const actions = titleRow.createDiv({ cls: "rn-actions" });
    const searchToggle = actions.createDiv({
      cls: "rn-search-toggle clickable-icon",
      attr: { "aria-label": "Search notes" },
    });
    setIcon(searchToggle, "search");
    searchToggle.addEventListener("click", () => this.toggleSearch());
    this.linksToggleEl = actions.createDiv({
      cls: "rn-links-toggle clickable-icon",
      attr: { "aria-label": "Show notes this note links to (MOC members)" },
    });
    setIcon(this.linksToggleEl, "link");
    this.linksToggleEl.addEventListener("click", () => this.toggleLinks());
    this.pinToggleEl = actions.createDiv({
      cls: "rn-pin-toggle clickable-icon",
      attr: { "aria-label": "Pin the panel to the current note" },
    });
    setIcon(this.pinToggleEl, "pin");
    this.pinToggleEl.addEventListener("click", () => this.togglePin());
    const refresh = actions.createDiv({
      cls: "rn-refresh clickable-icon",
      attr: { "aria-label": "Rebuild the index" },
    });
    setIcon(refresh, "refresh-cw");
    refresh.addEventListener("click", () => {
      void this.plugin.rebuildIndex();
    });

    // Hidden until the search icon is clicked. Typing runs a semantic search.
    this.searchRowEl = header.createDiv({ cls: "rn-search-row" });
    this.searchInputEl = this.searchRowEl.createEl("input", {
      cls: "rn-search-input",
      attr: {
        type: "text",
        placeholder: "Search notes by meaning…",
        "aria-label": "Search notes",
        spellcheck: "false",
      },
    });
    this.searchInputEl.addEventListener("input", () => {
      this.searchQuery = this.searchInputEl.value.trim();
      if (!this.searchQuery) {
        this.scheduleSearch.cancel();
        this.render();
      } else {
        this.scheduleSearch();
      }
    });
    this.searchInputEl.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        e.preventDefault();
        this.clearSearch();
      }
    });

    this.subtitleEl = header.createDiv({ cls: "rn-subtitle" });
    this.statusEl = header.createDiv({ cls: "rn-status" });

    this.listEl = root.createDiv({ cls: "rn-list" });

    // Restore a persisted pin (opt-in setting), resolved by path once here;
    // from then on the pin tracks the TFile instance as usual. A path that
    // does not resolve is skipped, not cleared: during a slow startup the
    // vault may simply not be done loading, and a genuinely deleted note's
    // stored path is overwritten by the next pin anyway.
    if (this.plugin.settings.pinPersist && this.plugin.settings.pinnedPath) {
      const f = this.app.vault.getAbstractFileByPath(this.plugin.settings.pinnedPath);
      if (f instanceof TFile && f.extension === "md") this.setPin(f);
    }

    // Live status line: track the index store's progress.
    this.unsubscribe = this.plugin.store.onProgress((p) => this.renderStatus(p));
    this.renderStatus(this.plugin.store.getProgress());
    this.render();
  }

  protected async onClose(): Promise<void> {
    this.unsubscribe?.();
    this.unsubscribe = null;
    // Cancel armed debouncers so a timer can't fire into the torn-down view.
    this.scheduleRender.cancel();
    this.scheduleSearch.cancel();
  }

  // Public so the plugin can poke it on active-leaf-change.
  requestRender(): void {
    this.scheduleRender();
  }

  private renderStatus(p: IndexProgress): void {
    if (!this.statusEl) return;
    this.statusEl.empty();
    // A cold-engine search is warming the model up: that hint owns the status
    // line until runSearch clears the flag (progress events must not clobber it).
    if (this.modelWarmup) {
      this.statusEl.setText("Loading the embedding model…");
      this.statusEl.removeClass("rn-status-error");
      return;
    }
    if (p.status === "building") {
      const pct = p.total > 0 ? Math.round((p.done / p.total) * 100) : 0;
      this.statusEl.setText(`Indexing… ${pct}% (${p.done}/${p.total})`);
      this.statusEl.removeClass("rn-status-error");
    } else if (p.status === "loading") {
      this.statusEl.setText("Loading index…");
      this.statusEl.removeClass("rn-status-error");
    } else if (p.status === "error") {
      this.statusEl.setText(p.message ? `Indexing failed: ${p.message}` : "Indexing failed");
      this.statusEl.addClass("rn-status-error");
    } else if (p.status === "ready") {
      this.statusEl.setText(`${p.done} notes indexed`);
      this.statusEl.removeClass("rn-status-error");
    } else {
      this.statusEl.setText("");
    }
    // When the index just turned ready, the active note may now have a real vector;
    // refresh the cards so the fallback ranking is replaced by semantic results.
    if (p.status === "ready") this.scheduleRender();
  }

  // The note the panel ranks against: the pinned note when a valid pin is set,
  // the active note otherwise. A stale pin (file deleted, or replaced by a new
  // file at the same path) clears itself.
  private anchorFile(): TFile | null {
    if (this.pinnedFile) {
      const current = this.app.vault.getAbstractFileByPath(this.pinnedFile.path);
      if (current === this.pinnedFile) {
        this.persistPin();
        return this.pinnedFile;
      }
      this.setPin(null);
    }
    return this.app.workspace.getActiveFile();
  }

  // Keep the stored path in step with the pin when persistence is on: set or
  // cleared with the pin, refreshed after a rename, and written when the
  // setting is turned on while a pin is already held. No-op otherwise.
  private persistPin(): void {
    const stored = this.pinnedFile && this.plugin.settings.pinPersist ? this.pinnedFile.path : "";
    if (this.plugin.settings.pinnedPath !== stored) {
      this.plugin.settings.pinnedPath = stored;
      void this.plugin.saveSettings();
    }
  }

  private setPin(file: TFile | null): void {
    this.pinnedFile = file;
    this.pinToggleEl.toggleClass("is-active", file !== null);
    this.pinToggleEl.setAttr(
      "aria-label",
      file ? `Pinned to ${file.basename} (click to unpin)` : "Pin the panel to the current note",
    );
    this.persistPin();
  }

  private togglePin(): void {
    if (this.pinnedFile) {
      this.setPin(null);
    } else {
      const active = this.app.workspace.getActiveFile();
      if (!active || active.extension !== "md") {
        new Notice("Open a note to pin the panel to it.");
        return;
      }
      this.setPin(active);
    }
    this.render();
  }


  // Card navigation, matching Obsidian's own panes: plain click opens in the
  // current leaf, cmd/ctrl click in a new tab (cmd+alt in a split), middle
  // click in a new tab, and right click offers the same choices as a menu.
  private bindCardOpen(card: HTMLElement, file: TFile): void {
    card.addEventListener("click", (evt) => {
      // The default target is a setting; a mod-click flips between here and
      // a new tab relative to it. Split/window mod-combos pass through.
      const mod = Keymap.isModEvent(evt);
      const newTab = this.plugin.settings.openInNewTab;
      const target =
        mod === false ? (newTab ? "tab" : false) : mod === "tab" ? (newTab ? false : "tab") : mod;
      void this.app.workspace.getLeaf(target).openFile(file);
    });
    card.addEventListener("auxclick", (evt) => {
      if (evt.button === 1) void this.app.workspace.getLeaf("tab").openFile(file);
    });
    card.addEventListener("contextmenu", (evt) => {
      evt.preventDefault();
      const menu = new Menu();
      menu.addItem((i) =>
        i.setTitle("Open in new tab").onClick(() => {
          void this.app.workspace.getLeaf("tab").openFile(file);
        }),
      );
      menu.addItem((i) =>
        i.setTitle("Open to the right").onClick(() => {
          void this.app.workspace.getLeaf("split").openFile(file);
        }),
      );
      menu.addItem((i) =>
        i.setTitle("Open here").onClick(() => {
          void this.app.workspace.getLeaf(false).openFile(file);
        }),
      );
      menu.showAtMouseEvent(evt);
    });
  }

  render(): void {
    if (!this.listEl) return;
    // A search is active — its (async) results own the list; don't clobber them.
    if (this.searchQuery) return;
    const active = this.anchorFile();
    this.listEl.empty();

    if (!active || active.extension !== "md") {
      if (this.linksMode) {
        this.subtitleEl.setText("Open a note to see what it links to");
        this.renderEmpty("No active note.");
        return;
      }
      this.renderRecent();
      return;
    }

    if (this.linksMode) {
      this.renderLinks(active);
      return;
    }

    this.subtitleEl.empty();
    this.subtitleEl.appendText("Based on ");
    this.subtitleEl.createSpan({ cls: "rn-based-on", text: active.basename });
    if (this.pinnedFile) this.subtitleEl.appendText(" (pinned)");

    this.renderTagChips(active);

    const ranked = this.plugin.store.rank(active);
    if (ranked.length === 0) {
      const status = this.plugin.store.getProgress().status;
      this.renderEmpty(
        status === "building" || status === "loading"
          ? "Indexing… related notes will appear here."
          : "No related notes found.",
      );
      return;
    }

    // Optional hard priority for the user's own links: linked cards first,
    // order inside each group untouched (rank() already ordered them).
    const ordered = this.plugin.settings.linkedFirst
      ? [...ranked.filter((r) => r.connection === "linked"), ...ranked.filter((r) => r.connection !== "linked")]
      : ranked;
    let prevPct: number | null = null;
    for (const item of ordered) {
      this.renderCard(item, prevPct);
      prevPct = Math.round(item.score * 100);
    }
  }

  private renderEmpty(text: string): void {
    this.listEl.createDiv({ cls: "rn-empty", text });
  }

  // Ghost tag chips: reader-verified suggestions for the active note. One
  // click writes the tag into frontmatter; the small x dismisses it for good.
  private renderTagChips(active: TFile): void {
    if (!this.plugin.settings.readerSuggestTags) return;
    const tags = this.plugin.reader?.suggestedTags(active.path) ?? [];
    if (tags.length === 0) return;
    const row = this.listEl.createDiv({ cls: "rn-tagchips" });
    row.createSpan({ cls: "rn-tagchips-label", text: "Suggested:" });
    for (const tag of tags) {
      const chip = row.createSpan({ cls: "rn-chip" });
      chip.createSpan({ text: `#${tag}` });
      chip.setAttr("aria-label", `Add #${tag} to this note`);
      chip.addEventListener("click", (e) => {
        e.stopPropagation();
        void this.app.fileManager.processFrontMatter(active, (fm: Record<string, unknown>) => {
          const cur = fm.tags;
          const list = Array.isArray(cur) ? cur : typeof cur === "string" && cur.length > 0 ? [cur] : [];
          if (!list.map(String).map((t) => t.toLowerCase()).includes(tag)) list.push(tag);
          fm.tags = list;
        });
        this.plugin.reader?.dismissTag(active.path, tag);
        this.requestRender();
      });
      const x = chip.createSpan({ cls: "rn-chip-x", text: "×" });
      x.setAttr("aria-label", `Dismiss #${tag}`);
      x.addEventListener("click", (e) => {
        e.stopPropagation();
        this.plugin.reader?.dismissTag(active.path, tag);
        this.requestRender();
      });
    }
  }

  private toggleSearch(): void {
    const willShow = !this.searchRowEl.hasClass("is-visible");
    this.searchRowEl.toggleClass("is-visible", willShow);
    if (willShow) {
      window.setTimeout(() => this.searchInputEl.focus(), 0);
    } else {
      this.scheduleSearch.cancel();
      this.searchInputEl.value = "";
      this.searchQuery = "";
      this.render();
    }
  }

  private clearSearch(): void {
    this.scheduleSearch.cancel();
    this.searchInputEl.value = "";
    this.searchQuery = "";
    this.searchRowEl.removeClass("is-visible");
    this.render();
  }

  // Toggle "linked notes" mode. Mutually exclusive with search (leaving search if open).
  private toggleLinks(): void {
    this.linksMode = !this.linksMode;
    this.linksToggleEl.toggleClass("is-active", this.linksMode);
    if (this.searchRowEl.hasClass("is-visible") || this.searchQuery) {
      this.scheduleSearch.cancel();
      this.searchInputEl.value = "";
      this.searchQuery = "";
      this.searchRowEl.removeClass("is-visible");
    }
    this.render();
  }

  // Show the notes the active note links to (resolved wikilinks) as cards. For a MOC
  // this is its member list — the structural counterpart to similarity ranking.
  private renderLinks(active: TFile): void {
    this.subtitleEl.empty();
    this.subtitleEl.appendText("Linked from ");
    this.subtitleEl.createSpan({ cls: "rn-based-on", text: active.basename });
    if (this.pinnedFile) this.subtitleEl.appendText(" (pinned)");

    const resolved = this.app.metadataCache.resolvedLinks[active.path] ?? {};
    const targets: TFile[] = [];
    const seen = new Set<string>();
    for (const path of Object.keys(resolved)) {
      if (seen.has(path)) continue;
      seen.add(path);
      const f = this.app.vault.getAbstractFileByPath(path);
      if (f instanceof TFile && f.extension === "md") targets.push(f);
    }
    if (targets.length === 0) {
      this.renderEmpty("This note doesn't link to any notes yet.");
      return;
    }
    for (const file of targets) this.renderLinkCard(file);
  }

  private renderLinkCard(file: TFile): void {
    const card = this.listEl.createDiv({ cls: "rn-card" });
    const top = card.createDiv({ cls: "rn-card-top" });
    top.createDiv({ cls: "rn-title", text: file.basename });
    const pills = card.createDiv({ cls: "rn-pills" });
    pills.createSpan({ cls: "rn-conn rn-conn-linked", text: "Linked" });
    const parentPath = file.parent?.path ?? "";
    if (this.plugin.settings.showFolder && parentPath.length > 0 && parentPath !== "/") {
      card.createDiv({ cls: "rn-path", text: parentPath });
    }
    this.bindCardOpen(card, file);
  }

  // Semantic search: rank notes by similarity to the typed query (keyword fallback
  // in the store when the engine isn't ready). Guarded against stale async results.
  private async runSearch(): Promise<void> {
    const query = this.searchQuery;
    if (!query) {
      this.render();
      return;
    }
    const seq = ++this.searchSeq;
    this.subtitleEl.empty();
    this.subtitleEl.appendText("Search: ");
    this.subtitleEl.createSpan({ cls: "rn-based-on", text: query });
    // A cold engine (first search, or unloaded after the idle timeout) must be
    // re-initialised before rankByQuery resolves — a few seconds during which the
    // await below would otherwise just hang silently. Explain the wait.
    const engineCold = !this.plugin.store.engineLoaded();
    if (engineCold) {
      this.modelWarmup = true;
      this.renderStatus(this.plugin.store.getProgress());
    }
    let results: RankedNote[] = [];
    try {
      results = await this.plugin.store.rankByQuery(query);
    } catch {
      results = [];
    } finally {
      // Restore the live index status once the (possible) warm-up is over.
      if (engineCold) {
        this.modelWarmup = false;
        this.renderStatus(this.plugin.store.getProgress());
      }
    }
    if (seq !== this.searchSeq || this.searchQuery !== query) return; // superseded
    this.listEl.empty();
    if (results.length === 0) {
      const status = this.plugin.store.getProgress().status;
      this.renderEmpty(
        status === "building" || status === "loading"
          ? "Indexing… search results will improve as notes are added."
          : "No matches found.",
      );
      return;
    }
    let prevSearchPct: number | null = null;
    for (const item of results) {
      this.renderCard(item, prevSearchPct);
      prevSearchPct = Math.round(item.score * 100);
    }
  }

  // With no active note, surface recent notes so the panel stays useful: recently
  // opened, falling back to recently modified for a vault with no open-history yet.
  private renderRecent(): void {
    const recent = this.recentNotes();
    if (recent.length === 0) {
      this.subtitleEl.setText("Open a note to see related notes");
      this.renderEmpty("No active note.");
      return;
    }
    this.subtitleEl.setText("Recent notes");
    for (const file of recent) this.renderRecentCard(file);
  }

  private recentNotes(): TFile[] {
    const out: TFile[] = [];
    const seen = new Set<string>();
    for (const path of this.app.workspace.getLastOpenFiles()) {
      if (seen.has(path)) continue;
      const f = this.app.vault.getAbstractFileByPath(path);
      if (f instanceof TFile && f.extension === "md") {
        seen.add(path);
        out.push(f);
        if (out.length >= this.plugin.settings.topK) break;
      }
    }
    if (out.length > 0) return out;
    // Fresh vault / no open-history: fall back to the most recently modified notes.
    return this.app.vault
      .getMarkdownFiles()
      .sort((a, b) => b.stat.mtime - a.stat.mtime)
      .slice(0, this.plugin.settings.topK);
  }

  private renderRecentCard(file: TFile): void {
    const card = this.listEl.createDiv({ cls: "rn-card" });
    const top = card.createDiv({ cls: "rn-card-top" });
    top.createDiv({ cls: "rn-title", text: file.basename });
    const parentPath = file.parent?.path ?? "";
    if (this.plugin.settings.showFolder && parentPath.length > 0 && parentPath !== "/") {
      card.createDiv({ cls: "rn-path", text: parentPath });
    }
    const rel = relativeTime(file.stat.mtime);
    if (rel) card.createDiv({ cls: "rn-recency", text: `edited ${rel}` });
    this.bindCardOpen(card, file);
  }

  private renderCard(item: RankedNote, prevPct: number | null = null): void {
    const card = this.listEl.createDiv({ cls: "rn-card" });

    const top = card.createDiv({ cls: "rn-card-top" });
    top.createDiv({ cls: "rn-title", text: item.file.basename });

    // Similarity pill: a "~" prefix flags the keyword fallback (approximate).
    const pct = Math.round(item.score * 100);
    const pill = top.createDiv({ cls: "rn-score" });
    pill.setText(`${item.approximate ? "~" : ""}${pct}%`);
    if (item.approximate) pill.addClass("rn-score-approx");
    // The pill is wording similarity; the ORDER is wording fused with the link
    // graph. So a card can legitimately sit below one with a lower number, and
    // in the panel that read as a bug: 20, 16, 13, 15, 21 down the list. A card
    // scoring higher than the one above it is there because the graph put it
    // there, which is the only thing that can reorder against the number. Say so
    // instead of leaving the reader to reconcile it.
    if (prevPct !== null && pct > prevPct) {
      pill.addClass("rn-score-linkranked");
      pill.setAttr(
        "aria-label",
        `${pct}% wording match, placed here by your links rather than by wording`,
      );
    }

    // Second row: connection/why pills on the left, folder path right-aligned
    // on the same line. keywordRank results carry no reason/connection — render
    // no pill rather than a wrong one.
    const parentPath = item.file.parent?.path ?? "";
    const hasPath =
      this.plugin.settings.showFolder && parentPath.length > 0 && parentPath !== "/";
    const showWhy =
      item.reason !== undefined &&
      (item.reason.kind === "shared-tags"
        ? this.plugin.settings.showTagPills
        : this.plugin.settings.showWhyPills);
    if (showWhy || item.connection || hasPath) {
      const pills = card.createDiv({ cls: "rn-pills" });
      if (item.connection === "linked") {
        pills.createSpan({ cls: "rn-conn rn-conn-linked", text: "Linked" });
      } else if (item.connection === "related") {
        pills.createSpan({ cls: "rn-conn", text: "Related" });
      }
      if (showWhy && item.reason) {
        const why = this.whyLabel(item.reason);
        // Skip a redundant "Linked" why when the connection pill already says it.
        if (why && !(item.reason.kind === "linked" && item.connection === "linked")) {
          pills.createSpan({ cls: "rn-why", text: why });
        }
      }
      if (hasPath) pills.createDiv({ cls: "rn-path rn-path-row", text: parentPath });
    }

    // Summary line: the reader's one-liner when it has read this note, else the
    // keyphrase label, else the snippet. The gist rides along as a hover title.
    if (this.plugin.settings.showSummary) {
      const one = this.plugin.settings.readerOneLiners
        ? this.plugin.reader?.oneLiner(item.file.path) ?? null
        : null;
      const summary = one ?? this.plugin.store.getSummary(item.file);
      const line = summary.length > 0 ? summary : this.plugin.getSnippet(item.file);
      if (line.length > 0) {
        const el = card.createDiv({ cls: "rn-snippet", text: line });
        const gist = one ? this.plugin.reader?.gist(item.file.path) : null;
        if (gist) el.setAttr("title", gist);
      }
    } else if (this.plugin.settings.showSnippet) {
      const snippet = this.plugin.getSnippet(item.file);
      if (snippet.length > 0) {
        card.createDiv({ cls: "rn-snippet", text: snippet });
      }
    }

    if (this.plugin.settings.showRecency) {
      const rel = relativeTime(item.file.stat.mtime);
      if (rel) card.createDiv({ cls: "rn-recency", text: `edited ${rel}` });
    }

    this.bindCardOpen(card, item.file);
  }

  // Human label for a why-reason. Names the top shared tag for the shared-tags kind.
  private whyLabel(reason: WhyReason): string {
    switch (reason.kind) {
      case "linked":
        return "Linked";
      case "shared-tags":
        return reason.detail ? `#${reason.detail}` : "Shared tags";
      case "co-cited":
        return "Co-cited";
      case "semantic":
        return "Similar text";
      // Surfaced by the link graph rather than by the prose: name the note that
      // bridges the two, because the similarity score alone would not explain
      // why this card is here.
      case "graph":
        return reason.detail ? `via ${reason.detail}` : "Shared context";
      default:
        return "";
    }
  }
}

// Compact "edited 3d ago"-style relative time from an mtime (ms). Returns "" for a
// missing/invalid timestamp.
function relativeTime(mtime: number): string {
  if (!mtime || !Number.isFinite(mtime)) return "";
  const diff = Date.now() - mtime;
  if (diff < 0) return "";
  const min = 60 * 1000;
  const hour = 60 * min;
  const day = 24 * hour;
  if (diff < hour) return "just now";
  if (diff < day) return `${Math.floor(diff / hour)}h ago`;
  if (diff < 30 * day) return `${Math.floor(diff / day)}d ago`;
  if (diff < 365 * day) return `${Math.floor(diff / (30 * day))}mo ago`;
  return `${Math.floor(diff / (365 * day))}y ago`;
}
