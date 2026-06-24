import {
  ItemView,
  WorkspaceLeaf,
  setIcon,
  debounce,
  type Debouncer,
} from "obsidian";
import type RelatedNotesPlugin from "./main";
import type { IndexProgress, RankedNote } from "./index-store";

export const VIEW_TYPE_RELATED = "related-notes";

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

  constructor(leaf: WorkspaceLeaf, plugin: RelatedNotesPlugin) {
    super(leaf);
    this.plugin = plugin;
    this.scheduleRender = debounce(() => this.render(), 300, false);
  }

  getViewType(): string {
    return VIEW_TYPE_RELATED;
  }

  getDisplayText(): string {
    return "Related notes";
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
    titleRow.createDiv({ cls: "rn-heading", text: "Related notes" });
    const refresh = titleRow.createDiv({
      cls: "rn-refresh clickable-icon",
      attr: { "aria-label": "Rebuild the index" },
    });
    setIcon(refresh, "refresh-cw");
    refresh.addEventListener("click", () => {
      void this.plugin.rebuildIndex();
    });

    this.subtitleEl = header.createDiv({ cls: "rn-subtitle" });
    this.statusEl = header.createDiv({ cls: "rn-status" });

    this.listEl = root.createDiv({ cls: "rn-list" });

    // Live status line: track the index store's progress.
    this.unsubscribe = this.plugin.store.onProgress((p) => this.renderStatus(p));
    this.renderStatus(this.plugin.store.getProgress());
    this.render();
  }

  protected async onClose(): Promise<void> {
    this.unsubscribe?.();
    this.unsubscribe = null;
  }

  // Public so the plugin can poke it on active-leaf-change.
  requestRender(): void {
    this.scheduleRender();
  }

  private renderStatus(p: IndexProgress): void {
    if (!this.statusEl) return;
    this.statusEl.empty();
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

  render(): void {
    if (!this.listEl) return;
    const active = this.app.workspace.getActiveFile();
    this.listEl.empty();

    if (!active || active.extension !== "md") {
      this.subtitleEl.setText("Open a note to see related notes");
      this.renderEmpty("No active note.");
      return;
    }

    this.subtitleEl.empty();
    this.subtitleEl.appendText("Based on ");
    this.subtitleEl.createSpan({ cls: "rn-based-on", text: active.basename });

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

    for (const item of ranked) this.renderCard(item);
  }

  private renderEmpty(text: string): void {
    this.listEl.createDiv({ cls: "rn-empty", text });
  }

  private renderCard(item: RankedNote): void {
    const card = this.listEl.createDiv({ cls: "rn-card" });

    const top = card.createDiv({ cls: "rn-card-top" });
    top.createDiv({ cls: "rn-title", text: item.file.basename });

    // Similarity pill: a "~" prefix flags the keyword fallback (approximate).
    const pct = Math.round(item.score * 100);
    const pill = top.createDiv({ cls: "rn-score" });
    pill.setText(`${item.approximate ? "~" : ""}${pct}%`);
    if (item.approximate) pill.addClass("rn-score-approx");

    const parentPath = item.file.parent?.path ?? "";
    if (parentPath.length > 0 && parentPath !== "/") {
      card.createDiv({ cls: "rn-path", text: parentPath });
    }

    if (this.plugin.settings.showSnippet) {
      const snippet = this.plugin.getSnippet(item.file);
      if (snippet.length > 0) {
        card.createDiv({ cls: "rn-snippet", text: snippet });
      }
    }

    card.addEventListener("click", () => {
      void this.app.workspace.getLeaf(false).openFile(item.file);
    });
  }
}
