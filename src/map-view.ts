import { ItemView, TFile, setIcon, type WorkspaceLeaf } from "obsidian";
import type RelatedNotesPlugin from "./main";
import { buildVaultMap, type VaultMap } from "./vault-map";

export const VIEW_TYPE_MAP = "smart-related-notes-map";

// Cluster colours. Chosen to stay distinguishable on both Obsidian themes and to
// survive the common forms of colour blindness reasonably well (no red/green
// pair carrying meaning on its own; the label text carries it too).
const CLUSTER_COLORS = [
  "#5fe3a3", "#a6b0f5", "#f7d68a", "#6fe0ea", "#e09aab", "#b79af0",
  "#8ecfa0", "#8aa9e6", "#e6b36f", "#7fd6c8", "#d69ac0", "#a3c78a",
  "#c8a6f0", "#6fc4b0", "#eaa98a", "#94b8d6",
];

// =============================================================================
// VAULT MAP VIEW.
//
// A scatter of the whole vault, drawn as inline SVG so it stays crisp at any zoom
// and every point is a real DOM node we can attach a click handler to. Rendering
// is deliberately dumb: buildVaultMap() does the maths once, this only paints.
// =============================================================================
export class VaultMapView extends ItemView {
  private map: VaultMap | null = null;
  private hidden = new Set<number>();

  constructor(leaf: WorkspaceLeaf, private plugin: RelatedNotesPlugin) {
    super(leaf);
  }

  getViewType(): string { return VIEW_TYPE_MAP; }
  getDisplayText(): string { return "Vault map"; }
  getIcon(): string { return "scatter-chart"; }

  async onOpen(): Promise<void> {
    this.contentEl.addClass("rn-map-view");
    this.rebuild();
  }

  async onClose(): Promise<void> {
    this.contentEl.empty();
  }

  /** Recompute from the current index and repaint. */
  rebuild(): void {
    const input = this.plugin.store.vaultMapInput();
    this.map = input.length >= 3 ? buildVaultMap(input) : null;
    this.hidden.clear();
    this.render();
  }

  private render(): void {
    const el = this.contentEl;
    el.empty();

    const header = el.createDiv({ cls: "rn-map-header" });
    header.createDiv({ cls: "rn-map-title", text: "Vault map" });
    const refresh = header.createEl("button", { cls: "rn-map-refresh" });
    setIcon(refresh, "refresh-cw");
    refresh.setAttribute("aria-label", "Rebuild the map");
    refresh.addEventListener("click", () => this.rebuild());

    if (!this.map || this.map.points.length === 0) {
      el.createDiv({
        cls: "rn-map-empty",
        text:
          this.plugin.store.getProgress().status === "ready"
            ? "Not enough indexed notes yet to draw a map."
            : "Waiting for the index to finish building.",
      });
      return;
    }

    // Legend doubles as the filter: clicking a cluster hides or shows it.
    const body = el.createDiv({ cls: "rn-map-body" });
    const plot = body.createDiv({ cls: "rn-map-plot" });
    const legend = body.createDiv({ cls: "rn-map-legend" });
    legend.createDiv({ cls: "rn-map-legend-head", text: "Clusters" });
    for (const cluster of this.map.clusters) {
      const chip = legend.createDiv({ cls: "rn-map-chip" });
      if (this.hidden.has(cluster.id)) chip.addClass("is-off");
      chip.createDiv({ cls: "rn-map-swatch" }).style.backgroundColor =
        CLUSTER_COLORS[cluster.id % CLUSTER_COLORS.length];
      const meta = chip.createDiv({ cls: "rn-map-chip-meta" });
      meta.createDiv({ cls: "rn-map-chip-name", text: cluster.label });
      meta.createDiv({ cls: "rn-map-chip-count", text: `${cluster.size} notes` });
      chip.addEventListener("click", () => {
        if (this.hidden.has(cluster.id)) this.hidden.delete(cluster.id);
        else this.hidden.add(cluster.id);
        this.render();
      });
    }

    const W = 1000, H = 700, PAD = 28;
    const svg = plot.createSvg("svg", {
      cls: "rn-map-svg",
      attr: { viewBox: `0 0 ${W} ${H}` },
    });

    const sx = (x: number): number => PAD + x * (W - 2 * PAD);
    const sy = (y: number): number => PAD + y * (H - 2 * PAD);
    const visible = this.map.points.filter((p) => !this.hidden.has(p.cluster));

    for (const p of visible) {
      const circle = svg.createSvg("circle", {
        cls: "rn-map-dot",
        attr: {
          cx: sx(p.x).toFixed(1),
          cy: sy(p.y).toFixed(1),
          r: "4.5",
          fill: CLUSTER_COLORS[p.cluster % CLUSTER_COLORS.length],
        },
      });
      circle.createSvg("title").textContent = p.title;
      circle.addEventListener("click", () => {
        const file = this.app.vault.getAbstractFileByPath(p.path);
        if (file instanceof TFile) void this.app.workspace.getLeaf(false).openFile(file);
      });
    }

    // One label per visible cluster, placed at its centre of mass. Drawn last so
    // it sits above the points, with a halo so it stays readable over them.
    for (const cluster of this.map.clusters) {
      if (this.hidden.has(cluster.id)) continue;
      const members = visible.filter((p) => p.cluster === cluster.id);
      if (members.length < 3) continue;
      const cx = members.reduce((s, p) => s + sx(p.x), 0) / members.length;
      const cy = members.reduce((s, p) => s + sy(p.y), 0) / members.length;
      // Two copies: a thick stroked one underneath as a halo, then the fill on top.
      // paint-order would be neater but is not reliable across the Electron versions
      // Obsidian ships, and a label that vanishes into the points is worse than a
      // slightly heavier one.
      for (const cls of ["rn-map-label-halo", "rn-map-label"]) {
        svg.createSvg("text", {
          cls,
          attr: { x: cx.toFixed(1), y: cy.toFixed(1), "text-anchor": "middle" },
        }).textContent = cluster.label;
      }
    }

    plot.createDiv({
      cls: "rn-map-foot",
      text: `${visible.length} of ${this.map.points.length} notes. Click a point to open it, or a cluster to hide it.`,
    });
  }
}
