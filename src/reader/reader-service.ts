// The reader service: background scheduler + artifact cache + the tiny sync
// API the panel reads. Invisible by design — it works only while Obsidian is
// idle, pauses for typing and indexing, caches everything, and re-reads a
// note only when its content changes.
import { TFile } from "obsidian";
import { assetsReady, autoRung, ensureAssets, RUNGS, RungSpec } from "./reader-assets";
import { ReaderEngine } from "./reader-engine";
import { detectLanguage, gistTask, NoteContextInput, oneLinerTask, pickTagsTask, TaskResult } from "./reader-tasks";

export interface ReaderHost {
  // Environment
  vaultRead(file: TFile): Promise<string>;
  markdownFiles(): TFile[];
  fileByPath(path: string): TFile | null;
  fileTags(path: string): string[]; // lowercased frontmatter+inline tags
  neighborPaths(path: string, k: number): Promise<string[]>;
  recentPaths(): string[];
  // Coordination
  indexBusy(): boolean; // building / embedding / persisting
  lastUserActivity(): number;
  requestRender(): void;
  // Persistence of the artifact file (kept out of data.json)
  loadArtifacts(): Promise<string | null>;
  saveArtifacts(json: string): Promise<void>;
  // Engine bundle bytes shipped inside the plugin folder
  readEngineBundle(): Promise<Uint8Array>;
}

interface Artifact {
  h: string; // srcHash = mtime:size at read time
  read?: boolean; // a full read pass (one-liner + gist) happened for this h
  one?: string;
  gist?: string;
  model: string;
  tags?: string[]; // verified suggestions (may be empty = none fit)
  tagsH?: string; // hash of (h + candidate set) the tags were computed for
  dismissed?: string[];
}

export type ReaderPace = "light" | "balanced" | "fast";
const TICK_MS: Record<ReaderPace, number> = { light: 45_000, balanced: 12_000, fast: 6_000 };
const IDLE_MS = 20_000;

export interface ReaderStatus {
  state: "off" | "downloading" | "ready" | "error";
  detail: string;
  read: number;
  total: number;
}

export class ReaderService {
  private engine: ReaderEngine | null = null;
  private artifacts = new Map<string, Artifact>();
  private timer: number | null = null;
  private busy = false;
  private saveQueued = false;
  private tagDf: Map<string, number> | null = null;
  private tagDfAt = 0;
  status: ReaderStatus = { state: "off", detail: "", read: 0, total: 0 };
  onStatus: (() => void) | null = null;

  constructor(
    private host: ReaderHost,
    private opts: { rung: "auto" | RungSpec["id"]; pace: ReaderPace; idleUnloadMinutes: number },
  ) {}

  rung(): RungSpec {
    return this.opts.rung === "auto" ? autoRung() : RUNGS.find((r) => r.id === this.opts.rung) ?? autoRung();
  }

  configure(opts: { rung: "auto" | RungSpec["id"]; pace: ReaderPace; idleUnloadMinutes: number }): void {
    const prevRung = this.rung();
    this.opts = opts;
    if (this.engine) this.engine.setRung(this.rung());
    if (this.timer !== null) this.startTicking(); // re-arm with the new pace
    if (prevRung.id !== this.rung().id && this.status.state === "ready" && !assetsReady(this.rung())) {
      void this.enable(); // new rung needs its weights
    }
  }

  private setStatus(s: Partial<ReaderStatus>): void {
    this.status = { ...this.status, ...s };
    this.onStatus?.();
  }

  // ---------------------------------------------------------- lifecycle ----

  async enable(): Promise<void> {
    const raw = await this.host.loadArtifacts();
    if (raw && this.artifacts.size === 0) {
      try {
        this.artifacts = new Map(Object.entries(JSON.parse(raw) as Record<string, Artifact>));
      } catch {
        this.artifacts = new Map();
      }
    }
    const rung = this.rung();
    try {
      if (!assetsReady(rung)) {
        this.setStatus({ state: "downloading", detail: "Preparing download" });
        await ensureAssets(() => this.host.readEngineBundle(), rung, (label, done, total) => {
          const pct = total > 1 ? ` ${Math.round((done / total) * 100)}%` : "";
          this.setStatus({ state: "downloading", detail: `${label}${pct}` });
        });
      }
      this.engine = new ReaderEngine(rung, this.opts.idleUnloadMinutes);
      this.setStatus({ state: "ready", detail: "" });
      this.refreshCounts();
      this.startTicking();
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      // Corporate proxies and blocklists surface here as fetch/TLS failures.
      // Name the way out instead of leaving a bare network error.
      const blocked = /fetch failed|failed to fetch|ENOTFOUND|ECONN|ETIMEDOUT|EAI_AGAIN|certificat|CERT_|self.signed|local issuer|ERR_TLS|ERR_NETWORK|ERR_CERT|403|407|HTTP 5/i.test(msg);
      this.setStatus({
        state: "error",
        detail: blocked
          ? `${msg} — this network seems to block downloads. Use "Offline setup" below: download the files in your browser, then import them.`
          : msg,
      });
    }
  }

  async disable(): Promise<void> {
    if (this.timer !== null) window.clearInterval(this.timer);
    this.timer = null;
    await this.engine?.unload();
    this.engine = null;
    this.setStatus({ state: "off", detail: "" });
  }

  engineError(): string | null {
    return this.status.state === "error" ? this.status.detail : this.engine?.lastError ?? null;
  }

  // ----------------------------------------------------------- schedule ----

  private startTicking(): void {
    if (this.timer !== null) window.clearInterval(this.timer);
    this.timer = window.setInterval(() => void this.tick(), TICK_MS[this.opts.pace]);
  }

  private srcHash(f: TFile): string {
    return `${f.stat.mtime}:${f.stat.size}`;
  }

  private upToDate(f: TFile): boolean {
    const a = this.artifacts.get(f.path);
    return a !== undefined && a.h === this.srcHash(f) && a.read === true;
  }

  private refreshCounts(): void {
    const files = this.host.markdownFiles();
    this.setStatus({ read: files.filter((f) => this.upToDate(f)).length, total: files.length });
  }

  private nextFile(): TFile | null {
    for (const p of this.host.recentPaths()) {
      const f = this.host.fileByPath(p);
      if (f && !this.upToDate(f)) return f;
    }
    for (const f of this.host.markdownFiles()) if (!this.upToDate(f)) return f;
    return null;
  }

  private async tick(): Promise<void> {
    if (this.busy || !this.engine || this.status.state !== "ready") return;
    if (this.host.indexBusy()) return;
    if (Date.now() - this.host.lastUserActivity() < IDLE_MS) return;
    const file = this.nextFile();
    if (!file) return;
    this.busy = true;
    try {
      await this.readNote(file);
      this.refreshCounts();
      this.host.requestRender();
    } catch (e) {
      console.warn("[related-notes] reader:", e);
    } finally {
      this.busy = false;
    }
  }

  private async runTask(t: TaskResult): Promise<string | string[] | null> {
    if (!this.engine) return null;
    const raw = await this.engine.generate(t.prompt, t.maxTokens);
    return t.validate(raw);
  }

  private async noteInput(file: TFile): Promise<NoteContextInput> {
    const raw = await this.host.vaultRead(file);
    const body = raw.replace(/^---\n[\s\S]*?\n---\n?/, "");
    return { title: file.basename, body, language: detectLanguage(body) };
  }

  private async readNote(file: TFile): Promise<void> {
    const n = await this.noteInput(file);
    const h = this.srcHash(file);
    const prev = this.artifacts.get(file.path);
    if (n.body.trim().length < 60) {
      // Stubs carry no gist worth writing; mark them read so the queue moves on.
      this.artifacts.set(file.path, { h, read: true, model: this.rung().id, dismissed: prev?.dismissed });
      this.queueSave();
      return;
    }
    const one = await this.runTask(oneLinerTask(n));
    const gist = await this.runTask(gistTask(n));
    this.artifacts.set(file.path, {
      h,
      read: true,
      one: typeof one === "string" ? one : undefined,
      gist: typeof gist === "string" ? gist : undefined,
      model: this.rung().id,
      dismissed: prev?.dismissed,
    });
    this.queueSave();
  }

  private queueSave(): void {
    if (this.saveQueued) return;
    this.saveQueued = true;
    window.setTimeout(() => {
      this.saveQueued = false;
      void this.host.saveArtifacts(JSON.stringify(Object.fromEntries(this.artifacts)));
    }, 4000);
  }

  // ---------------------------------------------------------- panel API ----

  oneLiner(path: string): string | null {
    const f = this.host.fileByPath(path);
    const a = this.artifacts.get(path);
    if (!f || !a || a.h !== this.srcHash(f)) return null;
    return a.one ?? null;
  }

  gist(path: string): string | null {
    const f = this.host.fileByPath(path);
    const a = this.artifacts.get(path);
    if (!f || !a || a.h !== this.srcHash(f)) return null;
    return a.gist ?? null;
  }

  dismissTag(path: string, tag: string): void {
    const a = this.artifacts.get(path);
    if (!a) return;
    a.dismissed = [...(a.dismissed ?? []), tag];
    a.tags = a.tags?.filter((t) => t !== tag);
    this.queueSave();
  }

  // Verified tag suggestions for the ACTIVE note. Nomination is structural
  // (discriminative tags carried by ranked neighbors); the reader only picks
  // from that closed set. Returns cached results synchronously; computes in
  // the background and re-renders when fresh ones land.
  suggestedTags(path: string): string[] {
    const f = this.host.fileByPath(path);
    const a = this.artifacts.get(path);
    if (!f || !this.engine || this.status.state !== "ready") return [];
    const cached = a && a.tagsH?.startsWith(this.srcHash(f)) ? a.tags ?? [] : [];
    if (a && a.tagsH?.startsWith(this.srcHash(f))) {
      return cached.filter((t) => !(a.dismissed ?? []).includes(t));
    }
    void this.computeTags(f);
    return [];
  }

  private tagDocFreq(): Map<string, number> {
    if (this.tagDf && Date.now() - this.tagDfAt < 300_000) return this.tagDf;
    const df = new Map<string, number>();
    for (const f of this.host.markdownFiles()) {
      for (const t of new Set(this.host.fileTags(f.path))) df.set(t, (df.get(t) ?? 0) + 1);
    }
    this.tagDf = df;
    this.tagDfAt = Date.now();
    return df;
  }

  private tagsInFlight = new Set<string>();

  private async computeTags(file: TFile): Promise<void> {
    if (this.tagsInFlight.has(file.path) || this.busy) return;
    // Invisibility rule: never spin the model up because the user LOOKED at a
    // note. Compute chips only when the engine is already warm, or the user
    // has gone idle anyway.
    if (!this.engine) return;
    if (!this.engine.loaded && Date.now() - this.host.lastUserActivity() < IDLE_MS) return;
    this.tagsInFlight.add(file.path);
    try {
      const df = this.tagDocFreq();
      const total = this.host.markdownFiles().length;
      const own = new Set(this.host.fileTags(file.path));
      const counts = new Map<string, number>();
      for (const p of await this.host.neighborPaths(file.path, 12)) {
        for (const t of new Set(this.host.fileTags(p))) {
          const d = df.get(t) ?? 0;
          if (d >= 3 && d <= total * 0.5 && !own.has(t)) counts.set(t, (counts.get(t) ?? 0) + 1);
        }
      }
      const candidates = [...counts.entries()]
        .filter(([, c]) => c >= 2)
        .sort((x, y) => y[1] - x[1])
        .slice(0, 8)
        .map(([t]) => t);
      const h = this.srcHash(file);
      const key = `${h}|${candidates.join(",")}`;
      let tags: string[] = [];
      if (candidates.length > 0) {
        const n = await this.noteInput(file);
        const picked = await this.runTask(pickTagsTask(n, candidates));
        tags = Array.isArray(picked) ? picked : [];
      }
      const prev = this.artifacts.get(file.path);
      const carry = prev?.h === h ? prev : { dismissed: prev?.dismissed };
      this.artifacts.set(file.path, { model: this.rung().id, ...carry, h, tags, tagsH: key });
      this.queueSave();
      this.host.requestRender();
    } catch (e) {
      console.warn("[related-notes] reader tags:", e);
    } finally {
      this.tagsInFlight.delete(file.path);
    }
  }
}
