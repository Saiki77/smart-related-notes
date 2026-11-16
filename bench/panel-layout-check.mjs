// The panel layouts, driven through the REAL src/view.ts: render(), the search
// path, the link and recent modes and the status line all run on a small fake
// DOM, and each test reads back what the panel drew.
//   node --experimental-transform-types --import ./bench/register-stub.mjs --test bench/panel-layout-check.mjs
import assert from "node:assert/strict";
import { test } from "node:test";
import { TFile } from "./obsidian-stub.mjs";

const { RelatedNotesView } = await import("../src/view.ts");
const { DEFAULT_SETTINGS } = await import("../src/main.ts");

// Just enough of Obsidian's HTMLElement helpers for the panel code.
class El {
  constructor(tag = "div", o = {}) {
    this.tag = tag;
    this.nodes = [];
    this.classes = new Set();
    this.attrs = {};
    this.listeners = {};
    for (const c of String(o.cls ?? "").split(/\s+/).filter(Boolean)) this.classes.add(c);
    if (o.text !== undefined) this.nodes.push(String(o.text));
    for (const [k, v] of Object.entries(o.attr ?? {})) this.attrs[k] = String(v);
  }
  get children() { return this.nodes.filter((n) => n instanceof El); }
  get textContent() { return this.nodes.map((n) => (n instanceof El ? n.textContent : n)).join(""); }
  createEl(tag, o = {}) { const el = new El(tag, o); this.nodes.push(el); return el; }
  createDiv(o = {}) { return this.createEl("div", o); }
  createSpan(o = {}) { return this.createEl("span", o); }
  setText(t) { this.nodes = t === "" ? [] : [String(t)]; }
  appendText(t) { this.nodes.push(String(t)); }
  empty() { this.nodes = []; }
  addClass(...c) { for (const x of c) this.classes.add(x); }
  removeClass(...c) { for (const x of c) this.classes.delete(x); }
  toggleClass(c, on) { if (on) this.classes.add(c); else this.classes.delete(c); }
  hasClass(c) { return this.classes.has(c); }
  setAttr(k, v) { this.attrs[k] = String(v); }
  getAttr(k) { return this.attrs[k] ?? null; }
  removeAttribute(k) { delete this.attrs[k]; }
  addEventListener(type, fn) { (this.listeners[type] ??= []).push(fn); }
  setCssProps() {}
  all(cls) {
    const out = [];
    const walk = (e) => { for (const c of e.children) { if (c.classes.has(cls)) out.push(c); walk(c); } };
    walk(this);
    return out;
  }
  one(cls) { return this.all(cls)[0] ?? null; }
  querySelector(sel) { return this.one(sel.replace(/^\./, "")); }
}

function note(path) {
  const f = new TFile(path, Date.now() - 3 * 3600_000);
  const dir = path.includes("/") ? path.slice(0, path.lastIndexOf("/")) : "/";
  f.parent = { path: dir };
  return f;
}
const active = note("Concepts/Eigenvektoren.md");
const eigenwerte = note("Concepts/Eigenwerte.md");
const basis = note("Concepts/Basiswechsel.md");
const pca = note("ML/PCA.md");
const matrix = note("Concepts/Matrix.md");
const inbox = note("Inbox.md");
const files = new Map([active, eigenwerte, basis, pca, matrix, inbox].map((f) => [f.path, f]));
const summaries = {
  Eigenwerte: "Both diagonalise symmetric matrices",
  Basiswechsel: "Same map, other coordinates",
  PCA: "Top eigenvectors of the covariance",
  Matrix: "Rectangular scheme behind every linear map",
};
// Matrix scores 45 but sits under PCA at 41: the link graph placed it there.
const ranked = [
  { file: eigenwerte, score: 0.51, approximate: false, connection: "related", reason: { kind: "semantic" } },
  { file: basis, score: 0.44, approximate: false, connection: "linked", reason: { kind: "linked" } },
  { file: pca, score: 0.41, approximate: false, connection: "related", reason: { kind: "graph", detail: "Kovarianzmatrix" } },
  { file: matrix, score: 0.45, approximate: false, connection: "related", reason: { kind: "shared-tags", detail: "mathe" } },
];

function makeView(overrides = {}, { activeFile = active } = {}) {
  const plugin = {
    settings: {
      ...DEFAULT_SETTINGS,
      showSummary: true,
      showRecency: false,
      readerSuggestTags: false,
      linkedFirst: false,
      showFolder: true,
      showWhyPills: true,
      showTagPills: true,
      ...overrides,
    },
    reader: null,
    store: {
      rank: () => ranked,
      rankByQuery: async () => ranked,
      engineLoaded: () => true,
      getProgress: () => ({ status: "ready", done: 147, total: 147, skipped: 0 }),
      getSummary: (f) => summaries[f.basename] ?? "",
    },
    getSnippet: () => "",
  };
  const view = new RelatedNotesView({}, plugin);
  view.app = {
    workspace: {
      getActiveFile: () => activeFile,
      getLastOpenFiles: () => [basis.path, inbox.path],
      getLeaf: () => ({ openFile: async () => {} }),
    },
    vault: {
      getAbstractFileByPath: (p) => files.get(p) ?? null,
      getMarkdownFiles: () => [...files.values()],
    },
    metadataCache: { resolvedLinks: { [active.path]: { [basis.path]: 1, [inbox.path]: 1 } } },
  };
  view.listEl = new El();
  view.subtitleEl = new El();
  view.countEl = new El();
  view.statusEl = new El();
  return { view, plugin };
}
const cards = (view) => view.listEl.children.filter((c) => c.hasClass("rn-card"));
const scoreHover = (card) => card.one("rn-score-text")?.getAttr("aria-label") ?? null;
const LAYOUTS = ["rows", "cards", "pills", "classic"];

test("the list carries exactly one layout class, and a change replaces it", () => {
  const { view, plugin } = makeView();
  for (const layout of LAYOUTS) {
    plugin.settings.cardLayout = layout;
    view.render();
    const on = LAYOUTS.filter((l) => view.listEl.hasClass(`rn-layout-${l}`));
    assert.deepEqual(on, [layout]);
  }
});

test("Rows is the default: title, quiet score and summary; the rest on hover", () => {
  assert.equal(DEFAULT_SETTINGS.cardLayout, "rows");
  const { view } = makeView();
  view.render();
  const list = cards(view);
  assert.equal(list.length, 4);
  for (const card of list) {
    for (const cls of ["rn-pills", "rn-pill", "rn-meta", "rn-path", "rn-conn", "rn-why", "rn-score"]) {
      assert.equal(card.one(cls), null, `rows must not draw ${cls}`);
    }
    assert.ok(card.one("rn-score-text"));
    assert.ok(card.one("rn-snippet"));
  }
  const [eig, bas, , mat] = list;
  assert.equal(eig.one("rn-title").textContent, "Eigenwerte");
  assert.equal(eig.one("rn-score-text").textContent, "51%");
  assert.equal(eig.one("rn-snippet").textContent, "Both diagonalise symmetric matrices");
  assert.equal(scoreHover(eig), "51% wording match · Similar text · Concepts");
  // A linked note: the icon before the title, and no second "Linked" anywhere.
  assert.ok(bas.one("rn-title").one("rn-linked-icon"));
  assert.equal(bas.one("rn-title").textContent, "Basiswechsel");
  assert.equal(eig.one("rn-linked-icon"), null);
  assert.equal(scoreHover(bas), "44% wording match · Concepts");
  assert.equal(scoreHover(mat), "45% wording match · #mathe · Concepts · placed here by your links rather than by wording");
});

test("Cards: one faint line under the summary, gated by the pill and folder settings", () => {
  const { view, plugin } = makeView({ cardLayout: "cards" });
  view.render();
  const meta = () => cards(view).map((c) => c.one("rn-meta")?.textContent ?? "");
  assert.deepEqual(meta(), ["Similar text · Concepts", "Linked · Concepts", "via Kovarianzmatrix · ML", "#mathe · Concepts"]);
  const [eig, bas] = cards(view);
  assert.ok(eig.children.indexOf(eig.one("rn-meta")) > eig.children.indexOf(eig.one("rn-snippet")));
  assert.ok(bas.one("rn-linked-icon"));
  assert.equal(scoreHover(eig), null, "everything is on the card: no hover needed");

  plugin.settings.showFolder = false;
  view.render();
  assert.deepEqual(meta(), ["Similar text", "Linked", "via Kovarianzmatrix", "#mathe"]);
  assert.equal(scoreHover(cards(view)[0]), "51% wording match · Concepts", "a hidden folder moves to the hover");

  plugin.settings.showFolder = true;
  plugin.settings.showWhyPills = false;
  view.render();
  assert.deepEqual(meta(), ["Concepts", "Linked · Concepts", "ML", "#mathe · Concepts"]);
  assert.equal(scoreHover(cards(view)[0]), "51% wording match · Similar text");

  plugin.settings.showWhyPills = true;
  plugin.settings.showTagPills = false;
  view.render();
  assert.equal(meta()[3], "Concepts");
});

test("Cards with pills: outline pills, no link icon, no filled 4.0 pills", () => {
  const { view } = makeView({ cardLayout: "pills" });
  view.render();
  const [eig, bas] = cards(view);
  assert.equal(view.listEl.one("rn-linked-icon"), null);
  assert.equal(view.listEl.one("rn-conn"), null);
  assert.equal(view.listEl.one("rn-score"), null);
  assert.deepEqual(bas.one("rn-pills").children.map((c) => c.textContent), ["Linked", "Concepts"]);
  assert.ok(bas.one("rn-pill-linked"));
  assert.deepEqual(eig.one("rn-pills").children.map((c) => c.textContent), ["Similar text", "Concepts"]);
  assert.equal(eig.one("rn-pill-linked"), null);
});

test("Classic keeps the 4.0 card", () => {
  const { view } = makeView({ cardLayout: "classic" });
  view.render();
  const [eig, bas, , mat] = cards(view);
  assert.equal(view.listEl.one("rn-score-text"), null);
  assert.equal(view.listEl.one("rn-linked-icon"), null);
  assert.equal(view.listEl.one("rn-meta"), null);
  assert.equal(eig.one("rn-score").textContent, "51%");
  assert.equal(eig.one("rn-conn").textContent, "Related");
  assert.equal(eig.one("rn-why").textContent, "Similar text");
  assert.equal(eig.one("rn-path").textContent, "Concepts");
  assert.ok(bas.one("rn-conn-linked"));
  assert.ok(mat.one("rn-score").hasClass("rn-score-linkranked"));
});

test("search results use the chosen layout", async () => {
  const { view } = makeView({ cardLayout: "pills" });
  view.searchQuery = "matrix";
  await view.runSearch();
  assert.ok(view.listEl.hasClass("rn-layout-pills"));
  assert.equal(view.subtitleEl.textContent, "Search: matrix");
  assert.equal(cards(view).length, 4);
  assert.ok(cards(view)[1].one("rn-pill-linked"));
});

test("link and recent cards follow the layout", () => {
  const folderLines = (view) => cards(view).map((c) => c.one("rn-path")?.textContent ?? "");
  const titleHovers = (view) => cards(view).map((c) => c.one("rn-title").getAttr("aria-label"));

  for (const [mode, activeFile] of [["links", active], ["recent", null]]) {
    const rows = makeView({}, { activeFile });
    rows.view.linksMode = mode === "links";
    rows.view.render();
    assert.equal(cards(rows.view).length, 2, mode);
    assert.deepEqual(folderLines(rows.view), ["", ""], `${mode}: no folder line in rows`);
    assert.deepEqual(titleHovers(rows.view), ["Concepts", null], `${mode}: the folder is on the title hover`);
    assert.equal(rows.view.listEl.one("rn-conn"), null);

    const boxed = makeView({ cardLayout: "cards" }, { activeFile });
    boxed.view.linksMode = mode === "links";
    boxed.view.render();
    assert.deepEqual(folderLines(boxed.view), ["Concepts", ""], `${mode}: the folder line in cards`);
  }

  const classic = makeView({ cardLayout: "classic" });
  classic.view.linksMode = true;
  classic.view.render();
  assert.equal(classic.view.listEl.all("rn-conn-linked").length, 2);
});

test("the note count sits next to the subtitle; the status line shows only work in progress", () => {
  const { view } = makeView();
  view.renderStatus({ status: "ready", done: 147, total: 147, skipped: 0 });
  assert.equal(view.countEl.textContent, "147 notes");
  assert.equal(view.countEl.getAttr("aria-label"), null);
  assert.equal(view.statusEl.textContent, "");
  assert.equal(view.subtitleEl.textContent, "Based on Eigenvektoren", "ready re-renders the list");

  view.renderStatus({ status: "ready", done: 147, total: 149, skipped: 2 });
  assert.equal(view.countEl.textContent, "147 notes · 2 skipped");
  assert.match(view.countEl.getAttr("aria-label"), /developer console/);

  view.renderStatus({ status: "building", done: 50, total: 100, skipped: 2 });
  assert.equal(view.countEl.textContent, "");
  assert.equal(view.countEl.getAttr("aria-label"), null);
  assert.equal(view.statusEl.textContent, "Indexing… 50% (50/100) · 2 skipped");
  assert.match(view.statusEl.getAttr("aria-label"), /developer console/);

  view.renderStatus({ status: "error", done: 0, total: 0, skipped: 0, message: "boom" });
  assert.equal(view.countEl.textContent, "");
  assert.equal(view.statusEl.textContent, "Indexing failed: boom");
  assert.ok(view.statusEl.hasClass("rn-status-error"));
});
