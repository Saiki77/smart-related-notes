// The declarative settings tab, driven through the REAL src/main.ts: every control
// binds to a real setting, every user-facing setting has exactly one control, the
// controls accept the default values, every render row runs, and a change still
// saves and reaches the panel, the title index, the reader and the first-run gate.
//   node --experimental-transform-types --import ./bench/register-stub.mjs --test bench/settings-definitions-check.mjs
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, test } from "node:test";
import { Setting } from "./obsidian-stub.mjs";

// The reader's storage row reads its asset folder: point it at an empty one.
const home = mkdtempSync(join(process.env.TMPDIR || tmpdir(), "srn-settings-"));
process.env.SRN_READER_HOME = home;
after(() => rmSync(home, { recursive: true, force: true }));
const { RelatedNotesSettingTab, DEFAULT_SETTINGS } = await import("../src/main.ts");

// Settings that are state, not choices: they have no control on purpose.
const INTERNAL = new Set(["modelChosen", "centeredScaleMigrated", "pinnedPath", "whatsNewShownId", "suggesterTakeOverUserSet"]);

function makeTab(overrides = {}) {
  const calls = { save: 0, panel: 0, viewRender: 0, reader: 0, markDirty: 0, rebuild: 0 };
  const plugin = {
    settings: { ...DEFAULT_SETTINGS, ...overrides },
    store: { getProgress: () => ({ status: "ready", done: 10, total: 12, skipped: 2 }), count: 10 },
    reader: null,
    titleIndex: { markDirty: () => calls.markDirty++ },
    saveSettings: async () => { calls.save++; },
    rebuildIndex: async () => { calls.rebuild++; },
    requestPanelRender: () => calls.panel++,
    getView: () => ({ requestRender: () => calls.viewRender++ }),
    applyReaderSettings: () => calls.reader++,
  };
  return { tab: new RelatedNotesSettingTab({}, plugin), plugin, calls };
}

function collect(items, out = { controls: [], renders: [], empties: [], pages: [], groups: [] }) {
  for (const it of items) {
    if (it.type === "group" || it.type === "list") {
      out.groups.push(it);
      collect(it.items ?? [], out);
    } else if (it.type === "page") {
      out.pages.push(it);
      collect(it.items ?? [], out);
    } else if (it.control) out.controls.push(it);
    else if (it.render) out.renders.push(it);
    else out.empties.push(it);
  }
  return out;
}
const flush = () => new Promise((r) => setImmediate(r));

test("every control binds to a real setting, and every choice has exactly one control", () => {
  const { tab } = makeTab();
  const { controls } = collect(tab.getSettingDefinitions());
  const keys = controls.map((c) => c.control.key);
  for (const k of keys) assert.ok(k in DEFAULT_SETTINGS, `control key ${k} is not a setting`);
  assert.equal(new Set(keys).size, keys.length, `a setting has two controls: ${keys.filter((k, i) => keys.indexOf(k) !== i)}`);
  const expected = Object.keys(DEFAULT_SETTINGS).filter((k) => !INTERNAL.has(k)).sort();
  assert.deepEqual([...keys].sort(), expected);
});

test("dropdowns offer the default value, and sliders reach it", () => {
  const { tab } = makeTab();
  const { controls } = collect(tab.getSettingDefinitions());
  for (const { name, control } of controls) {
    const value = tab.getControlValue(control.key);
    if (control.type === "dropdown") {
      assert.ok(String(value) in control.options, `${name}: default ${value} is not an option`);
    } else if (control.type === "slider") {
      assert.ok(value >= control.min && value <= control.max, `${name}: default ${value} outside ${control.min}..${control.max}`);
      const steps = (value - control.min) / control.step;
      assert.ok(Math.abs(steps - Math.round(steps)) < 1e-6, `${name}: default ${value} is not on a step`);
    } else if (control.type === "toggle") {
      assert.equal(typeof value, "boolean", `${name}: toggle value is not a boolean`);
    } else if (control.type === "textarea") {
      assert.equal(typeof value, "string", `${name}: text value is not a string`);
    }
  }
});

test("a custom model id and a custom idle timeout stay selectable", () => {
  const { tab } = makeTab({ modelId: "someone/custom-model", idleUnloadMinutes: 7 });
  const { controls } = collect(tab.getSettingDefinitions());
  const byKey = Object.fromEntries(controls.map((c) => [c.control.key, c.control]));
  assert.ok("someone/custom-model" in byKey.modelId.options);
  assert.ok("7" in byKey.idleUnloadMinutes.options);
});

test("every row has a name, every control a description, every page a summary line", () => {
  const { tab } = makeTab();
  const { controls, renders, pages } = collect(tab.getSettingDefinitions());
  for (const row of [...controls, ...renders]) assert.ok(row.name && row.name.length > 0);
  for (const c of controls) assert.ok(typeof c.desc === "string" && c.desc.length > 0, `${c.name} has no description`);
  assert.deepEqual(pages.map((p) => p.name), ["Reader", "Linking", "Scope", "Engine", "Advanced matching"]);
  for (const p of pages) assert.ok(p.desc, `${p.name} has no description`);
});

test("every render row runs on a Setting row", async () => {
  const { tab } = makeTab();
  const { renders } = collect(tab.getSettingDefinitions());
  for (const r of renders) {
    const setting = new Setting();
    assert.doesNotThrow(() => r.render(setting, undefined), `${r.name} threw`);
  }
  await flush();
  const index = renders.find((r) => r.name === "Index");
  const row = new Setting();
  index.render(row, undefined);
  assert.match(row.desc, /10 notes embedded, 2 skipped/);
  assert.ok(row.components.some((c) => c.kind === "button"));
  const storage = renders.find((r) => r.name === "On this device");
  const store = new Setting();
  storage.render(store, undefined);
  assert.equal(store.desc, "Nothing downloaded.");
  assert.equal(store.components.length, 0, "no remove button without downloads");
});

test("predicates and page summaries follow the settings", () => {
  const { tab, plugin } = makeTab({ modelChosen: false, excludeFolders: "Templates\nArchive", isolatedAreas: "goa" });
  const items = tab.getSettingDefinitions();
  const { controls, empties, pages } = collect(items);
  const gate = empties.find((e) => e.name === "Pick a model to start");
  assert.equal(gate.visible(), true);
  plugin.settings.modelChosen = true;
  assert.equal(gate.visible(), false);
  const snippet = controls.find((c) => c.control.key === "showSnippet").control;
  plugin.settings.showSummary = true;
  assert.equal(snippet.disabled(), true);
  plugin.settings.showSummary = false;
  assert.equal(snippet.disabled(), false);
  const page = (n) => pages.find((p) => p.name === n);
  assert.equal(page("Reader").displayValue(), "Off");
  assert.equal(page("Scope").displayValue(), "2 folders left out · 1 isolated area");
  plugin.settings.excludeFolders = "";
  plugin.settings.isolatedAreas = "";
  assert.equal(page("Scope").displayValue(), "Whole vault");
  assert.match(page("Engine").displayValue(), /Auto/);
});

test("a change saves and reaches the panel, the title index and the reader", async () => {
  const { tab, plugin, calls } = makeTab({ pinnedPath: "Notes/A.md" });
  await tab.setControlValue("topK", 20);
  assert.equal(plugin.settings.topK, 20);
  assert.equal(calls.save, 1);

  await tab.setControlValue("showFolder", false);
  assert.equal(calls.panel, 1);

  await tab.setControlValue("idleUnloadMinutes", "5");
  assert.equal(plugin.settings.idleUnloadMinutes, 5);
  assert.equal(tab.getControlValue("idleUnloadMinutes"), "5");

  await tab.setControlValue("suggesterTakeOver", false);
  assert.equal(plugin.settings.suggesterTakeOverUserSet, true);

  await tab.setControlValue("pinPersist", false);
  assert.equal(plugin.settings.pinnedPath, "");
  assert.equal(calls.viewRender, 1);

  await tab.setControlValue("excludeFolders", "Archive");
  assert.equal(calls.markDirty, 1);

  const updatesBefore = tab.updateCalls;
  await tab.setControlValue("readerPace", "fast");
  assert.equal(calls.reader, 1);
  assert.equal(tab.updateCalls, updatesBefore, "a pace change only refreshes the predicates");
  await tab.setControlValue("readerEnabled", true);
  assert.equal(tab.updateCalls, updatesBefore + 1, "enabling the reader rebuilds the tab");
  assert.ok(tab.refreshCalls > 0);
});

test("the first model choice opens the gate and starts the first build", async () => {
  const { tab, plugin, calls } = makeTab({ modelChosen: false });
  await tab.setControlValue("modelId", DEFAULT_SETTINGS.modelId);
  assert.equal(plugin.settings.modelChosen, true);
  assert.equal(tab.updateCalls, 1);
  await flush();
  assert.equal(calls.save, 1);
  assert.equal(calls.rebuild, 1, "same model as before: the first build starts here");

  const changed = makeTab({ modelChosen: false });
  await changed.tab.setControlValue("modelId", "Xenova/multilingual-e5-small");
  await flush();
  assert.equal(changed.calls.rebuild, 0, "a new model: saveSettings() rebuilds by itself");
});

test("a change behind a page summary rebuilds the tab; typing does not", async () => {
  const { tab } = makeTab();
  for (const [key, value] of [["glowEnabled", false], ["device", "wasm"], ["readerRung", "mid"]]) {
    const before = tab.updateCalls;
    await tab.setControlValue(key, value);
    assert.equal(tab.updateCalls, before + 1, `${key} should refresh its page summary`);
  }
  const before = tab.updateCalls;
  await tab.setControlValue("excludeFolders", "Archive");
  assert.equal(tab.updateCalls, before, "typing in a text area must not rebuild the tab");
  tab.hide();
  assert.equal(tab.updateCalls, before + 1, "closing the pane refreshes the summaries");
});

test("drawing the reader status never rebuilds the tab from inside a render", () => {
  const { tab, plugin } = makeTab();
  let error = "network blocked";
  plugin.reader = { engineError: () => error, status: { state: "error", detail: "network blocked" } };
  const { renders } = collect(tab.getSettingDefinitions());
  const status = renders.find((r) => r.name === "Status");
  const row = new Setting();
  status.render(row, undefined);
  assert.equal(tab.updateCalls, 0, "the render itself must not call update()");
  assert.match(row.descEl.children[0].text, /Engine error/);
  tab.updateReaderStatus();
  assert.equal(tab.updateCalls, 0, "same error: redraw in place");
  error = null;
  plugin.reader.status = { state: "ready", read: 1, total: 4 };
  tab.updateReaderStatus();
  assert.equal(tab.updateCalls, 1, "the error cleared: the error row must go");
});
