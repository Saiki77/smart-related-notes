// Exercise the real engine and index store with deterministic worker responses.
import assert from "node:assert/strict";
import { after, afterEach, beforeEach, test } from "node:test";
import { MessageChannel } from "node:worker_threads";
import { EmbeddingEngine, setOrtAssetLoader } from "../src/embeddings.ts";
import { IndexStore } from "../src/index-store.ts";
import { TFile } from "./obsidian-stub.mjs";

const originals = { Worker: globalThis.Worker, window: globalThis.window, MessageChannel: globalThis.MessageChannel };
const channels = [];
globalThis.window = globalThis;
globalThis.MessageChannel = class extends MessageChannel {
  constructor() { super(); channels.push(this); }
};

class FakeWorker {
  static instances = [];
  static onEmbed = (worker, request) => worker.succeed(request);
  terminated = false;
  poisoned = false;
  requests = [];
  constructor() { FakeWorker.instances.push(this); }
  postMessage(request) {
    if (request.type === "embed") this.requests.push(request);
    queueMicrotask(() => {
      if (this.terminated) return;
      if (request.type === "init") this.reply({ id: request.id, type: "ready", device: "wasm" });
      else if (this.poisoned) this.fail(request, "std::bad_alloc");
      else FakeWorker.onEmbed(this, request);
    });
  }
  reply(message) { this.onmessage({ data: message }); }
  fail(request, message = "std::bad_alloc") {
    this.reply({ id: request.id, type: "error", message });
  }
  succeed(request) {
    this.reply({ id: request.id, type: "result", dims: 2, count: request.texts.length,
      data: new Float32Array(request.texts.flatMap(() => [0.6, 0.8])) });
  }
  terminate() { this.terminated = true; }
}
globalThis.Worker = FakeWorker;
setOrtAssetLoader(async () => ({ glueText: "", wasmBinary: new ArrayBuffer(0) }));

const engines = [];
function makeEngine() {
  const engine = new EmbeddingEngine("jinaai/jina-embeddings-v5-text-nano-text-matching", "wasm");
  engines.push(engine);
  return engine;
}
beforeEach(() => {
  FakeWorker.instances = [];
  FakeWorker.onEmbed = (worker, request) => worker.succeed(request);
});
afterEach(async () => {
  await Promise.all(engines.splice(0).map((engine) => engine.dispose()));
});
after(() => {
  for (const channel of channels) { channel.port1.close(); channel.port2.close(); }
  for (const [key, value] of Object.entries(originals)) {
    if (value === undefined) delete globalThis[key];
    else globalThis[key] = value;
  }
});

const options = {
  excludeFolders: [], topK: 12, minSimilarity: 0.2, chunking: true,
  structureInfluence: 0.15, maxChunks: 48, shortlistSize: 60, showSummary: false,
  headingContext: false, ideaInfluence: 0.3, isolatedAreas: [], graphInfluence: 1,
};
function makeStore(engine, notes = []) {
  const files = notes.map(([path]) => new TFile(path, 1));
  const bodies = new Map(notes);
  const app = {
    vault: {
      getMarkdownFiles: () => files,
      getAbstractFileByPath: (path) => files.find((file) => file.path === path) ?? null,
      cachedRead: async (file) => bodies.get(file.path),
      adapter: { exists: async () => true, write: async () => {}, writeBinary: async () => {},
        rename: async () => {}, remove: async () => {}, mkdir: async () => {} },
    },
    metadataCache: { getFileCache: () => null, getCache: () => null, resolvedLinks: {} },
  };
  return new IndexStore(app, engine, "test-index", options);
}

test("whole-note budgets account for CJK after an English introduction", () => {
  const store = makeStore(makeEngine());
  const length = (body) => store.chunkNoteWholeNote(new TFile("note.md"), body, "")[0].embedText.length;
  assert.equal(length("a".repeat(12000)), 10000);
  assert.ok(length("中".repeat(12000)) <= 4010);
  assert.ok(length("a".repeat(2048) + "中".repeat(10000)) < 6000);
  assert.equal(length("a".repeat(10000) + "中".repeat(10000)), 10000);
});

test("trailing English cannot increase the allowance for a CJK-heavy prefix", () => {
  const store = makeStore(makeEngine());
  const embed = (body) => store.chunkNoteWholeNote(new TFile("note.md"), body, "")[0].embedText;
  const prefix = "中".repeat(6250);
  const text = embed(prefix + "a".repeat(3750));
  assert.equal(text, embed(prefix));
  assert.ok(text.length <= 4010);
  assert.ok([...text].filter((c) => c === "中").length <= 4000);
  assert.equal(text.includes("a"), false);
});

test("alternating scripts consume the budget of the retained prefix", () => {
  const store = makeStore(makeEngine());
  const text = store.chunkNoteWholeNote(new TFile("note.md"), "中a".repeat(6000), "")[0].embedText;
  // 'note. ' costs 6; 2,855 pairs cost 9,992.5. The next CJK character exceeds the budget.
  assert.equal(text, "note. " + "中a".repeat(2855));
});

test("pure Japanese kana and Korean syllables use the CJK budget too", () => {
  const store = makeStore(makeEngine());
  for (const char of ["あ", "한"]) {
    const text = store.chunkNoteWholeNote(new TFile("note.md"), char.repeat(12000), "")[0].embedText;
    assert.ok(text.length <= 4010);
    assert.ok([...text].filter((c) => c === char).length <= 4000);
  }
});

test("a capacity error is propagated without retrying the same input", async () => {
  const engine = makeEngine();
  FakeWorker.onEmbed = (worker, request) => {
    if (request.texts.includes("oversized")) { worker.poisoned = true; worker.fail(request); }
    else worker.succeed(request);
  };
  await assert.rejects(engine.embed("oversized"), /bad_alloc/);
  const failed = FakeWorker.instances[0];
  assert.equal(failed.requests.length, 1);
  assert.equal(failed.terminated, true);
  assert.equal(engine.loaded, false);
  assert.equal((await engine.embed("small")).length, 2);
  assert.equal(FakeWorker.instances.length, 2);
});

test("unrelated errors preserve the worker and do not trigger smaller tiers", async () => {
  const engine = makeEngine();
  const store = makeStore(engine, [["note.md", "ordinary text"]]);
  FakeWorker.onEmbed = (worker, request) => worker.fail(request, "backend unavailable");
  assert.equal(await store.embedFile(new TFile("note.md")), null);
  assert.equal(FakeWorker.instances.length, 1);
  assert.equal(FakeWorker.instances[0].requests.length, 1);
  assert.equal(FakeWorker.instances[0].terminated, false);
});

test("a delayed capacity error from an old session cannot terminate its replacement", async () => {
  const engine = makeEngine();
  await engine.init();
  const session = engine.session;
  const embed = session.embed.bind(session);
  let rejectDelayed;
  const delayed = new Promise((_, reject) => { rejectDelayed = reject; });
  // Hold one rejection until the other request has already replaced the session.
  session.embed = (texts, kind) => texts.includes("second") ? delayed : embed(texts, kind);
  FakeWorker.onEmbed = (worker, request) => worker.fail(request);
  const first = engine.embed("first");
  const second = engine.embed("second");
  const secondFailure = assert.rejects(second, /bad_alloc/);
  await assert.rejects(first, /bad_alloc/);
  FakeWorker.onEmbed = (worker, request) => worker.succeed(request);
  await engine.embed("replacement");
  rejectDelayed(new Error("std::bad_alloc"));
  await secondFailure;
  assert.equal((await engine.embed("next")).length, 2);
  assert.equal(FakeWorker.instances.length, 2);
  assert.equal(FakeWorker.instances[1].terminated, false);
});

test("an in-flight request interrupted by cleanup retries on a fresh worker", async () => {
  const engine = makeEngine();
  const pending = [];
  FakeWorker.onEmbed = (worker, request) => pending.push({ worker, request });
  const oversized = engine.embed("oversized");
  const normal = engine.embed("normal");
  const failure = assert.rejects(oversized, /bad_alloc/);
  while (pending.length < 2) await new Promise((resolve) => setImmediate(resolve));
  FakeWorker.onEmbed = (worker, request) => worker.succeed(request);
  pending[0].worker.fail(pending[0].request);
  await failure;
  assert.equal((await normal).length, 2);
  assert.equal(FakeWorker.instances.length, 2);
});

test("an oversized note falls back to ideas without skipping later batches", async () => {
  const engine = makeEngine();
  const notes = [["oversized.md", "中".repeat(12000)],
    ...Array.from({ length: 10 }, (_, i) => [`normal-${i}.md`, "Small ordinary note."])];
  const store = makeStore(engine, notes);
  FakeWorker.onEmbed = (worker, request) => {
    if (request.texts.some((text) => text.length > 3500)) {
      worker.poisoned = true; worker.fail(request);
    } else worker.succeed(request);
  };
  await store.build();
  assert.equal(store.getProgress().status, "ready");
  assert.equal(store.count, notes.length);
  assert.equal(store.getProgress().done, notes.length);
  assert.equal(store.getProgress().skipped, 0);
  assert.ok(FakeWorker.instances.length >= 3);
});

test("failure of every splitting tier still leaves later notes a clean worker", async () => {
  const engine = makeEngine();
  const store = makeStore(engine, [["oversized.md", "中".repeat(12000)], ["normal.md", "Small note."]]);
  let failures = 0;
  FakeWorker.onEmbed = (worker, request) => {
    if (request.texts.includes("oversized")) {
      failures++; worker.poisoned = true; worker.fail(request);
    } else worker.succeed(request);
  };
  await store.build();
  assert.equal(failures, 4); // One failed batch, followed by auto / ideas / windows.
  assert.equal(store.getProgress().status, "ready");
  assert.equal(store.count, 1);
  // The status reports what is in the index, not every note looked at.
  assert.equal(store.getProgress().done, 1);
  assert.equal(store.getProgress().total, 2);
  assert.equal(store.getProgress().skipped, 1);
  assert.equal(FakeWorker.instances.length, 5);
  assert.ok(FakeWorker.instances.slice(0, -1).every((worker) => worker.terminated));
  assert.equal(FakeWorker.instances.at(-1).terminated, false);
});

test("the skipped count follows later fixes and deletions", async () => {
  const engine = makeEngine();
  const store = makeStore(engine, [
    ["oversized.md", "中".repeat(12000)],
    ["normal.md", "Small note."],
    ["other.md", "Another small note."],
  ]);
  let failing = true;
  FakeWorker.onEmbed = (worker, request) => {
    if (failing && request.texts.includes("oversized")) {
      worker.poisoned = true; worker.fail(request);
    } else worker.succeed(request);
  };
  await store.build();
  const counts = () => { const p = store.getProgress(); return [p.done, p.total, p.skipped]; };
  assert.deepEqual(counts(), [2, 3, 1]);

  // Fixed by a later edit: it moves from skipped to indexed.
  failing = false;
  await store.updateFile(new TFile("oversized.md", 2));
  assert.deepEqual(counts(), [3, 3, 0]);

  // Failing again while it still has its earlier entry keeps it indexed.
  failing = true;
  await store.updateFile(new TFile("oversized.md", 3));
  assert.deepEqual(counts(), [3, 3, 0]);

  // A deleted note leaves both counts.
  store.removeFile("oversized.md");
  assert.deepEqual(counts(), [2, 2, 0]);
});

test("deleting a skipped note removes it from the count", async () => {
  const engine = makeEngine();
  const store = makeStore(engine, [["oversized.md", "中".repeat(12000)], ["normal.md", "Small note."]]);
  FakeWorker.onEmbed = (worker, request) => {
    if (request.texts.includes("oversized")) { worker.poisoned = true; worker.fail(request); }
    else worker.succeed(request);
  };
  await store.build();
  assert.equal(store.getProgress().skipped, 1);
  store.removeFile("oversized.md");
  const p = store.getProgress();
  assert.deepEqual([p.done, p.total, p.skipped], [1, 1, 0]);
});
