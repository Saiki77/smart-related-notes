# Smart Related Notes 4.0 scope: the Reader release

> ## STATUS — spike (step 0) run 2026-08-29, `lab/results/REPORT-reader-spike.md`
>
> | gate | outcome |
> | --- | --- |
> | native engine in Node (load, Metal, generate, dispose-returns-memory) | **PASS** — 15 MB self-contained package proven, staged at `~/.cache/srn-lab/reader-pkg` |
> | in-Obsidian load test | **staged, awaiting user** (`lab/OBSIDIAN-READER-TEST.md`, 30 s console paste) |
> | judge (answers vs restates vs neither) | **8B: 6/7 found, 0 trap leaks, 1/44 false; 4B-2507: 6/7, 0 leaks, 2/44; 1.7B: degenerate** — AQ blocker broken at 4B+ |
> | tags closed-set | passes every rung: 0.94 (8B) / 0.88 (4B) / 0.81 (1.7B) top-1, zero invented |
> | perf per gist, Metal | 2.6 s (8B, peak 6.1 GB) / 1.7 s (4B, 3.6 GB) / 0.8 s (1.7B, 2.0 GB); CPU-only 4B 3.6 s — Windows-viable |
> | blind gist panel (3 judges, 30 notes x 3 rungs) | **4B-2507 9.26 ties 8B 9.23 (of 10); 8B edges head-to-head wins 10 vs 7; 1.7B 8.32** |
> | ladder decision | 8B default where it fits (>= ~24 GB), **4B-2507 mid + CPU rung (the spike's surprise)**, 1.7B small-RAM rung with judge tasks disabled |
> | next | user pastes the Obsidian console test; then step 1 (engine build) + AQ judge replay on real mined sparks |

Written against the repo at **3.1.0** (2026-08-29). Nothing below is built.
Every feature follows the house rule: **measured in the lab before it ships**,
against the real `IndexStore` where ranking is touched (`bench/v3-integration.mjs`
pattern, assertions proven by ablation).

## 0. What this release is

A local **reader model** joins the plugin: a text model that *reads* notes in
the background and writes small artifacts: a chat-title-style one-liner, a
one-to-two sentence gist, a topic list, tag votes, and short verdicts about
pairs of notes.

It runs on a **two-engine design**, because capability and run-anywhere
pull in opposite directions:

- **Native engine (primary): Qwen3-8B, fully inside the plugin.** llama.cpp
  through its Node bindings (`node-llama-cpp`), loaded in-process: Obsidian
  desktop plugins have Node access, and the bindings are N-API, which is
  ABI-stable across Electron versions, with prebuilt binaries per platform
  (Metal on macOS, Vulkan/CPU on Windows). No external app, no server, no
  ports: enable the reader, the engine + GGUF weights download like any
  other model, done. An 8B is a different species from a 0.3B: reliable
  judging, clean structured output, real receipts. Ollama was considered
  and rejected as a dependency; note that Ollama is itself a wrapper
  around llama.cpp, so this IS the same engine, minus the app.
- **Web engine (safety net): a 0.2-0.6B ONNX model** on the existing
  transformers.js v4 engine. Exists for machines where the native module
  cannot load or 8 GB-class RAM is not there, and as the only permissible
  path if the plugin ever enters the community store (store policy forbids
  downloading executable code at runtime; the plugin ships via release
  zip/BRAT today, so this is a future concern, noted in §7).

Why 8B cannot run on the web engine: onnxruntime-web's WASM heap is
hard-capped at 4 GiB and 8B weights are ~4.6 GB at q4 before KV cache; and
ort-web's WebGPU backend is the component that crashed Obsidian twice in
this plugin's history. Anything larger than ~1B needs the native route.
A bonus of the native engine: it runs ANY size GGUF, so the quality ladder
(8B / 4B / 1.7B) is one engine with a RAM-probed default, not different
stacks.

Naming rule: the word "LLM" appears nowhere in UI or docs. Terminology is
"reader model"; the engine picker reads "Full (recommended, ~5 GB
download, needs ~6 GB free memory)" vs "Lite (small, runs anywhere)".
Opt-in, off by default; weights download through the existing model
manager (extended to list GGUF files).

**Design law (user directive, 2026-08-29): the reader works the way the
embeddings do: invisibly, in the background.** Its jobs are grouping,
selecting, verifying and gating what the other channels nominate. It never
fronts its own prose; the only generated text a user ever sees is the
requested one-liners/gists, short because/receipt lines, and small labels
on groups. Any feature whose point is to SHOWCASE generation (session
recaps, note naming, question resurfacing, anything chat-shaped) is out on
principle, regardless of feasibility.

Corollary on delivery (2026-08-29): **Surprising Connections was REMOVED
from the plugin** (code, report section, README/ARCHITECTURE) at the user's
call: "it just generates a weird markdown file nobody uses." The graph
fusion it was built on stays where it always earned its keep, invisibly
inside `rank()`. The lesson generalizes: reader features deliver IN PLACE
(panel pills, map, editor, on-demand briefs), not as generated report
files. The Vault Insights report survives for its original hygiene lists,
but nothing new targets it as a primary surface.

**The embeddings stay the ranker.** The reader never sits in the hot path and
never replaces a similarity score wholesale. Its role:

- **Annotator**: produce artifacts (gists, topics, tags, reasons) that are
  cached in the index manifest and rendered as progressive enhancement.
- **Judge**: verify or refute candidates the embedding/graph channels nominate
  (top-K pairs, tag suggestions, question-answer candidates). Nomination is
  cheap and broad (embeddings); verification is expensive and narrow (reader).

This division is what the lab history demands: every discovery feature that
failed, failed on *precision* ("obvious", "paraphrase leak", "register match"),
which is exactly the judgment cosine cannot make and a reader can.

## 1. Model

### 1.1 The ladder

**Native engine (primary): Qwen3-8B GGUF q4_k_m** (Apache-2.0, ~4.7 GB, 32k
ctx, 119 languages incl. DE, weights from Hugging Face) through
`node-llama-cpp`. Thinking mode OFF for reader tasks (`/no_think` chat
template flag); these are one-shot extraction calls and thinking would
triple latency for nothing. Expected on the dev Mac (Apple Silicon, 48 GB,
Metal): roughly 30-80 tok/s decode, so a gist is ~1-3 s and a full first
pass over ~650 notes is an idle half-hour. Same engine, smaller rungs for
smaller machines: Qwen3-4B (~2.4 GB, 16 GB-RAM machines) and Qwen3-1.7B
(~1.1 GB, 8 GB), default rung picked by a RAM probe (`os.totalmem`), user
can override. Engine binaries: N-API prebuilds fetched at first enable from
the package's official registry artifacts, checksum-pinned, cached in the
plugin data dir beside the weights (~10-40 MB per platform).

**Web engine (safety net), transformers.js-ready ONNX, chosen by the spike:**

| model | params | ctx | languages | why |
| --- | --- | --- | --- | --- |
| `onnx-community/LFM2.5-350M-ONNX` | 354M | 32k | EN AR ZH FR **DE** JA KO ES | CPU-first design, ~2x faster decode than Qwen3 on CPU; default candidate |
| `LiquidAI/LFM2.5-230M-ONNX` | 230M | 32k | same family | smallest usable; best speed if quality holds |
| `onnx-community/Qwen3-0.6B-ONNX` | 0.6B | 32k | 119 incl. DE | quality ceiling of the built-in tier; slowest; q4f16 |

Licenses: LFM Open License (fine for this MIT plugin's runtime-download model,
same pattern as jina CC-BY-NC); Qwen3 Apache-2.0. Built-in download
~0.2-0.5 GB at q4 (exact sizes recorded during the spike).

Both engines run the SAME task registry, budgets, validators and caches
(§2.2); an engine only swaps the `generate()` backend and the per-task
quality bars. Artifacts are tagged with the model that wrote them, so
switching engine or rung re-reads notes the same way switching embedding
models re-embeds.

Sidebar candidate, not in scope but worth one lab afternoon later:
`onnx-community/Qwen3-Reranker-0.6B-ONNX`, a cross-encoder that scores a pair
in ONE forward pass (no generation). If the generative pair-judge proves too
slow, a reranker is the cheap fallback for pair scoring (but it cannot write
reasons, tags, or gists, and it would wrongly score a paraphrase as a great
"answer", so it does not replace the reader for AQ).

### 1.2 Selection is a measurement, not a vibe (spike, session 1)

New harnesses, run against BOTH engines exactly as they will ship (the
v3-crowding lesson: measure the real code path, not a proxy): Qwen3-8B via
`node-llama-cpp` itself in Node, and Node ORT for the web-engine candidates.
Plus the single most important de-risk, in the 2.0 engine-only-beta mold:
**a minimal load test inside the user's actual Obsidian** that requires the
downloaded `.node` binding, loads the 1.7B rung, and generates one sentence.
If that works, everything else is engineering; if it fails, plan B is a
bundled llama.cpp server spawned via `child_process` bound to 127.0.0.1
(still fully contained in the plugin), and plan C is web-engine-only.

1. `bench/reader-bench.mjs`: per-note wall-clock and tok/s for a realistic
   gist prompt on each engine (8B/4B/1.7B GGUF on Metal and CPU-only mode;
   web smalls at q4, 1 and `cores-2` threads), RSS before/during/after with
   idle-unload verified (load model, run, dispose, confirm memory returns).
2. `bench/reader-quality.mjs`: gists for 30 DE + 30 EN real notes; blind
   3-judge panel (existing `lab-eval-methodology` protocol): faithful? fluent?
   language-matched?
3. Tag pick accuracy: hide the frontmatter tags of the ~116 tagged real notes,
   offer the discriminative-tag candidate set, measure agreement + judged
   precision of extras.
4. Pair judge replay: (a) `aq-eval.mjs` paraphrase traps: does a judge gate
   drive `paraphrase_still_leaks` 0 while keeping recall? (b) continuity
   judged-pairs table: does the judge's strong/weak/none agree with the human
   panels (labels are pair-level and model-independent, so they transfer)?

**Kill criteria** (decided before running, per pre-registration habit):
- **The Obsidian load test is the gate for the whole native engine.** Fail
  -> plan B (bundled server process), fail that -> the release re-scopes
  around the web engine. Decided in session 0, not discovered in session 3.
- **Native 8B** is expected to clear every quality bar; what it must prove
  is judged usefulness of because-lines and the AQ paraphrase gate (~0
  leaks). If 8B itself cannot judge answer-vs-paraphrase, AQ stays dead and
  no smaller model needs testing for it. The 4B/1.7B rungs inherit the same
  evals so each rung gets its own per-task pass/fail row.
- **Web engine**: throughput worse than ~2 notes/min end-to-end at balanced
  threads demotes it to overnight-batch-only; gist quality judged unusable
  on the 350M steps up to Qwen3-0.6B; if that also fails, the web engine
  ships one-liners/gists only and everything judge-shaped requires the
  native engine.
- Tag precision below ~0.8 at top-1 on a given engine/rung: that
  configuration renders tags report-only (Vault Insights), not inline chips.
- Per-task gating is the release valve: a task that fails its bar on one
  rung but passes on a bigger one ships gated, with the settings row saying
  so plainly.

## 2. Engine architecture

### 2.1 Placement

- One interface, two backends: `ReaderBackend.generate(prompt, {maxTokens,
  stop}) -> string`. Everything above it (registry, scheduler, caches,
  validators, UI) is backend-blind.
- **Native backend** (`src/reader-native.ts`): `node-llama-cpp` loaded
  in-process. Model lifecycle is explicit: load on first task, keep warm
  while the queue drains, dispose after N idle minutes (configurable, and
  RSS-verified in the spike; the WebGPU trauma makes "memory comes back
  when idle" a hard requirement, only now it is OUR dispose call, not a
  black box). Inference runs on llama.cpp's own C++ threads, so the
  renderer never blocks; calls are plain async. First-enable flow:
  download engine binary for `process.platform`/`process.arch` +
  chosen GGUF rung, both checksum-pinned, both resumable, both listed in
  the model manager. Everything stays on disk in the plugin data dir; no
  network after download; nothing ever leaves the machine.
- **Web backend**: new `src/worker/reader-worker.ts` beside
  `embed-worker.ts`, sharing the protocol style but its OWN ORT session and
  heap. The embedding init cascade (`configureEnv`, `ort-shim.ts`,
  `gen-ort.mjs`, wasmPaths) is untouched; that saga is settled and stays
  settled. Multi-threaded WASM default; WebGPU only by explicit pin, same
  policy and warning as embeddings. Weights live in the same
  `transformers-cache`; the model manager lists and removes them like any
  other model.

### 2.1b Mac and Windows

The web engine is WASM, identical on both platforms. The native engine is
where platform work lives, and it is a known matrix, not an open problem;
llama.cpp prebuilds exist for every cell:

| platform | binary | acceleration |
| --- | --- | --- |
| macOS arm64 | mac-arm64 | Metal (automatic) |
| macOS x64 / Rosetta | mac-x64 | CPU (detect `process.arch`, fetch matching) |
| Windows x64 | win-x64 | Vulkan (covers NVIDIA + AMD + Intel without CUDA's ~300 MB payload), CPU fallback |
| Linux x64 | linux-x64 | Vulkan / CPU (Obsidian runs there; cheap to include) |

Consequences we own:

- In-process means no port, no server, no Windows Firewall prompt, and the
  engine dies with the plugin; there is no orphan-process lifecycle at all
  (only plan B, the spawned server, would buy that problem).
- Budgets adapt to measured speed, not assumed hardware: the scheduler
  times its own calls and sets notes-per-minute from reality, so a GPU-less
  Windows laptop on the 1.7B rung just reads more slowly instead of
  anything freezing.
- The RAM probe picks the default rung per machine (8B >= 24 GB, 4B >= 16,
  else 1.7B); Windows machines are where the smaller rungs earn their keep.
- Release gate: one full pass on a Windows machine or VM (engine download,
  load, read, chips, map) before 4.0 stable; a Windows beta tester covers
  the Vulkan path better than a VM's CPU-only path.

### 2.2 Task registry ("skills for the reader")

Every reader use is a declared task, not an ad-hoc prompt:

```
task = {
  id,                    // "gist" | "tags" | "judge-pair" | "name-cluster" | "expand-query"
  buildContext(store, …) // deterministic, budgeted; pulls ONLY from the index
                         //   (chunkTexts, idea units, titles, tags, graph facts)
  template,              // instruction + few-shot, language-matched DE/EN
  maxNewTokens,          // hard cap per task (gist ≤ 96, verdict ≤ 48, label ≤ 16)
  validate(raw),         // line-format regex; closed-set checks; null on failure
  cacheKey, invalidate   // srcHash of the same inputs that trigger re-embed
}
```

Context management rules (the whole point of making a tiny model workable):

1. **One call, one task, full reset.** No conversation, no shared KV state
   across tasks; every call starts from a fresh context. The reader has no
   memory except the persisted artifacts.
2. **Nothing raw.** Context builders assemble from what the index already
   distilled: title, tags/aliases, heading breadcrumbs, first window per idea
   unit. Input budget ~1,200-1,600 tokens; long notes never enter whole.
3. **Map-reduce for long notes.** Above a per-engine threshold (~4k chars
   web, ~12k native; 8B reads most notes in one call): idea-level
   micro-gists first (each its own reset call), then one reduce call over
   the micro-gists. Idea units (v1.12) are already the right granularity;
   no new segmentation.
4. **Summaries feed summaries.** Pair judging, tag picking, cluster naming all
   read GISTS, never bodies. After the one-time gist pass, every downstream
   task is cheap and bounded.
5. **Closed-set everything.** Tags must be quoted from the candidate list,
   verdicts from an enum, labels length-capped. Validator failure = retry once
   with shrunk context, then drop silently. Broken output never renders.

### 2.3 Scheduler and persistence

- Idle queue in the `drainLabels` mold: paused while indexing or embedding
  (`labelsAllowed()` pattern), yields via MessageChannel, resumable, priority
  = active note, then recents, then rest.
- Effort setting mirrors `indexSpeed` (light/balanced/fast -> notes per idle
  minute); plus a global kill switch and a status row ("312/635 notes read").
- Manifest gains `reader: {model, srcHash, gist, topics[], tagVotes[]}` per
  entry plus a small LRU (~2k) of pair verdicts keyed by hash pairs. Additive:
  no INDEX_VERSION bump, no re-embed; wiped when the reader model changes.

## 3. Release features (five, no more)

1. **One-liners + gists.** Two artifacts per note, in the note's own
   language. The **one-liner** works like a chat title in the Claude apps: a
   ≤ 10 word distillation, written automatically after the note settles and
   refreshed when the content meaningfully changes (same srcHash trigger as
   re-embedding, so an idle edit does not churn it). It replaces the
   KeyBERT-style keyphrase label as the line under the title on panel
   cards, search results, recents, and map hover; the existing lazy label
   queue (`scheduleLabel`/`drainLabels`) is the delivery mechanism, just
   with better content, and the keyphrase path stays as the instant
   fallback until the reader catches up. The **gist** (1-2 sentences) lives
   behind hover/expand and in Vault Insights, and is what all downstream
   tasks read. Opt-in extra: write the one-liner to a frontmatter property
   (default `summary`) so Bases/Dataview columns and hover previews can use
   it; off by default because it edits the user's files, guarded to only
   ever touch its own property.
2. **Tag suggestions v2** (the "highlight tags that likely fit" ask).
   Propagation nominates (v1.17 mechanism + area gates), reader verifies via
   closed-set pick from gist + candidates. UI: ghost chips in the panel
   ("Suggested: #goa/character +"), one click writes frontmatter, dismiss is
   remembered. Precision bar from the spike decides chips vs report-only.
3. **"Because" lines + bounded re-rank** (the "better rate how notes relate"
   ask). For the visible top-K only (K ≤ 10), judge pairs from gists + topics
   + structural facts; render one reason sentence (≤ 15 words) under the card
   label. A "none" verdict may demote a card below the fold; verdicts NEVER
   promote anything the embeddings didn't shortlist. Cached; trickles in as
   computed. Integration-tested through the real `rank()` with an ablation
   flag, per the bench rule.
4. **Cluster names on the vault map.** c-TF-IDF terms + member titles in,
   2-4 word label out; replaces the raw term labels only when validation
   passes. Directly fixes the known DE/EN duplicate-label wart.
5. **Bilingual query expansion in search.** On explicit search only: one call
   expands the query with ≤ 4 cross-language terms (DE query gains EN terms
   and vice versa), union-only like the concept-search stage (adds recall,
   never demotes). Cached per query. Attacks the measured cross-vocab
   weakness (qa5/7/8/9/10 degraded with scale).

## 4. Measured-before-shipped experiments (lab first, ship only on a win)

- **Gist vector channel.** Embed each gist with the embedding model; a
  denoised whole-note vector. Honest expectation: HIGH rank-correlation with
  the content channel, so per the channel-independence rule
  (`bench/v4-channels.mjs`) it probably fails admission. Measure correlation
  FIRST, recall second; the possible win is on long/template-heavy notes.
- **Answered Questions revival: the 4.0 magic candidate.** AQ died on exactly
  one blocker: cosine cannot tell an answer from a restatement. That is a
  short-text classification the reader can attempt: "Does B answer A, or
  restate it?" over the already-gated candidates. The harness and planted
  traps exist (`aq-eval.mjs`, `paraphrase_still_leaks`). If the judge takes
  the metric to ~0 with recall intact on real sparks, AQ ships in Vault
  Insights and becomes the headline. This experiment is part of the spike
  (§1.2.4a), promotion decided on numbers.
- **Reranker second engine.** Only if generative pair-judging is too slow
  (§1.1 sidebar).

## 4b. Reader-enabled backlog (unscoped, so the release stays at five)

All candidates obey the design law in §0: verification and group building,
in the background. Each promotes only through the usual lab gate:

1. **Cluster referee (group building).** Verify k-means membership on the
   vault map with abstention: "does this note belong with this group, or
   with none?" (the co-association-with-abstention lever from RESEARCH v3,
   now affordable as judgment instead of ensembles). Cleaner map silently;
   and verified membership is the prerequisite that makes Emergent MOCs
   (managed member lists) shippable as real group building.
2. **Dupe-vs-sibling verdicts.** The dupe alarm was rejected because
   sibling notes score 0.86, indistinguishable from paraphrases by any
   threshold. "Duplicate / sibling / unrelated" is exactly a reader-sized
   distinction. Delivery in place: a quiet "possible duplicate" pill on the
   panel card of the twin when you are IN one of them, not a report row.
3. **Continuity receipts for the book.** 8B judge over the G1-G7 continuity
   candidates (0.60 precision offline): same fact / restated /
   CONTRADICTION, with receipts. Delivery in place: a pill on the affected
   chapter's card. The emergent-lab-finding feature for a novelist, and
   pure verification.
4. **Property autofill for Bases.** Closed-set classification into the
   user's own property vocabulary; suggested, never auto-written.
5. **Split & merge briefs.** Atom-Advisor-lite (no raw-file offsets
   needed): a "3 ideas live here" hint on the active note, brief on click;
   merge briefs behind the duplicate pill above. Group building at the
   note level, delivered where the note is, not in a report.

One survivor from a cut item, kept ONLY because it is display-free: silent
DE<->EN alias/inflection enrichment of the glow's mention vocabulary, so
German plural mentions start glowing (today's variant matching is
English-regex-only). Nothing is ever shown or generated for the user; say
the word and this goes too.

Cut on principle, not feasibility: "where was I" cards, open-loop
resurfacing, note naming: obvious generation-showcase applications.
Also parked: search answer card (chat-shaped), entity dossiers.

## 5. Spatial track (parallel, does not depend on the reader)

Motivating post (r/ObsidianMD "almost completely abandoned Obsidian",
2026-08) was unreachable from this environment (Reddit blocked); designed
from the stated ask: let the user relate ideas SPATIALLY. The recurring story
in that genre: people leave for whiteboard-first tools (Heptabase/Scrintal)
because arranging cards spatially is how they think, and core Canvas feels
disconnected from their notes. Verify against the actual post before build.

Decision: **integrate with native Canvas, do not build a spatial editor.**
`.canvas` files are plain JSON, interop for free, zero editor maintenance.

1. **Seed a canvas.** Commands: "Arrange related notes on a canvas" (active
   note centered, top-K placed by similarity, closer = more related),
   "Canvas from cluster/MOC/search". User then drags freely; the layout is
   theirs.
2. **Canvas as a signal.** Opt-in: co-presence on a user's canvas, explicit
   canvas edges, and drag-proximity groups become weak graph-channel edges
   (weight below real wikilinks). This is user-authored relatedness: exactly
   the behavioural channel the v4-channels analysis named as the one
   remaining independent signal, and it makes arranging cards FEED the
   ranker, which no other plugin does. Admission by the same independence +
   recall measurement, on real canvases dumped via the console like notes.
3. **Map pinning (later).** Persist user position overrides on the vault map
   across recomputes (Procrustes alignment); drag-to-pin turns the map from
   a picture into a workspace.

## 6. Sequencing

| step | content | est. |
| --- | --- | --- |
| 0 | spike: node-llama-cpp bench (8B/4B/1.7B) + in-Obsidian load test + judge evals + AQ replay; fix per-task gates | 1 session |
| 1 | engine: backend interface + native backend (binary/GGUF download, lifecycle, RAM probe) + reader worker (web) + task registry, scheduler, persistence, model manager | 2 sessions |
| 2 | one-liners/gists + tag chips (+ frontmatter property opt-in) | 1 session |
| 3 | because-lines + cluster names + query expansion | 1-2 sessions |
| 4 | spatial: canvas seed + canvas signal (measured) | 1-2 sessions |
| 5 | AQ ship if the spike says yes | 1 session |
| 6 | Windows pass (VM or beta tester) before stable | 0.5 session |

Rollout like the 2.0 jina engine jump: `4.0.0-beta.N` pre-releases (new model
subsystem = major), engine validated live in the user's Obsidian before
features stack on top. Betas install from the release zip as usual.

## 7. Risks (ranked)

1. **Native module load in Obsidian.** The one genuinely novel bet: a
   runtime-downloaded N-API binding must `require()` cleanly inside
   Obsidian's renderer on macOS AND Windows. N-API's ABI stability and
   node-llama-cpp's documented Electron support say yes; the spike's
   in-Obsidian load test proves it before anything is built on top, and
   plan B (bundled llama-server via `child_process`) plus plan C (web
   engine) are pre-committed fallbacks, not improvisation. Community-store
   note: runtime-downloaded executables would violate store policy; the
   plugin distributes via release zip/BRAT, and a store submission would
   ship binaries inside the zip instead (decision deferred until relevant).
2. **Built-in tier limits.** WASM throughput unknown until the spike, and a
   second ORT session costs ~0.5-1 GB (the jina 4 GiB-heap lesson is
   designed around via hard input caps). Everything is sized so slow is
   survivable (background, cached, resumable, per-task tier gates); verify
   RSS across a full pass in the spike.
3. **Small-model output quality.** Contained by closed sets, validators,
   drop-on-fail, language matching, and tier gating; nothing unvalidated
   ever renders on either tier.
4. **Rank interference.** Reader influence on ordering is demote-only within
   the embedding top-K, integration-tested with ablation; embeddings remain
   the ranker by construction.
5. **UI noise.** Everything reader-made is progressive enhancement with one
   kill switch; if a surface annoys, it turns off without touching ranking.
