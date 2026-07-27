# Smart Related Notes 3.0 — scope proposal

> ## STATUS as built (2026-07-27)
>
> | item | state |
> | --- | --- |
> | 0.1 template de-crowding | **done** (crowding 9.9 -> 0.7) |
> | 1.2 graph x content fusion | **done** (recall 0.66 -> 0.75) |
> | 1.4 model decision | **done** (measurement said keep MiniLM default) |
> | 2.1 Surprising Connections | **done** |
> | 2.3 vault map | **done** (purity 0.65 vs 0.29) |
> | 2.4 fused score in the `[[` suggester | **done** |
> | 1.1 CSLS hubness correction | **dropped**, and correctly so: measured +0.005 on MiniLM but -0.030 on jina |
> | **1.3 calibrated scores (Mutual Proximity)** | **NOT BUILT** |
> | **1.5a fold structureInfluence into the fusion combiner** | **NOT BUILT** (still a separate additive boost, never re-tuned) |
> | **2.2 suggested links with receipts** | **NOT BUILT** (moved to 4.0) |
>
> **All three were then measured (2026-07-27) against the rule "only add it if it
> increases performance". Two failed and are now closed:**
>
> - **1.3 Mutual Proximity: REJECTED.** It fails both of its own claims.
>   Recall drops (0.664 -> 0.643 alone; 0.745 -> 0.706 with the graph channel),
>   and on the calibration test it is *worse* than plain cosine: holding a fixed
>   threshold across nested vault sizes, cosine drifts 0.018 and MP drifts 0.030.
>   The one thing it improves is standalone hub-skew (1.66 -> 1.35), which does
>   not convert into ranking quality. `bench/v3-calibration.mjs`.
> - **1.5a fold structureInfluence: NO MEASURABLE PAYOFF, closed.** The boost's
>   dominant term is direct links, and a link-prediction protocol excludes linked
>   notes as candidates by construction, so that term is untestable here. Its
>   other term, shared tags, was tested: `content+graph+tags` scores 0.7453,
>   identical to `content+graph`. Tag overlap correlates 0.916 with the graph
>   channel, so the graph already carries it. Folding would be a refactor with no
>   measured gain. `bench/v4-channels.mjs`.
> - **2.2 receipts: still worth building, and now correctly sized.** Judged
>   precision 0.40 -> 0.60 stands. Two facts discovered while scoping the port:
>   detail extraction can run off the `chunkTexts` the index already keeps (no
>   vault re-read needed), but the **cheap G2 substitution does not work**.
>   Deriving "language-common" from the vault's own word frequencies instead of
>   the bundled lists loses a planted positive (cpB, the rare-technical-term
>   class) and rescaling the thresholds to the vault's 12,678-word vocabulary
>   does not recover it. So receipts needs the ~200 KB CC-BY-SA frequency lists
>   bundled, with attribution. That is a build-time asset decision, which is why
>   it is a 4.0 item rather than something to rush into 3.0.

Written against the repo at **2.1.6** and the measured results in
`lab/results/REPORT-20260702.md`. Every quality number below is measured on the
488-note bilingual lab vault under **jina-v5-nano** unless marked *unmeasured*.

Ordering principle: **engine first, surfaces second.** Tier 1 changes lift the
sidebar, search, the `[[` suggester and Vault Insights *simultaneously*, because
all four read the same ranking. Shipping a new panel on an un-upgraded ranker
buys one feature; upgrading the ranker buys all of them.

---

## 0. Where 2.1.6 actually is

### 0.1 Shipped (the docs understate this)

| Area | Shipped |
| --- | --- |
| Core panel | chunk-level biMax, idea blend, structural boost, mean-centering, keyphrase summary, similarity pill, snippet, recency |
| Search | semantic vault search in the panel header |
| Linked-notes mode | panel toggle showing outgoing links (MOC members) |
| Feature A | inline glow + 1-click link, "link all unlinked mentions", context gate |
| Feature B | smart `[[` suggester ranked semantically + "create new note" rows |
| Vault Insights | suggested links, **suggested tags**, orphans, near-duplicates, stale notes |
| Infra | persisted index, incremental re-embed, int8 chunks + LRU, keyword fallback, isolated areas, exclude folders, model picker, device pref, perf profiles, 6-group settings |

**Docs are stale — fix as part of 3.0.** `README.md` Roadmap still promises "link
recommendations" and a "stale note finder" as *next up*; both shipped inside
Vault Insights. `ARCHITECTURE.md` §5 presents tag-free concept search as the
roadmap centrepiece, but the lab **measured that exact design failing** (§3.1).

### 0.2 The 2.1.x arc was reliability, and it constrains 3.0

Six releases of runtime fragility, not features:

| Rel | What broke / changed |
| --- | --- |
| 2.1.0 | idle RAM → embedding engine moved into a **terminable Web Worker** |
| 2.1.1 | aggressive idle-unload (30 s / 1 min) |
| 2.1.2 | **community-plugin review**: no bare `fetch`, no release zip, self-cached ORT runtime |
| 2.1.3 | jina-v5-nano needed **both ORT builds**, picked per device |
| 2.1.4 | threaded bring-up: patch the ORT glue's Node branch in every realm |
| 2.1.6 | **one oversized note broke indexing**; whole-note models pinned to 1 thread |

Three hard constraints fall out of this, and every Tier 1/2 item below must
respect them:

1. **No runtime fetch, no external assets.** Anything a feature needs must be
   bundled or derived on-device (review requirement from 2.1.2).
2. **Nothing may block or break indexing.** 2.1.6 established the pattern:
   degrade (`auto → ideas → windows`), skip by name, continue. New subsystems
   must be cancellable and must fail soft.
3. **The ORT wasm heap is hard-capped at 4 GiB** (32-bit address space — machine
   RAM is irrelevant) and inference is ~quadratic in tokens. This caps the
   *model* path, not CPU-side vector math — but it is why the whole-note path is
   fragile.

---

## Tier 0 — Correctness

### 0.1 Daily-note / template crowding — measured bug
Every daily note's top-10 is **100% other daily notes**; the shared template
block dominates the note vector. Fix = per-section content-hash dedup at index
time (lab gate **G1**: drop folded lines/sentences with document frequency ≥ 3
before embedding).
*Touches:* `index-store.ts` chunking. *Effort:* S. *Risk:* low — notes that are
only boilerplate fall back to the title chunk.

### 0.2 Empty-wikilink and hub artefacts
`[[]]` survives parsing today; hub/index notes (`* MOC`, `Übersicht`,
`Zettelkasten Index`, the generated `Vault Insights` note) bleed into pair-mining
as bridges. One shared hub/exclusion predicate, used by insights + fusion.
*Effort:* S.

---

## Tier 1 — Engine upgrades (invisible, lift every surface)

All untrained or near-parameter-free, offline, incremental. **All of them are
CPU-side vector math — they never touch the ORT heap**, so they do not inherit
the 2.1.6 fragility. Their budget problem is different: they are *O(N²)-shaped*
and must be approximated (§1.5).

### 1.1 Hubness correction (CSLS / Mutual Proximity) — best value/effort
Subtract each note's mean similarity to its own neighbourhood from both sides:
`csls(a,b) = 2·cos(a,b) − r(a) − r(b)`. Kills the "one note is everyone's
neighbour" failure.
- Measured: link-recall@10 **0.693 → 0.708**; **non-obvious** link recall
  **0.181 → 0.260 (+44%)**; hubness skew **0.57 → 0.20**.
- Mutual Proximity is the calibrated variant — same correction, but the output is
  a [0,1] probability, which fixes the pill and thresholds (§1.3).
- *Touches:* `vector-math.ts`, `index-store.ts` (cache `r(a)`). *Effort:* S–M.

### 1.2 Graph × content fusion — the biggest single jump
Add a **structural channel** (Resource Allocation over shared neighbours +
rooted PageRank) and fuse with content by rank.
- Measured: **0.693 → 0.782 link-recall@10 (+12.8%)**, untrained.
- This is **not** today's `structureInfluence`, which rewards notes that are
  *already linked* (direct link 1.0, shared tags 0.6, biblio 0.5, frontmatter
  0.3). The fusion channel predicts **missing** links from shared neighbourhoods.
  That is why it recovers links cosine cannot see: **30.7% of the user's real
  links have the target outside the source's content top-10.**
- Cold-start safe: a note with no links falls back to pure content.
- *Touches:* new `graph-signals.ts` (adjacency from `metadataCache.resolvedLinks`,
  RA + PPR, incremental), ranking in `index-store.ts`. *Effort:* M.

### 1.3 Calibrated scores
Mutual-Proximity output makes `minSimilarity` mean the same thing in a 200-note
and a 20 000-note vault, and makes the % pill honest.
*Risk:* changes displayed numbers → needs a migration flag like the existing
`centeredScaleMigrated`. *Effort:* S–M.

### 1.4 ⚠️ Model strategy — reconsider in light of 2.1.6
Do **not** simply promote jina-v5-nano. The measured picture is now two-sided:

| | MiniLM (default) | jina-v5-nano |
| --- | --- | --- |
| link-recall@10 (plugin A/B) | 0.43 | **0.52** |
| download | small, permissive | ~250 MB, **CC-BY-NC** |
| indexing | batches 8, threads pay off | **1 thread**, no batching possible |
| cost on a 10 000-char note | — | **~5.0 s** (4 threads was *worse*: 7.2 s) |
| failure mode | robust | caused the 2.1.4 "memory access out of bounds" report |

**The important open question:** every lab number in this document is jina, but
the Tier 1 engine levers are worth **+0.09 to +0.13 recall on whatever model they
sit on**. So *MiniLM + CSLS + fusion* may well beat *jina alone* — at a fraction
of the indexing cost and none of the fragility. **This is one cheap experiment
(re-run the fusion harness on MiniLM vectors) and it should gate the model
decision for 3.0.** Until it runs, keep MiniLM as default and describe jina
honestly: higher quality, much slower, non-commercial licence.

### 1.5 ⚠️ Two integration risks to resolve during 1.1–1.2

**(a) CSLS fights the existing structural boost.** CSLS *penalises* hub notes;
the structural boost *rewards* well-linked (= hub) notes. Naively combined they
double-count in opposite directions. Fold the existing boost into the fusion
combiner as one more channel rather than a post-hoc additive term, and re-tune
`structureInfluence` once against held-out links.

**(b) Both are O(N²) as specified.** Exact CSLS radii need a k-NN for every note;
at 20 000 notes that is 4·10⁸ vector comparisons. The implementable form is a
**random-sample estimate** (mean/σ of each note's similarity to ~500 sampled
notes, computed once, updated incrementally), reusing the Stage-1 shortlist
machinery and the existing `async-yield` budget so the UI never stalls.
PPR likewise runs as a truncated walk from the active note only, not all-pairs.
**The lab must confirm the gain survives the approximation** — measure sampled
CSLS vs exact CSLS on the same harness before shipping.

---

## Tier 2 — New surfaces (lab-validated)

### 2.1 **Surprising Connections** — the 3.0 flagship
Pairs ranked by *predicted link* (fusion), then filtered to **non-obvious**: both
notes outside each other's content top-10, ≥1 shared neighbour, hub-filtered,
content-cosine floored.
- Measured: **precision@5 = 1.00**, **36% judged genuinely surprising**, 868
  candidates mined.
- Why it is different: every earlier discovery feature collapsed into obvious
  siblings because ranking by similarity *is* ranking by obviousness. Here
  non-obviousness is the **filter**, so it cannot structurally collapse.
- Receipt = the two notes + the bridging neighbour ("connected via
  [[Hash Function]]").
- *Effort:* M, mostly reuses 1.2. *Depends on:* 1.2.

### 2.2 Suggested links **with receipts**
Upgrade Vault Insights' suggested links from bare similarity to detail-backed
suggestions using the continuity gate stack (boilerplate dedup, language-frequency
gate with term-evidence rescue, number-trigram consistency, co-citation kill,
copy-paste kill, header-receipt kill).
- Measured: judged precision **0.40 → 0.60 @15**.
- ⚠️ **Review constraint:** the lab version bundles ~200 KB of DE/EN frequency
  lists (CC-BY-SA). Bundling is allowed but inflates `main.js` and needs
  attribution. **Cheaper alternative worth measuring first:** derive
  "language-common" from the *vault's own* document frequencies instead of an
  external list — no bundle, no licence, and it adapts to the user's languages.
- *Effort:* M. Reference implementation: `bench/continuity-eval.mjs`.

### 2.3 Vault map (cartography)
2-D projection, clusters coloured and labelled.
- Measured: purity **0.605** vs folders, against a permutation baseline of 0.36
  and a language-split baseline of 0.40 — the structure is real, not an artefact.
- Zero false-positive cost (it is a visualisation), high demo value.
- *Effort:* M. Working prototype: `lab/results/vault-map.svg`.

### 2.4 Write-time link autocomplete (the honest half of "resurfacing")
Measured **useful 16/23 (0.70)** but non-obvious only **1/23** — a very good link
assistant, *not* the magic feature. Feature B already owns this surface; the
upgrade is to rank with the fused score and show the matching passage as receipt.
*Effort:* S given 1.2. Do not market it as discovery.

---

## Tier 3 — One lab run away

### 3.1 Concept search, redesigned (fixes ARCHITECTURE §5)
The planned cluster-seeded prototype **failed**: P@10 0.3–0.7 erratic, cold-start
(Physik, no MOC) **0.0**. Root cause is hypernym asymmetry — cosine is symmetric,
so the note *about* a category outranks its members. Replacement to test: Rocchio
pseudo-relevance feedback + CSLS + demoting the category note itself.
**Rewrite §5 either way.**

### 3.2 Clustering with abstention + better labels
Co-association ensemble over restarts; notes below a stability threshold left
**unassigned** (precision-first clustering); labels from c-TF-IDF + MMR rather
than df-idf keyphrases. Feeds 2.3 and suggested-tags quality.

### 3.3 Tag suggestion via label propagation with a margin gate
Propagate over the fused graph, gate on confidence margin (the margin is the
silence mechanism).

### 3.4 Graded surprise (PURS-style) for 2.1
Replace the binary "outside top-10" filter with a graded unexpectedness score, so
the feed ranks by *how* surprising.

### 3.5 Cross-language twins — hygiene, not discovery
9 DE↔EN pairs at ≥0.85, planted pair at #5. Judging was cut short by a session
limit; provisional read is that most are *deliberate* bilingual counterparts →
fold into merge/hygiene suggestions and use as the dupe-alarm exemption class.
Finish the judging pass before deciding.

---

## Tier 4 — Platform & polish

- **Bases API integration** — semantic columns / live semantic views. Highest new
  surface value; needs a spike against the current API.
- **Reliability budget (explicit).** Given the 2.1.x arc, each new subsystem
  ships with: a degradation path, a cancel token, an "off" switch, and a rule
  that it never blocks indexing. Extend the existing bench CI
  (`glue-prelude-check`, `ort-runtime-check`) with a graph-signals smoke test.
- **Cold-start onboarding** — measured as the weakest moment: with few notes the
  rankings are noise, and that is exactly when a user judges the plugin. Lead
  with structural signals (which work immediately) and set expectations.
- **Index format v3 + one migration** — CSLS/MP constants, graph signals and
  calibration all want persisting. Do it once, with 3.0.
- **Performance** — the whole-note path is single-threaded by design now; if jina
  stays a supported option, consider a progress/ETA affordance for long first
  indexes, since ~5 s/note on long notes is user-visible.
- **Docs** — rewrite README Roadmap, ARCHITECTURE §5, add a measured-numbers
  table, regenerate `docs/roadmap.svg`.
- **Mobile** — `isDesktopOnly: true`. Out of scope unless you want it; WASM
  threading and the model download are the blockers.

---

## Explicitly OUT of 3.0 (measured dead ends)

| Rejected | Why (measured) |
| --- | --- |
| Bridges (A–B–C triangles) | 0/23 endpoint recall, ≤0.08 precision; template artefacts |
| Residual / centroid-subtracted "parallels" | 0/12 blind precision; dominated by language & register |
| Answered Questions as an answer surface | retrieval fine (recall@3 = 1.0) but symmetric embeddings cannot judge answer-relevance; 0/14 judged. Keep as link fuel |
| Dupe alarm as a loud alarm | no threshold gives recall 0.9 at ≤1% false alarms — sibling notes score 0.86. Exact copies → content hash; paraphrase signal → hygiene |
| Resurfacing as *the magic feature* | 1/23 non-obvious. Ships as autocomplete (2.4) |
| Chat / generation / runtime LLM | product constraint |

---

## Suggested release shape

**Before 3.0 starts — two cheap experiments that change the plan:**
1. **MiniLM + CSLS + fusion vs jina alone** (§1.4). Decides the model story.
2. **Sampled vs exact CSLS** (§1.5b). Decides whether 1.1 is shippable at scale.

**3.0 — "The engine and the flagship"**
Tier 0 · 1.1 · 1.2 · 1.3 · 2.1 · 2.2 · 2.4 · index migration · docs rewrite.
The release that can honestly claim *+13% retrieval quality on every surface*
plus one genuinely new discovery feature.

**3.1 — "Seeing the vault"**
2.3 map · 3.2 clustering + labels · 3.3 tag propagation · cold-start onboarding.

**3.2 — "Search that understands categories"**
3.1 concept search (after its experiment) · Bases · 3.4 graded surprise.

Sequencing: 2.1 and 2.2 both depend on 1.2; 1.1/1.3 change stored scores — so the
index migration lands with 3.0 rather than being retrofitted.
