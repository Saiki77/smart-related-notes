# Smart Related Notes 3.0 — the sidebar now reads your links, not just your words

Draft for the Obsidian forum (Share & showcase). Numbers checked against
community-plugin-stats.json on 2026-07-28; the download figure there was last
refreshed 2026-07-26 19:19 UTC.

---

Smart Related Notes just passed **1,870 downloads**, and 3.0 is out today. Thank you — genuinely. It started as a thing I built because I kept failing to find notes I knew I had written.

![Smart Related Notes 3.0](https://raw.githubusercontent.com/Saiki77/smart-related-notes/main/docs/demo.gif)

## What it does

A panel in the left sidebar showing the notes most related to the one you are reading. No query, no tags, no folders. It runs a small multilingual embedding model **entirely on your machine** — after a one-time model download it works fully offline, and it matches across German, English and 100+ other languages, so a German note surfaces for an English one.

## What is new in 3.0

**Ranking stopped being about wording alone.**

I measured this before building it, by masking 20% of the wikilinks in a 494-note vault and asking whether the panel could recover them. Similarity alone found 0.664 of them in its top 10. The thing that struck me: about **30% of the links people actually make point at a note that is not in the source's content top-10 at all**. Similarity structurally cannot look there.

So 3.0 adds a second channel that reads the shape of your own link graph — notes that share context you have already connected them through. Fusing it with content:

| | content only | with the link graph |
|---|---|---|
| recall@10 | 0.664 | **0.745** |

That channel earned its place by being *independent* of the first: its rank correlation with the content channel is 0.075. I also built a lexical keyword channel, measured it at 0.30 correlation, found it made things worse, and cut it.

**Cards now say why.** When the graph is the reason a note surfaced, the card names the note that bridges them — `via Hash Function`. You can judge it at a glance instead of trusting a percentage.

**Surprising connections.** A discovery list of pairs your links connect through shared context but whose wording is too different for similarity to ever pair them. Each note must sit *outside* the other's top matches to qualify, so it cannot fill up with obvious siblings.

**A map of your vault.** Every note a point, placed so related notes sit together, coloured by cluster and named automatically from the notes themselves. Cluster purity 0.65 against 0.29 for a shuffled baseline.

**Template de-crowding.** If you use a daily-note template, every daily note used to match only other daily notes — the template was doing the ranking, not the day. The shared skeleton is now subtracted in vector space. Measured end to end through the ranker: 8.17/10 down to 3.35/10.

## Things I cut

Feels worth saying, because "we added everything" is not a quality signal. Each of these was built, measured, and dropped: CSLS and Mutual Proximity hubness correction, a personal similarity metric learned from your own links, a lexical BM25 channel, a shared-tags channel, and rooted PageRank used as a score rather than as a nominator. They either failed to help or made things measurably worse.

Also worth admitting: the first 3.0 beta shipped a real bug. The template correction only reached the note-level vectors, while the panel actually ranks over chunk vectors, so my bench said 0.67/10 and the real app was still crowded. The harness and the plugin were measuring different code. The tests now drive the real index, and the de-crowding check is verified by ablation — with the fix stubbed out it has to fail, or it is not proving anything.

## Install

Settings → Community plugins → Browse → **Smart Related Notes**.

Desktop only, Obsidian 1.7.2+. Nothing is ever sent anywhere.

Repo, with the full architecture write-up and every measurement: https://github.com/Saiki77/smart-related-notes

Happy to answer anything about the ranking, and very interested in what it surfaces in vaults larger than mine.
