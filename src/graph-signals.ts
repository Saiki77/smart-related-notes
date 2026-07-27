import type { App } from "obsidian";

// =============================================================================
// GRAPH SIGNALS — the structural half of the hybrid ranker.
//
// The embedding answers "do these two notes SAY similar things". This module
// answers a different and largely independent question: "does the shape of the
// link graph predict that these two notes BELONG together". Measured on the lab
// vault (494 notes, 1327 links, 20% held out), adding it to the content channel
// moved held-out link-recall@10 from 0.357 to 0.653, and — the point of the
// exercise — recovered 60% of the links whose target is NOT in the source's
// content top-10, which the content channel cannot reach by construction.
//
// Two scores, both parameter-free, both computed from Obsidian's own
// resolvedLinks. No training, no persistence: the graph is rebuilt from the
// metadata cache in one pass and is cheap enough to redo when links change.
//
//   Resource Allocation  RA(s,t) = sum over shared neighbours x of 1/deg(x)
//     "how much evidence flows from s to t through the notes they both link".
//     A shared neighbour that links to everything (a MOC) carries almost no
//     evidence; a shared neighbour with two links carries a lot. This is the
//     degree penalty that makes RA beat plain shared-neighbour counting.
//
//   Rooted PageRank  ppr(s,.) = truncated random walk with restart from s
//     Catches evidence two and three hops out, which RA (one hop) misses.
//
// MEASURED DIVISION OF LABOUR — why RA scores and PPR only nominates.
// Held-out link-recall@10 on the lab vault, both supported model families:
//
//                              MiniLM      jina-v5-nano
//   content only               0.357          0.664
//   content + RA               0.651          0.745   <- best on both
//   content + RA + PPR         0.653          0.719   <- PPR costs jina 0.026
//
// PPR as a SCORE is a wash on one model and a regression on the other, so it is
// not in the score. It is still the only way to ENUMERATE candidates more than
// one hop out, which is where the non-obvious links live, so it nominates and RA
// judges. (Hubness is the reason: adding PPR to the score pushed the top-1
// concentration skew from 2.3 to 3.6 — a densely linked note is "near"
// everything. RA's 1/deg(x) term and the sqrt(deg) normalisation below are the
// counterweights.)
//
// Note also what is NOT here: CSLS/hubness correction of the CONTENT channel was
// measured and rejected — it helped MiniLM by 0.005 and cost jina 0.030. A
// correction that only helps the weaker model is not a platform feature.
// =============================================================================

// Walk depth for rooted PageRank. 3 hops is where the lab gains flattened; more
// hops cost time and mostly re-find hubs.
const PPR_STEPS = 3;
// Restart probability. 0.15 is the standard PageRank damping complement.
const PPR_ALPHA = 0.15;
// A note linking more than this is treated as an index/hub for evidence
// purposes: it still appears in results, it just stops being counted as
// meaningful shared context between two other notes.
const HUB_DEGREE = 40;
// Widest frontier the rooted walk may carry between hops.
const PPR_FRONTIER_MAX = 400;

export interface GraphNeighbourEvidence {
  /** Shared neighbours, strongest evidence first (basenames, for receipts). */
  shared: string[];
  /** Resource-Allocation score. */
  ra: number;
}

export class GraphSignals {
  private adj = new Map<string, Set<string>>();
  private degree = new Map<string, number>();
  private built = false;

  constructor(private app: App) {}

  /**
   * Invalidate and RELEASE. Dropping the maps matters as much as the flag: an
   * invalidated-but-never-queried graph would otherwise pin its adjacency
   * indefinitely, which on a large vault is tens of MB held by an idle plugin.
   */
  invalidate(): void {
    this.built = false;
    this.adj = new Map();
    this.degree = new Map();
  }

  /** Rebuild the undirected adjacency from resolvedLinks. O(edges). */
  private ensure(): void {
    if (this.built) return;
    this.adj.clear();
    this.degree.clear();
    const resolved = this.app.metadataCache.resolvedLinks ?? {};
    const link = (a: string, b: string) => {
      let set = this.adj.get(a);
      if (!set) {
        set = new Set<string>();
        this.adj.set(a, set);
      }
      set.add(b);
    };
    for (const [from, targets] of Object.entries(resolved)) {
      for (const to of Object.keys(targets ?? {})) {
        if (to === from) continue;
        link(from, to);
        link(to, from);
      }
    }
    for (const [path, set] of this.adj) this.degree.set(path, set.size);
    this.built = true;
  }

  /** Undirected neighbours of a note (empty set when it has no resolved links). */
  neighbours(path: string): ReadonlySet<string> {
    this.ensure();
    return this.adj.get(path) ?? EMPTY;
  }

  hasGraph(): boolean {
    this.ensure();
    return this.adj.size > 0;
  }

  /**
   * Resource Allocation between two notes, plus the shared neighbours that
   * produced it — the receipt a suggestion can show ("connected via [[X]]").
   * Hub neighbours are excluded from the evidence: a note both sides link to
   * only because it is an index says nothing about the pair.
   */
  evidence(a: string, b: string): GraphNeighbourEvidence {
    this.ensure();
    const na = this.adj.get(a);
    const nb = this.adj.get(b);
    if (!na || !nb) return { shared: [], ra: 0 };
    const scored: { path: string; w: number }[] = [];
    let ra = 0;
    // Iterate the smaller side.
    const [small, large] = na.size <= nb.size ? [na, nb] : [nb, na];
    for (const x of small) {
      if (!large.has(x)) continue;
      const deg = this.degree.get(x) ?? 1;
      if (deg > HUB_DEGREE) continue;
      const w = 1 / Math.max(1, deg);
      ra += w;
      scored.push({ path: x, w });
    }
    scored.sort((p, q) => q.w - p.w);
    return { shared: scored.map((s) => s.path), ra };
  }

  /**
   * Rooted PageRank mass from `from`, as a map of path -> score. Truncated at
   * PPR_STEPS hops. Target-degree-normalised (see the hubness warning above):
   * we divide by sqrt(deg) so a note does not score highly on every query merely
   * for being well connected.
   */
  pprFrom(from: string): Map<string, number> {
    this.ensure();
    const acc = new Map<string, number>();
    let front = new Map<string, number>([[from, 1]]);
    for (let step = 0; step < PPR_STEPS; step++) {
      const next = new Map<string, number>();
      for (const [node, mass] of front) {
        const nbrs = this.adj.get(node);
        if (!nbrs || nbrs.size === 0) continue;
        // A hub expands the frontier by its whole neighbourhood. Three hops
        // through a couple of index notes reaches most of a large vault, so
        // stepping through one is not worth the fan-out it causes.
        if (nbrs.size > HUB_DEGREE) continue;
        const share = (mass * (1 - PPR_ALPHA)) / nbrs.size;
        if (share < 1e-6) continue;
        for (const nb of nbrs) next.set(nb, (next.get(nb) ?? 0) + share);
      }
      // Keep only the heaviest part of the frontier. Without a cap the walk is
      // unbounded: measured on a dense graph it can reach hundreds of thousands
      // of nodes by hop three, on the main thread, per ranked note.
      if (next.size > PPR_FRONTIER_MAX) {
        const top = [...next.entries()].sort((a, b) => b[1] - a[1]).slice(0, PPR_FRONTIER_MAX);
        front = new Map(top);
      } else {
        front = next;
      }
      for (const [node, mass] of front) acc.set(node, (acc.get(node) ?? 0) + mass);
    }
    acc.delete(from);
    for (const [node, mass] of acc) {
      acc.set(node, mass / Math.sqrt(Math.max(1, this.degree.get(node) ?? 1)));
    }
    return acc;
  }

  /**
   * Structural candidates for `from`, best first — notes the graph predicts are
   * related REGARDLESS of what the embedding thinks. This is what lets fusion
   * recover the ~30% of real links whose target sits outside the content top-10;
   * a shortlist built from content alone can never contain them.
   * Already-linked notes are excluded: those are not predictions.
   */
  candidates(from: string, limit: number): { path: string; score: number }[] {
    this.ensure();
    const direct = this.adj.get(from) ?? EMPTY;
    const ppr = this.pprFrom(from);
    const out: { path: string; score: number }[] = [];
    for (const [path, mass] of ppr) {
      if (path === from || direct.has(path)) continue;
      const { ra } = this.evidence(from, path);
      if (ra <= 0 && mass <= 0) continue;
      out.push({ path, score: mass + ra });
    }
    out.sort((a, b) => b.score - a.score);
    return out.slice(0, limit);
  }
}

const EMPTY: ReadonlySet<string> = new Set<string>();

/**
 * Rank-fusion helper: z-normalise a raw score array in place-safe fashion.
 * Returns a function mapping a raw score to its z-score. A degenerate spread
 * (all candidates identical) maps everything to 0, so a flat channel simply
 * stops contributing instead of amplifying noise.
 */
export function zNormaliser(values: number[]): (x: number) => number {
  if (values.length === 0) return () => 0;
  let mean = 0;
  for (const v of values) mean += v;
  mean /= values.length;
  let variance = 0;
  for (const v of values) variance += (v - mean) * (v - mean);
  const sd = Math.sqrt(variance / values.length);
  if (!isFinite(sd) || sd < 1e-9) return () => 0;
  return (x: number) => (x - mean) / sd;
}
