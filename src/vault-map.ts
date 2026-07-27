// =============================================================================
// VAULT MAP — the whole vault as one picture.
//
// Every note becomes a point, positioned so that notes about similar things sit
// near each other, coloured by the cluster it belongs to and labelled with what
// that cluster is about. Nothing here can produce a false positive the way a
// suggestion can: it is a view of the index, so the only failure mode is a map
// that misleads. Measured by bench/v3-map.mjs against THIS code (lab vault, 494
// notes, k=16 via suggestK): purity 0.65 against the vault's own folders, where
// shuffling the labels scores 0.29 and splitting by language alone scores 0.28.
// So the structure it draws is topical, and it is not chance.
//
// Three steps, all plain vector maths on vectors the index already holds:
//   1. spherical k-means over the centered note vectors  -> cluster per note
//   2. 2-D PCA of the same vectors                       -> x, y per note
//   3. c-TF-IDF over each cluster's titles               -> a label per cluster
//
// All three are deterministic (fixed seed) so reopening the map does not reshuffle
// the picture under the user.
// =============================================================================

export interface MapPoint {
  path: string;
  title: string;
  x: number; // normalised 0..1
  y: number; // normalised 0..1
  cluster: number;
}

export interface MapCluster {
  id: number;
  label: string;
  size: number;
}

export interface VaultMap {
  points: MapPoint[];
  clusters: MapCluster[];
}

export interface MapInput {
  path: string;
  title: string;
  vec: Float32Array;
}

// Deterministic PRNG so the same vault always draws the same map.
function mulberry32(seed: number): () => number {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function dot(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function normalise(v: Float32Array): Float32Array {
  let s = 0;
  for (const x of v) s += x * x;
  s = Math.sqrt(s) || 1;
  const out = new Float32Array(new ArrayBuffer(v.length * 4));
  for (let i = 0; i < v.length; i++) out[i] = v[i] / s;
  return out;
}

// A sensible cluster count for a vault this size: enough groups to be informative,
// few enough to label. sqrt(n/2) is the usual rule of thumb, clamped.
export function suggestK(n: number): number {
  return Math.max(3, Math.min(16, Math.round(Math.sqrt(n / 2))));
}

// Spherical k-means with k-means++ seeding and a few restarts. Vectors are unit
// length, so cosine is a dot product and the centroid is the normalised sum.
function kmeans(vectors: Float32Array[], k: number, seed: number): number[] {
  const n = vectors.length;
  const dims = vectors[0].length;
  let best: { assign: number[]; inertia: number } | null = null;

  for (let restart = 0; restart < 3; restart++) {
    const rnd = mulberry32(seed + restart * 977);
    const centroids: Float32Array[] = [vectors[Math.floor(rnd() * n)]];
    while (centroids.length < k) {
      // k-means++: pick the next centre with probability proportional to its
      // distance from the nearest existing centre.
      const d2: number[] = [];
      let total = 0;
      for (const v of vectors) {
        let nearest = -1;
        for (const c of centroids) nearest = Math.max(nearest, dot(v, c));
        const d = Math.max(0, 1 - nearest);
        d2.push(d);
        total += d;
      }
      let pick = rnd() * total;
      let chosen = 0;
      for (; chosen < n - 1 && pick > d2[chosen]; chosen++) pick -= d2[chosen];
      centroids.push(vectors[chosen]);
    }

    const assign = new Array<number>(n).fill(0);
    for (let iter = 0; iter < 30; iter++) {
      let moved = 0;
      for (let i = 0; i < n; i++) {
        let bestC = 0;
        let bestS = -Infinity;
        for (let c = 0; c < k; c++) {
          const s = dot(vectors[i], centroids[c]);
          if (s > bestS) { bestS = s; bestC = c; }
        }
        if (assign[i] !== bestC) { assign[i] = bestC; moved++; }
      }
      for (let c = 0; c < k; c++) {
        const sum = new Float32Array(new ArrayBuffer(dims * 4));
        let count = 0;
        for (let i = 0; i < n; i++) {
          if (assign[i] !== c) continue;
          count++;
          for (let d = 0; d < dims; d++) sum[d] += vectors[i][d];
        }
        if (count > 0) centroids[c] = normalise(sum);
      }
      if (moved === 0) break;
    }
    let inertia = 0;
    for (let i = 0; i < n; i++) inertia += 1 - dot(vectors[i], centroids[assign[i]]);
    if (!best || inertia < best.inertia) best = { assign: assign.slice(), inertia };
  }
  return best ? best.assign : new Array<number>(n).fill(0);
}

// First two principal components by power iteration, deflating the first before
// finding the second. Enough for a readable scatter; we do not need exactness.
function pca2(vectors: Float32Array[], seed: number): [number, number][] {
  const n = vectors.length;
  const dims = vectors[0].length;
  const mean = new Float32Array(new ArrayBuffer(dims * 4));
  for (const v of vectors) for (let d = 0; d < dims; d++) mean[d] += v[d] / n;
  const X = vectors.map((v) => {
    const out = new Float32Array(new ArrayBuffer(dims * 4));
    for (let d = 0; d < dims; d++) out[d] = v[d] - mean[d];
    return out;
  });

  const rnd = mulberry32(seed);
  const comps: Float32Array[] = [];
  for (let c = 0; c < 2; c++) {
    let w: Float32Array = new Float32Array(new ArrayBuffer(dims * 4));
    for (let d = 0; d < dims; d++) w[d] = rnd() - 0.5;
    w = normalise(w);
    for (let iter = 0; iter < 40; iter++) {
      const next = new Float32Array(new ArrayBuffer(dims * 4));
      for (const x of X) {
        const s = dot(x, w);
        for (let d = 0; d < dims; d++) next[d] += s * x[d];
      }
      for (const prev of comps) {
        const s = dot(next, prev);
        for (let d = 0; d < dims; d++) next[d] -= s * prev[d];
      }
      w = normalise(next);
    }
    comps.push(w);
  }
  return X.map((x) => [dot(x, comps[0]), dot(x, comps[1])] as [number, number]);
}

// Words that carry no topic. Kept deliberately small and bilingual; the c-TF-IDF
// weighting below suppresses the rest by itself.
const LABEL_STOP = new Set(
  ("the a an of and or to in on for is are was with that this it as by from at " +
    "der die das und oder ein eine in im auf mit von zu fuer ist sind war den dem " +
    "des als bei aus nach vor durch ueber um beim zur zum")
    .split(" "),
);

// c-TF-IDF: score a term by how often it appears in THIS cluster against how
// widely it is spread across all clusters. Picks out what makes a cluster
// distinct rather than what is merely frequent (BERTopic's labelling trick).
function labelClusters(titles: string[], assign: number[], k: number): string[] {
  const perCluster: Map<string, number>[] = [];
  for (let c = 0; c < k; c++) perCluster.push(new Map());
  const clusterTotals = new Array<number>(k).fill(0);
  const docFreq = new Map<string, number>();

  for (let i = 0; i < titles.length; i++) {
    const c = assign[i];
    const seen = new Set<string>();
    for (const raw of titles[i].toLowerCase().split(/[^\p{L}\p{N}]+/u)) {
      if (raw.length < 3 || LABEL_STOP.has(raw)) continue;
      // Dated titles (daily notes) would otherwise label their whole cluster
      // "2026", which says nothing about what the notes are about.
      if (!/\p{L}/u.test(raw)) continue;
      perCluster[c].set(raw, (perCluster[c].get(raw) ?? 0) + 1);
      clusterTotals[c]++;
      seen.add(raw);
    }
    for (const w of seen) docFreq.set(w, (docFreq.get(w) ?? 0) + 1);
  }

  const labels: string[] = [];
  for (let c = 0; c < k; c++) {
    const scored: { word: string; score: number }[] = [];
    for (const [word, count] of perCluster[c]) {
      const tf = count / Math.max(1, clusterTotals[c]);
      const idf = Math.log(1 + titles.length / Math.max(1, docFreq.get(word) ?? 1));
      scored.push({ word, score: tf * idf });
    }
    scored.sort((a, b) => b.score - a.score);
    // Maximal-marginal-relevance-lite: skip a term that is a prefix/suffix of one
    // already chosen, so a label does not read "regression regressions linear".
    const picked: string[] = [];
    for (const { word } of scored) {
      if (picked.length >= 3) break;
      if (picked.some((p) => p.startsWith(word) || word.startsWith(p))) continue;
      picked.push(word);
    }
    labels.push(picked.join(" · ") || `Group ${c + 1}`);
  }
  return labels;
}

/**
 * Build the map. Deterministic for a given input order and seed. Returns
 * normalised 0..1 coordinates so the view can scale them to any viewport.
 */
export function buildVaultMap(input: MapInput[], seed = 42): VaultMap {
  if (input.length < 3) return { points: [], clusters: [] };
  const dims = input[0].vec.length;
  const usable = input.filter((i) => i.vec.length === dims);
  const vectors = usable.map((i) => normalise(i.vec));

  const k = Math.min(suggestK(usable.length), usable.length);
  const assign = kmeans(vectors, k, seed);
  const coords = pca2(vectors, seed);

  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const [x, y] of coords) {
    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  }
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;

  const points: MapPoint[] = usable.map((item, i) => ({
    path: item.path,
    title: item.title,
    x: (coords[i][0] - minX) / spanX,
    y: (coords[i][1] - minY) / spanY,
    cluster: assign[i],
  }));

  const labels = labelClusters(usable.map((i) => i.title), assign, k);
  const sizes = new Array<number>(k).fill(0);
  for (const c of assign) sizes[c]++;
  const clusters: MapCluster[] = [];
  for (let c = 0; c < k; c++) {
    if (sizes[c] === 0) continue;
    clusters.push({ id: c, label: labels[c], size: sizes[c] });
  }
  return { points, clusters };
}
