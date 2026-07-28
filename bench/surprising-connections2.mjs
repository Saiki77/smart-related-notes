// "Surprising Connections" v2 — the 3.0 magic-feature candidate, hardened.
//
// Mechanism (unchanged core): rank unlinked note pairs by FUSED score
// z(content cosine) + z(Adamic-Adar), then KEEP ONLY pairs that are mutually
// OUTSIDE each other's content top-10. Non-obviousness is the filter, not an
// afterthought — so the surface structurally cannot collapse into the obvious
// siblings that killed continuity and resurfacing.
//
// v2 fixes (from the v1 inspection):
//   H  hub filter      — index/overview notes (Uebersicht, MOC, Index, Vault
//                        Insights) and top-degree notes may not be ENDPOINTS,
//                        and a shared neighbour only counts as evidence if it is
//                        a specific note (degree <= SC_BRIDGE_DEG)
//   F  content floor   — contentCos >= SC_COS_FLOOR kills near-orthogonal noise
//                        (v1 surfaced Markov<->Attention at cos 0.014)
//   R  receipts        — for each pair, the sentence from EACH note that mentions
//                        the shared bridge note, so the claim is verifiable in ~2s
//
//   node bench/surprising-connections2.mjs
// Env: SC_COS_FLOOR (0.15), SC_BRIDGE_DEG (25), SC_HUB_PCTL (0.95), SC_TOP (30)
import { readFileSync, writeFileSync } from "node:fs";
import { relative } from "node:path";
import { walk, stripFront, noteText } from "./jina-cache.mjs";

const VAULT = "/Users/justus/obsidian_atomized_intermediary/lab/vault";
const CACHE_PATH = process.env.JINA_CACHE || (process.env.HOME + "/.cache/srn-lab/jina-cache.json");
const OUT = "/Users/justus/obsidian_atomized_intermediary/lab/results/surprising-connections2.json";
const PREFIX = "Document: ";
const COS_FLOOR = Number(process.env.SC_COS_FLOOR ?? 0.15);
const BRIDGE_DEG = Number(process.env.SC_BRIDGE_DEG ?? 25);
const HUB_PCTL = Number(process.env.SC_HUB_PCTL ?? 0.95);
const TOP = Number(process.env.SC_TOP ?? 30);

const fold = (s) => s.toLowerCase().normalize("NFC").replace(/ä/g, "ae").replace(/ö/g, "oe").replace(/ü/g, "ue").replace(/ß/g, "ss").replace(/\s+/g, " ").trim();
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const l2 = (v) => { let s = 0; for (const x of v) s += x * x; s = Math.sqrt(s) || 1; return v.map((x) => x / s); };
const zn = (arr) => { const m = arr.reduce((s, x) => s + x, 0) / arr.length; const sd = Math.sqrt(arr.reduce((s, x) => s + (x - m) ** 2, 0) / arr.length) || 1; return arr.map((x) => (x - m) / sd); };

const manifest = JSON.parse(readFileSync("/Users/justus/obsidian_atomized_intermediary/lab/corpus-manifest.json", "utf8"));
const INDEXABLE = new Set(manifest.answer_paths.map((p) => "real/" + p));
const cache = JSON.parse(readFileSync(CACHE_PATH, "utf8"));
const vec = (t) => cache[PREFIX + t] || null;

const HUB_RE = /\bMOC\b|Uebersicht|Übersicht|Zettelkasten Index|Vault Insights|^Untitled$/i;
const notes = [];
for (const abs of walk(VAULT)) {
  const rel = relative(VAULT, abs);
  if (rel.startsWith("real/") && !INDEXABLE.has(rel)) continue;
  const bn = rel.replace(/\.md$/, "").split("/").pop();
  if (/(^|\/)Attachments\//.test(rel) || /\.dup$/.test(bn)) continue;
  const body = stripFront(readFileSync(abs, "utf8"));
  notes.push({ rel, bn, body, isHub: HUB_RE.test(bn), v: vec(noteText(bn, body)) });
}
const wv = notes.filter((n) => n.v);
const N = wv.length, DIM = wv[0].v.length;
const idx = new Map(wv.map((n, i) => [n, i]));
const byFold = new Map(wv.map((n) => [fold(n.bn), n]));
const mean = new Array(DIM).fill(0);
for (const n of wv) for (let i = 0; i < DIM; i++) mean[i] += n.v[i];
for (let i = 0; i < DIM; i++) mean[i] /= N;
const C = wv.map((n) => l2(n.v.map((x, i) => x - mean[i])));

// adjacency (undirected, resolved wikilinks)
const adj = Array.from({ length: N }, () => new Set());
for (const n of wv) for (const m of n.body.matchAll(/\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]/g)) {
  const t = byFold.get(fold(m[1]));
  if (!t || t === n) continue;
  adj[idx.get(n)].add(idx.get(t)); adj[idx.get(t)].add(idx.get(n));
}
const deg = adj.map((s) => s.size);
const degSorted = [...deg].sort((a, b) => a - b);
const HUB_DEG = degSorted[Math.floor(degSorted.length * HUB_PCTL)];
const isHubNode = (i) => wv[i].isHub || deg[i] > HUB_DEG;
console.log(`${N} notes; degree p${HUB_PCTL * 100} = ${HUB_DEG}; hub-excluded endpoints: ${wv.filter((_, i) => isHubNode(i)).length}`);

// content top-10 (the obviousness set)
const contentTop10 = [];
for (let s = 0; s < N; s++) {
  const sc = [];
  for (let t = 0; t < N; t++) if (t !== s) sc.push([dot(C[s], C[t]), t]);
  sc.sort((a, b) => b[0] - a[0]);
  contentTop10.push(new Set(sc.slice(0, 10).map((x) => x[1])));
}
// Adamic-Adar restricted to SPECIFIC bridges (hub notes carry no evidence)
const specificShared = (s, t) => [...adj[s]].filter((x) => adj[t].has(x) && !isHubNode(x) && deg[x] <= BRIDGE_DEG);
const aaSpecific = (s, t) => specificShared(s, t).reduce((a, x) => a + 1 / Math.log(1 + deg[x]), 0);

// receipt: the sentence in note n that mentions bridge b
function bridgeSentence(n, bridgeName) {
  const clean = n.body.replace(/```[\s\S]*?```/g, " ");
  const needle = fold(bridgeName);
  for (const raw of clean.split(/(?<=[.!?])\s+|\n/)) {
    const line = raw.replace(/^\s*[>*+\-#]+\s*/, "").replace(/\[\[([^\]|]+)\|([^\]]+)\]\]/g, "$2").replace(/\[\[([^\]]+)\]\]/g, "$1").replace(/[*_~`]/g, "").replace(/\s+/g, " ").trim();
    if (line.length >= 25 && fold(line).includes(needle)) return line.slice(0, 260);
  }
  return null;
}

const seen = new Set(), out = [];
let cutHub = 0, cutFloor = 0, cutObvious = 0, cutNoBridge = 0;
for (let s = 0; s < N; s++) {
  if (isHubNode(s)) { cutHub++; continue; }
  const cands = [];
  for (let t = 0; t < N; t++) {
    if (t === s || adj[s].has(t) || isHubNode(t)) continue;
    cands.push(t);
  }
  if (!cands.length) continue;
  const zc = zn(cands.map((t) => dot(C[s], C[t])));
  const za = zn(cands.map((t) => aaSpecific(s, t)));
  const ranked = cands.map((t, i) => [zc[i] + za[i], t]).sort((a, b) => b[0] - a[0]);
  for (const [score, t] of ranked.slice(0, 6)) {
    const cc = dot(C[s], C[t]);
    const shared = specificShared(s, t);
    if (!shared.length) { cutNoBridge++; continue; }
    if (cc < COS_FLOOR) { cutFloor++; continue; }
    if (contentTop10[s].has(t) || contentTop10[t].has(s)) { cutObvious++; continue; }
    const key = [Math.min(s, t), Math.max(s, t)].join("-");
    if (seen.has(key)) continue;
    seen.add(key);
    const bridges = shared.sort((a, b) => deg[a] - deg[b]).map((x) => wv[x].bn);
    const primary = bridges[0];
    out.push({
      score: +score.toFixed(2), contentCos: +cc.toFixed(3), aaSpecific: +aaSpecific(s, t).toFixed(2),
      a: wv[s].rel, b: wv[t].rel, bridges: bridges.slice(0, 4),
      receiptA: bridgeSentence(wv[s], primary), receiptB: bridgeSentence(wv[t], primary),
    });
  }
}
out.sort((a, b) => b.score - a.score);
const withReceipts = out.filter((e) => e.receiptA && e.receiptB);
console.log(`mined ${out.length} candidates (${withReceipts.length} with both receipts); filtered: hubEndpoint ${cutHub}, noSpecificBridge ${cutNoBridge}, belowCosFloor ${cutFloor}, obvious ${cutObvious}`);
for (const e of withReceipts.slice(0, 20)) {
  console.log(`  ${e.score.toFixed(2)} cos ${e.contentCos} | ${e.a.split("/").pop().replace(".md", "").slice(0, 28).padEnd(29)} <> ${e.b.split("/").pop().replace(".md", "").slice(0, 28).padEnd(29)} via [${e.bridges.slice(0, 2).join(", ")}]`);
}
writeFileSync(OUT, JSON.stringify({
  config: { COS_FLOOR, BRIDGE_DEG, HUB_PCTL, HUB_DEG },
  counts: { mined: out.length, withReceipts: withReceipts.length, cutHub, cutNoBridge, cutFloor, cutObvious },
  items: withReceipts.slice(0, TOP),
}, null, 1));
console.log("wrote", OUT);
