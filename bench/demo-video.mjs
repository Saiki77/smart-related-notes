// Generates the README showcase video as SVG frames, rendered to PNG and muxed by
// ffmpeg. Nothing here is a screen recording: the panel contents are the real
// rankings this plugin produced on the lab vault, and the map scene replays the
// actual projected points from bench/vault-map-figure.mjs.
//
//   node bench/demo-video.mjs <frames-dir> [mapdata.json]
//
// Design tokens are the 3.0 set, matching docs/*.svg exactly.
import { writeFileSync, readFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";

const OUT = process.argv[2];
const MAPDATA = process.argv[3];
mkdirSync(OUT, { recursive: true });

const W = 1200, H = 620, FPS = 25;

const C = {
  bg: "#191920", edge: "#2e2e38",
  card: "#15151b", cardEdge: "#2a2a34",
  inner: "#1e1e27", innerEdge: "#3a3a48",
  ink: "#ededf2", body: "#c8cbd6", mut: "#8a8f9e", dim: "#6b7080",
  green: "#4fc98a", recFill: "#1c2620", recEdge: "#3f6b52", recInk: "#8ee8ba",
  peri: "#8a93c8", gold: "#e6c074", cyan: "#52c8d0", grey: "#a7a7b4", rose: "#9a7f8b",
  track: "#23232d",
};
const FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif";

const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const clamp01 = (t) => Math.max(0, Math.min(1, t));
// easeOutCubic: motion decelerates into place, so a card settles rather than snapping.
const ease = (t) => 1 - Math.pow(1 - clamp01(t), 3);
const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const lerp = (a, b, t) => a + (b - a) * clamp01(t);

// ---------------------------------------------------------------- scene data
// Real panel output for "Backprop als Feedback beim Schreiben" in the lab vault.
const CARDS = [
  { t: "Backpropagation",          pct: 49, pills: [["Linked", "linked"]],                                 sub: "computing the gradient" },
  { t: "Gradient Descent",         pct: 35, pills: [["Related", "rel"], ["#machine-learning", "tag"]],     sub: "iterative optimization" },
  { t: "Machine Learning MOC",     pct: 32, pills: [["Related", "rel"], ["via Backpropagation", "via"]],   sub: "Klassifikation Decision Tree" },
  { t: "Recurrent Neural Network", pct: 16, pills: [["Related", "rel"], ["via Backpropagation", "via"]],   sub: "limitation during Backpropagation" },
];

const SCENES = [
  { id: "title",    dur: 2.4 },
  { id: "panel",    dur: 5.2 },
  { id: "graph",    dur: 5.6 },
  { id: "decrowd",  dur: 4.0 },
  { id: "map",      dur: 5.2 },
  { id: "outro",    dur: 2.6 },
];
const TOTAL = SCENES.reduce((a, s) => a + s.dur, 0);

// ---------------------------------------------------------------- primitives
function frameOpen() {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" font-family="${FONT}">
<rect width="${W}" height="${H}" fill="${C.bg}"/>`;
}
const frameClose = () => `</svg>`;

function text(x, y, s, { fill = C.ink, size = 15, weight = 400, anchor = "start", op = 1, ls = 0 } = {}) {
  if (op <= 0.001) return "";
  return `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" fill="${fill}" font-size="${size}" font-weight="${weight}" text-anchor="${anchor}" opacity="${op.toFixed(3)}" letter-spacing="${ls}">${esc(s)}</text>`;
}
function rect(x, y, w, h, { fill = C.card, stroke = C.cardEdge, r = 12, op = 1, sw = 1 } = {}) {
  if (op <= 0.001 || w <= 0 || h <= 0) return "";
  const st = stroke ? ` stroke="${stroke}" stroke-width="${sw}"` : "";
  return `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${w.toFixed(1)}" height="${h.toFixed(1)}" rx="${r}" fill="${fill}"${st} opacity="${op.toFixed(3)}"/>`;
}
// Pill widths are computed from a per-character advance so text never overruns its
// capsule; measuring properly is not available without a font engine.
const pillW = (label, size = 12) => Math.round(label.length * size * 0.58) + 22;
function pill(x, y, label, kind, op = 1) {
  if (op <= 0.001) return "";
  const w = pillW(label);
  const style = {
    linked: [C.recFill, C.recEdge, C.recInk],
    via:    [C.recFill, C.recEdge, C.recInk],
    rel:    [C.inner, C.innerEdge, C.mut],
    tag:    [C.inner, C.innerEdge, C.peri],
  }[kind] || [C.inner, C.innerEdge, C.mut];
  return rect(x, y, w, 20, { fill: style[0], stroke: style[1], r: 6, op })
    + text(x + w / 2, y + 14, label, { fill: style[2], size: 12, anchor: "middle", op });
}
function scorePill(x, y, pct, op = 1, tone = C.peri) {
  const w = 44;
  return rect(x, y, w, 22, { fill: tone, stroke: null, r: 11, op })
    + text(x + w / 2, y + 15.5, `${Math.round(pct)}%`, { fill: "#15151b", size: 12.5, weight: 600, anchor: "middle", op });
}
// Caption strip common to every scene, so the eye has one fixed place to read.
function caption(head, sub, op) {
  return text(40, 52, head, { size: 30, weight: 600, ls: -0.4, op })
    + text(40, 78, sub, { fill: C.mut, size: 15, op: op * 0.95 });
}

// ---------------------------------------------------------------- scenes
function sceneTitle(t, d) {
  const a = ease(t / 0.9);
  const b = ease((t - 0.5) / 1.0);
  const out = t > d - 0.4 ? 1 - ease((t - (d - 0.4)) / 0.4) : 1;
  let s = "";
  // wordmark spark
  const cx = 92, cy = 268;
  s += `<g opacity="${(a * out).toFixed(3)}"><path d="M ${cx} ${cy - 16} L ${cx + 5} ${cy - 5} L ${cx + 16} ${cy} L ${cx + 5} ${cy + 5} L ${cx} ${cy + 16} L ${cx - 5} ${cy + 5} L ${cx - 16} ${cy} L ${cx - 5} ${cy - 5} Z" fill="${C.peri}"/></g>`;
  s += text(124, cy + 14, "Smart Related Notes", { size: 52, weight: 600, ls: -1.2, op: a * out });
  s += text(126, cy + 58, "3.0", { size: 26, weight: 600, fill: C.green, op: b * out });
  s += text(126 + 56, cy + 58, "ranking stopped being about wording alone", { size: 20, fill: C.mut, op: b * out });
  return s;
}

function scenePanel(t, d) {
  const intro = ease(t / 0.5);
  const out = t > d - 0.4 ? 1 - ease((t - (d - 0.4)) / 0.4) : 1;
  const A = intro * out;
  let s = caption("The notes you would have linked", "Open a note. Every other note is ranked by meaning, on your machine.", A);
  const px = 40, py = 108, pw = 420, ph = 448;
  s += rect(px, py, pw, ph, { r: 14, op: A });
  s += text(px + 18, py + 30, "Smart related notes", { size: 15, weight: 600, op: A });
  s += text(px + 18, py + 50, "Based on Backprop als Feedback beim Schreiben", { fill: C.dim, size: 12, op: A });
  let cy = py + 68;
  for (let i = 0; i < CARDS.length; i++) {
    const c = CARDS[i];
    const start = 0.55 + i * 0.28;
    const k = ease((t - start) / 0.55);
    if (k <= 0.001) continue;
    const dy = (1 - k) * 16;
    const op = k * out;
    s += rect(px + 14, cy + dy, pw - 28, 82, { fill: C.inner, stroke: C.innerEdge, r: 10, op });
    s += text(px + 30, cy + dy + 24, c.t, { fill: C.body, size: 15, weight: 500, op });
    s += scorePill(px + pw - 72, cy + dy + 10, c.pct, op, c.pct >= 30 ? C.peri : C.innerEdge);
    let qx = px + 30;
    for (const [label, kind] of c.pills) { s += pill(qx, cy + dy + 34, label, kind, op); qx += pillW(label) + 8; }
    s += text(px + 30, cy + dy + 72, c.sub, { fill: C.dim, size: 11.5, op: op * 0.9 });
    cy += 92;
  }
  // right-hand explainer
  const e = ease((t - 1.6) / 0.8) * out;
  s += text(px + pw + 40, py + 96, "Not folders. Not tags.", { size: 22, weight: 600, op: e });
  s += text(px + pw + 40, py + 128, "A multilingual embedding model reads", { fill: C.mut, size: 15, op: e });
  s += text(px + pw + 40, py + 150, "each note and places it by what it means,", { fill: C.mut, size: 15, op: e });
  s += text(px + pw + 40, py + 172, "so a German note matches an English one.", { fill: C.mut, size: 15, op: e });
  const e2 = ease((t - 2.6) / 0.8) * out;
  s += rect(px + pw + 40, py + 200, 300, 64, { fill: C.recFill, stroke: C.recEdge, r: 10, op: e2 });
  s += text(px + pw + 58, py + 228, "Runs entirely on your machine", { fill: C.recInk, size: 14, weight: 600, op: e2 });
  s += text(px + pw + 58, py + 250, "No cloud, no API key, offline after setup", { fill: C.mut, size: 12.5, op: e2 });
  return s;
}

function sceneGraph(t, d) {
  const intro = ease(t / 0.5);
  const out = t > d - 0.4 ? 1 - ease((t - (d - 0.4)) / 0.4) : 1;
  const A = intro * out;
  let s = caption("Now it reads your links, not just your words", "Two notes that never mention each other, connected through one you linked to both.", A);
  const bx = 40, by = 112, bw = 700, bh = 300;
  s += rect(bx, by, bw, bh, { r: 14, op: A });

  const L = { x: bx + 130, y: by + 96 }, R = { x: bx + 570, y: by + 96 }, M = { x: bx + 350, y: by + 216 };
  // the two notes
  const nk = ease((t - 0.3) / 0.6) * out;
  const nodeBox = (p, title, sub, op) =>
    rect(p.x - 100, p.y - 34, 200, 62, { fill: C.inner, stroke: C.innerEdge, r: 10, op })
    + text(p.x, p.y - 10, title, { fill: C.body, size: 14, weight: 600, anchor: "middle", op })
    + text(p.x, p.y + 10, sub, { fill: C.dim, size: 11.5, anchor: "middle", op });
  s += nodeBox(L, "Entropy", "information theory", nk);
  s += nodeBox(R, "Hash Table", "data structures", nk);

  // "no shared wording" crossed link
  const xk = ease((t - 0.9) / 0.5) * out;
  if (xk > 0.001) {
    s += `<line x1="${L.x + 100}" y1="${L.y}" x2="${R.x - 100}" y2="${R.y}" stroke="${C.innerEdge}" stroke-width="1.5" stroke-dasharray="5 5" opacity="${(xk * out).toFixed(3)}"/>`;
    const mx = (L.x + R.x) / 2;
    s += text(mx, L.y - 12, "no wording in common", { fill: C.dim, size: 12, anchor: "middle", op: xk });
    s += `<g opacity="${xk.toFixed(3)}"><line x1="${mx - 9}" y1="${L.y - 9}" x2="${mx + 9}" y2="${L.y + 9}" stroke="${C.rose}" stroke-width="2"/><line x1="${mx + 9}" y1="${L.y - 9}" x2="${mx - 9}" y2="${L.y + 9}" stroke="${C.rose}" stroke-width="2"/></g>`;
  }
  // the bridging note and its two edges, drawn on
  const bk = ease((t - 1.6) / 0.7) * out;
  if (bk > 0.001) {
    s += nodeBox(M, "Hash Function", "you linked both of these", bk);
    const draw = (from, to, k) => {
      const x2 = lerp(from.x, to.x, k), y2 = lerp(from.y, to.y, k);
      return `<line x1="${from.x}" y1="${from.y}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="${C.green}" stroke-width="2" opacity="${(0.85 * out).toFixed(3)}"/>`;
    };
    const ek = ease((t - 2.1) / 0.7);
    if (ek > 0.001) {
      s += draw({ x: M.x - 60, y: M.y - 34 }, { x: L.x, y: L.y + 28 }, ek);
      s += draw({ x: M.x + 60, y: M.y - 34 }, { x: R.x, y: R.y + 28 }, ek);
    }
  }
  // the receipt card
  const rk = ease((t - 3.0) / 0.7) * out;
  if (rk > 0.001) {
    const cx = bx + bw + 40;
    s += text(cx, by + 40, "So the panel offers it,", { size: 20, weight: 600, op: rk });
    s += text(cx, by + 66, "and says why:", { size: 20, weight: 600, op: rk });
    s += rect(cx, by + 92, 340, 78, { fill: C.inner, stroke: C.innerEdge, r: 10, op: rk });
    s += text(cx + 18, by + 122, "Hash Table", { fill: C.body, size: 15, weight: 500, op: rk });
    s += scorePill(cx + 340 - 62, by + 106, 12, rk, C.innerEdge);
    s += pill(cx + 18, by + 136, "via Hash Function", "via", rk);
    const mk = ease((t - 3.7) / 0.7) * out;
    s += text(cx, by + 200, "Held-out link recall", { fill: C.mut, size: 13, op: mk });
    s += text(cx, by + 238, "0.75", { fill: C.green, size: 40, weight: 600, op: mk });
    s += text(cx + 92, by + 238, "from 0.66", { fill: C.dim, size: 18, op: mk });
  }
  return s;
}

function sceneDecrowd(t, d) {
  const intro = ease(t / 0.5);
  const out = t > d - 0.4 ? 1 - ease((t - (d - 0.4)) / 0.4) : 1;
  const A = intro * out;
  let s = caption("Notes from one template stop matching only each other", "A daily note used to return ten more daily notes. The shared skeleton is subtracted in vector space.", A);
  const bx = 40, by = 128, bw = 1120, bh = 300;
  s += rect(bx, by, bw, bh, { r: 14, op: A });
  // ten slots; the "before" row is all template, the "after" row mostly content
  const slot = (i, on, op, y) => {
    const x = bx + 60 + i * 100;
    return rect(x, y, 84, 56, { fill: on ? C.recFill : C.inner, stroke: on ? C.recEdge : C.innerEdge, r: 8, op })
      + text(x + 42, y + 33, on ? "daily" : "content", { fill: on ? C.recInk : C.dim, size: 12, anchor: "middle", op });
  };
  const k1 = ease((t - 0.3) / 0.6) * out;
  s += text(bx + 60, by + 40, "before", { fill: C.mut, size: 13, op: k1 });
  for (let i = 0; i < 10; i++) s += slot(i, true, k1 * ease((t - 0.3 - i * 0.03) / 0.5), by + 56);
  const k2 = ease((t - 1.5) / 0.8) * out;
  s += text(bx + 60, by + 172, "after", { fill: C.mut, size: 13, op: k2 });
  // 3.35 of 10 rounds to three template slots surviving
  for (let i = 0; i < 10; i++) s += slot(i, i < 3, k2 * ease((t - 1.5 - i * 0.03) / 0.5), by + 188);
  const k3 = ease((t - 1.5) / 0.6) * out;
  const val = lerp(8.2, 3.4, easeInOut(clamp01((t - 1.5) / 0.85)));
  s += text(bx + 60, by + 282, `crowding ${val.toFixed(1)} of 10`, { fill: C.green, size: 17, weight: 600, op: k3 });
  s += text(bx + 260, by + 282, "measured through the ranker itself, from 8.2", { fill: C.dim, size: 13, op: k3 });
  return s;
}

function sceneMap(t, d, map) {
  const intro = ease(t / 0.5);
  const out = t > d - 0.4 ? 1 - ease((t - (d - 0.4)) / 0.4) : 1;
  const A = intro * out;
  let s = caption("And a map of the whole vault", "Every note a point. Nothing was tagged or foldered: the positions and the cluster names come out of the notes.", A);
  const bx = 40, by = 112, bw = 760, bh = 452;
  s += rect(bx, by, bw, bh, { r: 14, op: A });
  const COL = [C.green, C.peri, C.gold, C.cyan, C.grey, C.rose];
  const { pts, xr, yr } = map;
  const sx = (x) => bx + 20 + (x - xr[0]) / (xr[1] - xr[0]) * (bw - 40);
  const sy = (y) => by + 20 + (1 - (y - yr[0]) / (yr[1] - yr[0])) * (bh - 40);
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    // points arrive in a wave ordered by cluster, so the grouping reveals itself
    const k = ease((t - 0.4 - p.k * 0.16 - (i % 40) * 0.004) / 0.5);
    if (k <= 0.001) continue;
    s += `<circle cx="${sx(p.x).toFixed(1)}" cy="${sy(p.y).toFixed(1)}" r="${(3 * k).toFixed(2)}" fill="${COL[p.k % 6]}" opacity="${(0.66 * k * out).toFixed(3)}"/>`;
  }
  const lx = bx + bw + 28;
  s += text(lx, by + 28, "CLUSTERS THE PLUGIN FOUND AND NAMED", { fill: C.dim, size: 12, op: A });
  let ly = by + 60;
  for (const l of map.legend) {
    const k = ease((t - 0.9 - l.k * 0.16) / 0.5) * out;
    s += `<circle cx="${lx + 7}" cy="${ly - 5}" r="5" fill="${COL[l.k % 6]}" opacity="${k.toFixed(3)}"/>`;
    s += text(lx + 22, ly, l.name, { fill: C.body, size: 14, op: k });
    s += text(lx + 22, ly + 17, `${l.n} notes`, { fill: C.dim, size: 12, op: k });
    ly += 46;
  }
  return s;
}

function sceneOutro(t, d) {
  const a = ease(t / 0.6);
  const out = t > d - 0.5 ? 1 - ease((t - (d - 0.5)) / 0.5) : 1;
  const A = a * out;
  let s = text(W / 2, 250, "Smart Related Notes 3.0", { size: 40, weight: 600, anchor: "middle", ls: -0.8, op: A });
  const b = ease((t - 0.4) / 0.6) * out;
  s += text(W / 2, 296, "Community plugins  ›  Browse  ›  Smart Related Notes", { fill: C.mut, size: 18, anchor: "middle", op: b });
  const c = ease((t - 0.8) / 0.6) * out;
  s += rect(W / 2 - 210, 330, 420, 44, { fill: C.recFill, stroke: C.recEdge, r: 10, op: c });
  s += text(W / 2, 358, "Local, offline, and measured before it ships", { fill: C.recInk, size: 15, anchor: "middle", op: c });
  return s;
}

// ---------------------------------------------------------------- driver
let map = { pts: [], legend: [], xr: [-1, 1], yr: [-1, 1] };
if (MAPDATA) {
  const raw = JSON.parse(readFileSync(MAPDATA, "utf8"));
  const NAMES = { 0: "daily notes", 1: "machine learning", 2: "patterns / databases", 3: "linear algebra", 4: "drafts and stubs", 5: "algorithms / search" };
  const xs = raw.pts.map((p) => p.x), ys = raw.pts.map((p) => p.y);
  const counts = new Map();
  for (const p of raw.pts) counts.set(p.k, (counts.get(p.k) ?? 0) + 1);
  map = {
    pts: raw.pts,
    xr: [Math.min(...xs), Math.max(...xs)],
    yr: [Math.min(...ys), Math.max(...ys)],
    legend: [...counts.entries()].sort((a, b) => b[1] - a[1]).map(([k, n]) => ({ k, n, name: NAMES[k] ?? `cluster ${k + 1}` })),
  };
}

const total = Math.round(TOTAL * FPS);
let n = 0;
for (const sc of SCENES) {
  const frames = Math.round(sc.dur * FPS);
  for (let f = 0; f < frames; f++) {
    const t = f / FPS;
    let body = "";
    if (sc.id === "title") body = sceneTitle(t, sc.dur);
    else if (sc.id === "panel") body = scenePanel(t, sc.dur);
    else if (sc.id === "graph") body = sceneGraph(t, sc.dur);
    else if (sc.id === "decrowd") body = sceneDecrowd(t, sc.dur);
    else if (sc.id === "map") body = sceneMap(t, sc.dur, map);
    else if (sc.id === "outro") body = sceneOutro(t, sc.dur);
    // progress hairline, so the viewer can see how long is left
    const prog = n / total;
    body += rect(0, H - 3, W * prog, 3, { fill: C.green, stroke: null, r: 0, op: 0.75 });
    writeFileSync(join(OUT, `f${String(n).padStart(5, "0")}.svg`), frameOpen() + body + frameClose());
    n++;
  }
}
console.log(`${n} frames, ${TOTAL.toFixed(1)}s at ${FPS}fps -> ${OUT}`);
