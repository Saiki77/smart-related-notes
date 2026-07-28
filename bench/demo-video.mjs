// Generates the README showcase video as SVG frames, rendered to PNG and muxed by
// ffmpeg. Not a screen recording: every frame is drawn, so the video is a diff and
// can be rebuilt when a number changes.
//
//   node bench/demo-video.mjs <frames-dir> [mapdata.json]
//
// The content is real. Panel cards are the rankings this plugin returned for
// "Backprop als Feedback beim Schreiben" in the lab vault, and the map scene
// replays the projected points from bench/vault-map-figure.mjs.
//
// It is built to be watched, not read: the note lives inside an Obsidian window the
// whole time, the abstract steps (a note becoming a point, a point finding its
// neighbours) happen ON that window, and captions are one short line.
import { writeFileSync, readFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";

const OUT = process.argv[2];
const MAPDATA = process.argv[3];
mkdirSync(OUT, { recursive: true });

const W = 1200, H = 620, FPS = 25;

const C = {
  page: "#111116", bg: "#191920", edge: "#2e2e38",
  card: "#15151b", cardEdge: "#2a2a34",
  inner: "#1e1e27", innerEdge: "#3a3a48",
  ink: "#ededf2", body: "#c8cbd6", mut: "#8a8f9e", dim: "#6b7080", faint: "#3a3a48",
  green: "#4fc98a", recFill: "#1c2620", recEdge: "#3f6b52", recInk: "#8ee8ba",
  peri: "#8a93c8", gold: "#e6c074", cyan: "#52c8d0", grey: "#a7a7b4", rose: "#9a7f8b",
};
const FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif";

const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const cl = (t) => Math.max(0, Math.min(1, t));
const ease = (t) => 1 - Math.pow(1 - cl(t), 3);            // decelerate into place
const easeIO = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const back = (t) => { const c = 1.70158, u = cl(t) - 1; return 1 + (c + 1) * u * u * u + c * u * u; };
const L = (a, b, t) => a + (b - a) * cl(t);

function T(x, y, s, o = {}) {
  const { fill = C.ink, size = 15, weight = 400, anchor = "start", op = 1, ls = 0 } = o;
  if (op <= 0.004) return "";
  return `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" fill="${fill}" font-size="${size}" font-weight="${weight}" text-anchor="${anchor}" opacity="${op.toFixed(3)}" letter-spacing="${ls}">${esc(s)}</text>`;
}
function R(x, y, w, h, o = {}) {
  const { fill = C.card, stroke = null, r = 10, op = 1, sw = 1 } = o;
  if (op <= 0.004 || w <= 0.2 || h <= 0.2) return "";
  const st = stroke ? ` stroke="${stroke}" stroke-width="${sw}"` : "";
  return `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${w.toFixed(1)}" height="${h.toFixed(1)}" rx="${r}" fill="${fill}"${st} opacity="${op.toFixed(3)}"/>`;
}
const DOT = (x, y, r, fill, op = 1) =>
  op <= 0.004 || r <= 0.05 ? "" : `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r.toFixed(2)}" fill="${fill}" opacity="${op.toFixed(3)}"/>`;
const LINE = (x1, y1, x2, y2, stroke, op = 1, sw = 1.5, dash = "") =>
  op <= 0.004 ? "" : `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="${stroke}" stroke-width="${sw}" opacity="${op.toFixed(3)}"${dash ? ` stroke-dasharray="${dash}"` : ""}/>`;
// Quadratic bezier point, for arcs that read as motion rather than teleporting.
const qbez = (p0, p1, p2, t) => ({
  x: (1 - t) * (1 - t) * p0.x + 2 * (1 - t) * t * p1.x + t * t * p2.x,
  y: (1 - t) * (1 - t) * p0.y + 2 * (1 - t) * t * p1.y + t * t * p2.y,
});

// ------------------------------------------------------------------ the window
// One Obsidian window holds every scene, so the viewer never loses the context of
// where this actually happens.
const WIN = { x: 40, y: 74, w: 1120, h: 508, ribbon: 44, tree: 158, panel: 316 };
WIN.ed = { x: WIN.x + WIN.ribbon + WIN.tree, y: WIN.y + 38 };
WIN.ed.w = WIN.w - WIN.ribbon - WIN.tree - WIN.panel;
WIN.ed.h = WIN.h - 38;
WIN.pn = { x: WIN.x + WIN.w - WIN.panel, y: WIN.y + 38, w: WIN.panel, h: WIN.h - 38 };

const TREE = ["Concepts", "Backpropagation", "Gradient Descent", "Loss Function", "Ideen", "Backprop als Fee…", "Rekursion in Ges…", "Daily", "2026-02-07"];
const BODY = [
  [0.94, 1], [0.86, 1], [0.55, 1], [0, 0],
  [0.91, 1], [0.72, 1], [0.88, 1], [0.42, 1],
];

function windowChrome(op = 1, o = {}) {
  const { title = "Backprop als Feedback beim Schreiben", activeTree = 5 } = o;
  let s = R(WIN.x, WIN.y, WIN.w, WIN.h, { fill: C.bg, stroke: C.edge, r: 12, op });
  // title bar + traffic lights
  s += LINE(WIN.x, WIN.y + 38, WIN.x + WIN.w, WIN.y + 38, C.edge, op, 1);
  for (let i = 0; i < 3; i++) s += DOT(WIN.x + 20 + i * 17, WIN.y + 19, 5, [C.rose, C.gold, C.green][i], op * 0.85);
  s += T(WIN.x + WIN.w / 2, WIN.y + 24, title, { fill: C.mut, size: 12.5, anchor: "middle", op: op * 0.9 });
  // ribbon
  s += LINE(WIN.x + WIN.ribbon, WIN.y + 38, WIN.x + WIN.ribbon, WIN.y + WIN.h, C.edge, op, 1);
  for (let i = 0; i < 6; i++) {
    const active = i === 3;
    s += R(WIN.x + 13, WIN.y + 58 + i * 32, 18, 18, { fill: active ? C.peri : C.faint, r: 5, op: op * (active ? 0.95 : 0.5) });
  }
  // file tree
  s += LINE(WIN.x + WIN.ribbon + WIN.tree, WIN.y + 38, WIN.x + WIN.ribbon + WIN.tree, WIN.y + WIN.h, C.edge, op, 1);
  for (let i = 0; i < TREE.length; i++) {
    const isFolder = [0, 4, 7].includes(i);
    const y = WIN.y + 62 + i * 25;
    if (i === activeTree) s += R(WIN.x + WIN.ribbon + 8, y - 13, WIN.tree - 16, 22, { fill: C.inner, r: 5, op: op * 0.9 });
    s += T(WIN.x + WIN.ribbon + (isFolder ? 16 : 26), y + 3, TREE[i], {
      fill: isFolder ? C.mut : i === activeTree ? C.body : C.dim, size: 11.5, weight: isFolder ? 600 : 400, op: op * 0.95,
    });
  }
  return s;
}

// The note body as typographic bars, so the eye reads "a note" without reading words.
function editorBody(op, o = {}) {
  const { reveal = 1, dim = 1 } = o;
  const x = WIN.ed.x + 34;
  let s = T(x, WIN.y + 84, "Backprop als Feedback", { size: 21, weight: 600, op: op * dim });
  let y = WIN.y + 118;
  for (let i = 0; i < BODY.length; i++) {
    const [wf] = BODY[i];
    if (wf === 0) { y += 14; continue; }
    const k = cl((reveal - i * 0.06) / 0.2);
    s += R(x, y, (WIN.ed.w - 90) * wf * k, 7, { fill: C.faint, r: 3.5, op: op * 0.75 * dim });
    y += 19;
  }
  return s;
}

function panelShell(op, o = {}) {
  const { sub = "Based on Backprop als Feedback" } = o;
  let s = LINE(WIN.pn.x, WIN.y + 38, WIN.pn.x, WIN.y + WIN.h, C.edge, op, 1);
  s += T(WIN.pn.x + 20, WIN.y + 68, "Smart related notes", { size: 13.5, weight: 600, op });
  s += T(WIN.pn.x + 20, WIN.y + 86, sub, { fill: C.dim, size: 11, op: op * 0.9 });
  return s;
}

const CARDS = [
  { t: "Backpropagation",          pct: 49, pills: [["Linked", "rec"]] },
  { t: "Gradient Descent",         pct: 35, pills: [["Related", "rel"]] },
  { t: "Machine Learning MOC",     pct: 32, pills: [["via Backpropagation", "rec"]] },
  { t: "Recurrent Neural Network", pct: 16, pills: [["via Backpropagation", "rec"]] },
];
const pillW = (s) => Math.round(s.length * 6.9) + 20;
function pill(x, y, label, kind, op) {
  const w = pillW(label);
  const st = kind === "rec" ? [C.recFill, C.recEdge, C.recInk] : [C.inner, C.innerEdge, C.mut];
  return R(x, y, w, 19, { fill: st[0], stroke: st[1], r: 6, op }) + T(x + w / 2, y + 13.5, label, { fill: st[2], size: 11, anchor: "middle", op });
}
function card(i, k, op) {
  if (k <= 0.004) return "";
  const c = CARDS[i];
  const y = WIN.y + 104 + i * 74 + (1 - k) * 14;
  const x = WIN.pn.x + 14, w = WIN.pn.w - 28;
  let s = R(x, y, w, 62, { fill: C.inner, stroke: C.innerEdge, r: 9, op });
  s += T(x + 16, y + 24, c.t, { fill: C.body, size: 13.5, weight: 500, op });
  s += R(x + w - 56, y + 11, 42, 20, { fill: c.pct >= 30 ? C.peri : C.innerEdge, r: 10, op });
  s += T(x + w - 35, y + 25, `${c.pct}%`, { fill: "#15151b", size: 11.5, weight: 600, anchor: "middle", op });
  s += pill(x + 16, y + 34, c.pills[0][0], c.pills[0][1], op);
  return s;
}

const cap = (s, op, sub) =>
  T(W / 2, 42, s, { size: 23, weight: 600, anchor: "middle", ls: -0.3, op })
  + (sub ? T(W / 2, 62, sub, { fill: C.dim, size: 13, anchor: "middle", op: op * 0.9 }) : "");

// ------------------------------------------------------------------ scenes
const SCENES = [
  { id: "open",    dur: 1.5 },
  { id: "embed",   dur: 4.6 },
  { id: "rank",    dur: 3.3 },
  { id: "graph",   dur: 4.1 },
  { id: "decrowd", dur: 2.9 },
  { id: "map",     dur: 3.4 },
  { id: "outro",   dur: 1.6 },
];
const TOTAL = SCENES.reduce((a, s) => a + s.dur, 0);
const OUTF = (t, d, f = 0.3) => (t > d - f ? 1 - ease((t - (d - f)) / f) : 1);

// 1. the window opens and a note is there. no claims yet.
function sOpen(t, d) {
  const a = ease(t / 0.55), o = OUTF(t, d);
  const A = a * o;
  let s = cap("Open a note.", A * ease((t - 0.5) / 0.5));
  s += windowChrome(A);
  s += editorBody(A, { reveal: ease((t - 0.35) / 0.9) });
  s += panelShell(A * ease((t - 0.7) / 0.5));
  return s;
}

// 2. the fundamental step: words leave the note, a model turns them into ONE point,
// and a note written in the other language lands next to it.
const WORDS = ["gradient", "loss", "chain rule", "backprop"];
const DEWORDS = ["Fehler", "rückwärts", "Kettenregel"];
function sEmbed(t, d) {
  const o = OUTF(t, d);
  let s = cap("Each note becomes one point, by meaning.", o * ease(t / 0.4), "German and English land in the same place.");
  s += windowChrome(o, {});
  s += editorBody(o, { reveal: 1, dim: L(1, 0.45, ease((t - 0.9) / 0.8)) });
  s += panelShell(o * 0.5);

  // the field the points land in, overlaid on the panel side
  const F = { x: WIN.pn.x + 22, y: WIN.y + 108, w: WIN.pn.w - 44, h: 300 };
  const fk = ease((t - 0.5) / 0.5) * o;
  s += R(F.x, F.y, F.w, F.h, { fill: C.card, stroke: C.cardEdge, r: 10, op: fk });
  s += T(F.x + F.w / 2, F.y + F.h + 22, "meaning space", { fill: C.dim, size: 11.5, anchor: "middle", op: fk });
  { let sd = 5; const rn = () => ((sd = (sd * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
    for (let i = 0; i < 70; i++) s += DOT(F.x + 12 + rn() * (F.w - 24), F.y + 12 + rn() * (F.h - 24), 2.2, C.faint, fk * 0.8); }

  const model = { x: WIN.ed.x + WIN.ed.w - 6, y: WIN.y + 250 };
  // word chips fly from the note into the model, then out as a single point
  const target = { x: F.x + F.w * 0.42, y: F.y + F.h * 0.40 };
  const de = { x: F.x + F.w * 0.52, y: F.y + F.h * 0.47 };

  for (let i = 0; i < WORDS.length; i++) {
    const st = 0.75 + i * 0.13;
    const k = ease((t - st) / 0.7);
    if (k <= 0.004 || k >= 1) continue;
    const from = { x: WIN.ed.x + 60 + (i % 2) * 130, y: WIN.y + 140 + i * 34 };
    const p = qbez(from, { x: (from.x + model.x) / 2, y: from.y - 46 }, model, k);
    const w = pillW(WORDS[i]);
    const op = (1 - k * k) * o;
    s += R(p.x - w / 2, p.y - 10, w, 20, { fill: C.inner, stroke: C.innerEdge, r: 6, op })
      + T(p.x, p.y + 4, WORDS[i], { fill: C.body, size: 11, anchor: "middle", op });
  }
  // the model itself
  const mk = ease((t - 0.7) / 0.4) * o;
  s += R(model.x - 46, model.y - 26, 92, 52, { fill: C.inner, stroke: C.peri, r: 10, op: mk * 0.95 });
  const pulse = 0.55 + 0.45 * Math.abs(Math.sin(t * 6));
  s += T(model.x, model.y - 2, "embed", { fill: C.peri, size: 12.5, weight: 600, anchor: "middle", op: mk * pulse });
  s += T(model.x, model.y + 15, "on your Mac", { fill: C.dim, size: 9.5, anchor: "middle", op: mk * 0.9 });

  // english point flies out of the model and lands
  const e1 = ease((t - 1.55) / 0.65);
  if (e1 > 0.004) {
    const p = qbez(model, { x: (model.x + target.x) / 2, y: model.y - 40 }, target, e1);
    s += DOT(p.x, p.y, L(9, 6, e1), C.peri, o);
    if (e1 >= 1) s += DOT(target.x, target.y, 6 + 10 * (1 - ease((t - 2.2) / 0.4)), C.peri, o * 0.25 * (1 - ease((t - 2.2) / 0.4)));
  }
  // then the german note does the same and lands right beside it
  const gk = ease((t - 2.35) / 0.4) * o;
  if (gk > 0.004) {
    s += R(WIN.ed.x + 40, WIN.y + 330, 210, 74, { fill: C.inner, stroke: C.innerEdge, r: 9, op: gk });
    s += T(WIN.ed.x + 56, WIN.y + 354, "2026-02-07", { fill: C.body, size: 12.5, weight: 600, op: gk });
    for (let i = 0; i < DEWORDS.length; i++) {
      const st = 2.6 + i * 0.1, k = ease((t - st) / 0.6);
      if (k <= 0.004 || k >= 1) continue;
      const from = { x: WIN.ed.x + 66 + i * 52, y: WIN.y + 378 };
      const p = qbez(from, { x: (from.x + model.x) / 2, y: from.y - 40 }, model, k);
      const w = pillW(DEWORDS[i]);
      s += R(p.x - w / 2, p.y - 9, w, 18, { fill: C.inner, stroke: C.gold, r: 6, op: (1 - k * k) * o })
        + T(p.x, p.y + 4, DEWORDS[i], { fill: C.gold, size: 10.5, anchor: "middle", op: (1 - k * k) * o });
    }
  }
  const e2 = ease((t - 3.3) / 0.6);
  if (e2 > 0.004) {
    const p = qbez(model, { x: (model.x + de.x) / 2, y: model.y - 34 }, de, e2);
    s += DOT(p.x, p.y, L(9, 6, e2), C.gold, o);
  }
  // the punchline: they are neighbours
  const nk = ease((t - 3.95) / 0.45) * o;
  if (nk > 0.004) {
    s += LINE(target.x, target.y, de.x, de.y, C.green, nk * 0.9, 2);
    s += T((target.x + de.x) / 2 + 34, (target.y + de.y) / 2 + 4, "same idea", { fill: C.recInk, size: 11, op: nk });
  }
  return s;
}

// 3. geometry becomes the panel: nearest points light up, then become cards.
function sRank(t, d) {
  const o = OUTF(t, d);
  let s = cap("The nearest points are the related notes.", o * ease(t / 0.35));
  s += windowChrome(o);
  s += panelShell(o);

  const F = { x: WIN.ed.x + 26, y: WIN.y + 70, w: WIN.ed.w - 52, h: WIN.h - 108 };
  const fade = o;
  const me = { x: F.x + F.w * 0.42, y: F.y + F.h * 0.40 };
  if (fade > 0.004) {
    s += R(F.x, F.y, F.w, F.h, { fill: C.card, stroke: C.cardEdge, r: 10, op: fade * o });
    // a cloud of other notes, with the four nearest picked out
    const near = [[0.56, 0.44], [0.34, 0.30], [0.62, 0.62], [0.30, 0.58]];
    let seed = 7;
    const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
    for (let i = 0; i < 150; i++) {
      const px = F.x + 14 + rnd() * (F.w - 28), py = F.y + 14 + rnd() * (F.h - 28);
      s += DOT(px, py, 2.4, C.faint, fade * 0.8);
    }
    const rk = ease((t - 0.3) / 0.7);
    for (let i = 0; i < near.length; i++) {
      const p = { x: F.x + F.w * near[i][0], y: F.y + F.h * near[i][1] };
      const k = ease((t - 0.35 - i * 0.1) / 0.5);
      s += LINE(me.x, me.y, L(me.x, p.x, k), L(me.y, p.y, k), C.peri, fade * o * 0.55, 1.2);
      s += DOT(p.x, p.y, 5 * k, i === 0 ? C.gold : C.peri, fade * o);
    }
    // ring pulse on the active note
    const pr = ((t * 0.9) % 1);
    s += `<circle cx="${me.x.toFixed(1)}" cy="${me.y.toFixed(1)}" r="${(8 + pr * 26).toFixed(1)}" fill="none" stroke="${C.peri}" stroke-width="1.5" opacity="${(fade * o * 0.4 * (1 - pr)).toFixed(3)}"/>`;
    s += DOT(me.x, me.y, 7, C.peri, fade * o);
    void rk;
  }
  for (let i = 0; i < CARDS.length; i++) s += card(i, ease((t - 1.7 - i * 0.14) / 0.5), o);
  return s;
}

// 4. the link graph: two notes with nothing in common, joined through a third.
function sGraph(t, d) {
  const o = OUTF(t, d);
  let s = cap("Your links see what wording cannot.", o * ease(t / 0.35), "recall 0.66 → 0.75 on held-out links");
  s += windowChrome(o);
  s += panelShell(o);
  s += editorBody(o * (1 - ease((t - 0.2) / 0.4)), { reveal: 1, dim: 0.4 });

  const G = { x: WIN.ed.x + 26, y: WIN.y + 70, w: WIN.ed.w - 52, h: WIN.h - 108 };
  const gk = ease((t - 0.3) / 0.5) * o;
  s += R(G.x, G.y, G.w, G.h, { fill: C.card, stroke: C.cardEdge, r: 10, op: gk });

  const A = { x: G.x + G.w * 0.22, y: G.y + G.h * 0.28 };
  const B = { x: G.x + G.w * 0.78, y: G.y + G.h * 0.30 };
  const M = { x: G.x + G.w * 0.50, y: G.y + G.h * 0.72 };
  const node = (p, label, sub, k, tone) => {
    if (k <= 0.004) return "";
    const w = 150, h = 46, r = back(k);
    return R(p.x - w / 2 * r, p.y - h / 2 * r, w * r, h * r, { fill: C.inner, stroke: tone, r: 9, op: k * o })
      + T(p.x, p.y - 2, label, { fill: C.body, size: 12.5, weight: 600, anchor: "middle", op: k * o })
      + T(p.x, p.y + 14, sub, { fill: C.dim, size: 10, anchor: "middle", op: k * o });
  };
  s += node(A, "Entropy", "information theory", ease((t - 0.55) / 0.4), C.innerEdge);
  s += node(B, "Hash Table", "data structures", ease((t - 0.7) / 0.4), C.innerEdge);

  // no shared wording
  const xk = ease((t - 1.15) / 0.4) * o;
  if (xk > 0.004) {
    s += LINE(A.x + 78, A.y, B.x - 78, B.y, C.innerEdge, xk * 0.9, 1.5, "5 5");
    const mx = (A.x + B.x) / 2, my = (A.y + B.y) / 2;
    s += LINE(mx - 8, my - 8, mx + 8, my + 8, C.rose, xk, 2.2);
    s += LINE(mx + 8, my - 8, mx - 8, my + 8, C.rose, xk, 2.2);
    s += T(mx, my - 20, "no words in common", { fill: C.dim, size: 11, anchor: "middle", op: xk });
  }
  // the bridge you built
  s += node(M, "Hash Function", "you linked both", ease((t - 1.75) / 0.4), C.recEdge);
  const ek = ease((t - 2.1) / 0.6);
  if (ek > 0.004) {
    s += LINE(M.x - 52, M.y - 20, L(M.x - 52, A.x, ek), L(M.y - 20, A.y + 24, ek), C.green, o * 0.95, 2.2);
    s += LINE(M.x + 52, M.y - 20, L(M.x + 52, B.x, ek), L(M.y - 20, B.y + 24, ek), C.green, o * 0.95, 2.2);
  }
  // the receipt lands in the panel
  const ck = ease((t - 2.8) / 0.5);
  if (ck > 0.004) {
    const x = WIN.pn.x + 14, w = WIN.pn.w - 28, y = WIN.y + 104 + (1 - ck) * 14;
    s += R(x, y, w, 62, { fill: C.inner, stroke: C.recEdge, r: 9, op: ck * o });
    s += T(x + 16, y + 24, "Hash Table", { fill: C.body, size: 13.5, weight: 500, op: ck * o });
    s += R(x + w - 56, y + 11, 42, 20, { fill: C.innerEdge, r: 10, op: ck * o });
    s += T(x + w - 35, y + 25, "12%", { fill: C.body, size: 11.5, weight: 600, anchor: "middle", op: ck * o });
    s += pill(x + 16, y + 34, "via Hash Function", "rec", ck * o);
  }
  return s;
}

// 5. de-crowding, shown as geometry: identical template notes pile onto one spot,
// then the shared direction is removed and they spread out.
function sDecrowd(t, d) {
  const o = OUTF(t, d);
  let s = cap("Notes from one template stop hiding each other.", o * ease(t / 0.35));
  s += windowChrome(o);
  s += panelShell(o * 0.4);
  const G = { x: WIN.ed.x + 26, y: WIN.y + 70, w: WIN.ed.w - 52, h: WIN.h - 108 };
  s += R(G.x, G.y, G.w, G.h, { fill: C.card, stroke: C.cardEdge, r: 10, op: o });

  const cxp = G.x + G.w * 0.5, cyp = G.y + G.h * 0.46;
  const k = easeIO(cl((t - 0.85) / 1.15));
  let seed = 31;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
  for (let i = 0; i < 22; i++) {
    const ang = rnd() * Math.PI * 2, rad = 46 + rnd() * 150;
    // clumped at the centre before, spread over the field after
    const x = L(cxp + Math.cos(ang) * 7, cxp + Math.cos(ang) * rad, k);
    const y = L(cyp + Math.sin(ang) * 6, cyp + Math.sin(ang) * rad * 0.6, k);
    s += DOT(x, y, 7, C.recInk, o * L(0.4, 0.95, k));
  }
  // the direction being subtracted
  const ak = ease((t - 0.85) / 0.5) * (1 - ease((t - 1.9) / 0.4)) * o;
  if (ak > 0.004) {
    s += LINE(cxp - 96, cyp + 96, cxp + 96, cyp - 96, C.rose, ak, 2, "6 5");
    s += T(cxp + 112, cyp - 100, "shared template direction, removed", { fill: C.rose, size: 11, anchor: "end", op: ak });
  }
  const val = L(8.2, 3.4, k);
  s += T(G.x + G.w / 2, G.y + G.h - 22, `${val.toFixed(1)} of a daily note's top 10 were other daily notes`, {
    fill: k > 0.5 ? C.recInk : C.mut, size: 13, anchor: "middle", op: o,
  });
  return s;
}

// 6. pull back: the whole vault, real points, clustered and named.
function sMap(t, d, map) {
  const o = OUTF(t, d);
  let s = cap("Then step back and see the whole vault.", o * ease(t / 0.35));
  s += windowChrome(o);
  s += panelShell(o * 0.35);
  const G = { x: WIN.ed.x + 26, y: WIN.y + 70, w: WIN.ed.w - 52, h: WIN.h - 108 };
  s += R(G.x, G.y, G.w, G.h, { fill: C.card, stroke: C.cardEdge, r: 10, op: o });
  const COL = [C.green, C.peri, C.gold, C.cyan, C.grey, C.rose];
  const { pts, xr, yr } = map;
  const sx = (x) => G.x + 16 + (x - xr[0]) / (xr[1] - xr[0]) * (G.w - 32);
  const sy = (y) => G.y + 16 + (1 - (y - yr[0]) / (yr[1] - yr[0])) * (G.h - 32);
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    // points arrive cluster by cluster, so the grouping assembles itself
    const k = ease((t - 0.35 - p.k * 0.13 - (i % 30) * 0.003) / 0.4);
    if (k <= 0.004) continue;
    s += DOT(sx(p.x), sy(p.y), 2.7 * k, COL[p.k % 6], 0.72 * k * o);
  }
  // names appear in the panel as each cluster finishes
  let ly = WIN.y + 112;
  for (let li = 0; li < map.legend.length; li++) {
    const l = map.legend[li];
    const k = ease((t - 0.7 - li * 0.12) / 0.4) * o;
    s += DOT(WIN.pn.x + 26, ly - 4, 4.5, COL[l.k % 6], k);
    s += T(WIN.pn.x + 40, ly, l.name, { fill: C.body, size: 12.5, op: k });
    s += T(WIN.pn.x + 40, ly + 15, `${l.n} notes`, { fill: C.dim, size: 10.5, op: k });
    ly += 40;
  }
  return s;
}

function sOutro(t, d) {
  const a = ease(t / 0.45), o = OUTF(t, d, 0.35);
  const A = a * o;
  let s = T(W / 2 - 34, 268, "Smart Related Notes", { size: 42, weight: 600, anchor: "middle", ls: -1, op: A });
  s += T(W / 2 + 236, 268, "3.0", { size: 42, weight: 600, anchor: "middle", fill: C.green, op: A });
  const b = ease((t - 0.35) / 0.45) * o;
  s += T(W / 2, 308, "Community plugins  ›  Browse  ›  Smart Related Notes", { fill: C.mut, size: 16, anchor: "middle", op: b });
  const c = ease((t - 0.6) / 0.45) * o;
  s += R(W / 2 - 176, 336, 352, 40, { fill: C.recFill, stroke: C.recEdge, r: 10, op: c });
  s += T(W / 2, 362, "Local, offline, and measured before it ships", { fill: C.recInk, size: 14, anchor: "middle", op: c });
  return s;
}

// ------------------------------------------------------------------ driver
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
  for (let f = 0; f < Math.round(sc.dur * FPS); f++) {
    const t = f / FPS;
    let b = "";
    if (sc.id === "open") b = sOpen(t, sc.dur);
    else if (sc.id === "embed") b = sEmbed(t, sc.dur);
    else if (sc.id === "rank") b = sRank(t, sc.dur);
    else if (sc.id === "graph") b = sGraph(t, sc.dur);
    else if (sc.id === "decrowd") b = sDecrowd(t, sc.dur);
    else if (sc.id === "map") b = sMap(t, sc.dur, map);
    else if (sc.id === "outro") b = sOutro(t, sc.dur);
    b += R(0, H - 3, W * (n / total), 3, { fill: C.green, r: 0, op: 0.7 });
    writeFileSync(join(OUT, `f${String(n).padStart(5, "0")}.svg`),
      `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" font-family="${FONT}"><rect width="${W}" height="${H}" fill="${C.page}"/>${b}</svg>`);
    n++;
  }
}
console.log(`${n} frames, ${TOTAL.toFixed(1)}s at ${FPS}fps -> ${OUT}`);
