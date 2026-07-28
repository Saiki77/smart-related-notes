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
// Two things shape it. It alternates between the app and the idea: a scene inside
// an Obsidian window, then the same step drawn full-bleed as geometry, so the
// viewer sees both what it looks like and why it works. And every cut is a blurred
// crossfade composited here, not a hard switch, so scenes dissolve into each other
// rather than reading as slides.
import { writeFileSync, readFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";

const OUT = process.argv[2];
const MAPDATA = process.argv[3];
mkdirSync(OUT, { recursive: true });

const W = 1200, H = 620, FPS = 25;
const XF = 0.6;        // crossfade length, seconds
const BLUR = 7;        // peak blur during a cut, px

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
const ease = (t) => 1 - Math.pow(1 - cl(t), 3);
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
const RING = (x, y, r, stroke, op, sw = 1.5) =>
  op <= 0.004 ? "" : `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r.toFixed(1)}" fill="none" stroke="${stroke}" stroke-width="${sw}" opacity="${op.toFixed(3)}"/>`;
const LINE = (x1, y1, x2, y2, stroke, op = 1, sw = 1.5, dash = "") =>
  op <= 0.004 ? "" : `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="${stroke}" stroke-width="${sw}" opacity="${op.toFixed(3)}"${dash ? ` stroke-dasharray="${dash}"` : ""}/>`;
const qbez = (p0, p1, p2, t) => ({
  x: (1 - t) * (1 - t) * p0.x + 2 * (1 - t) * t * p1.x + t * t * p2.x,
  y: (1 - t) * (1 - t) * p0.y + 2 * (1 - t) * t * p1.y + t * t * p2.y,
});
function rng(seed) { let s = seed; return () => ((s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff); }

// ------------------------------------------------------------------ the window
const WIN = { x: 40, y: 66, w: 1120, h: 500, ribbon: 44, tree: 158, panel: 316 };
WIN.ed = { x: WIN.x + WIN.ribbon + WIN.tree };
WIN.ed.w = WIN.w - WIN.ribbon - WIN.tree - WIN.panel;
WIN.pn = { x: WIN.x + WIN.w - WIN.panel, w: WIN.panel };

const TREE = ["Concepts", "Backpropagation", "Gradient Descent", "Loss Function", "Ideen", "Backprop als Fee…", "Rekursion in Ges…", "Daily", "2026-02-07"];
const BODY = [0.94, 0.86, 0.55, 0, 0.91, 0.72, 0.88, 0.42];

function windowChrome(o = {}) {
  const { title = "Backprop als Feedback beim Schreiben", activeTree = 5, op = 1 } = o;
  let s = R(WIN.x, WIN.y, WIN.w, WIN.h, { fill: C.bg, stroke: C.edge, r: 12, op });
  s += LINE(WIN.x, WIN.y + 38, WIN.x + WIN.w, WIN.y + 38, C.edge, op);
  for (let i = 0; i < 3; i++) s += DOT(WIN.x + 20 + i * 17, WIN.y + 19, 5, [C.rose, C.gold, C.green][i], op * 0.85);
  s += T(WIN.x + WIN.w / 2, WIN.y + 24, title, { fill: C.mut, size: 12.5, anchor: "middle", op: op * 0.9 });
  s += LINE(WIN.x + WIN.ribbon, WIN.y + 38, WIN.x + WIN.ribbon, WIN.y + WIN.h, C.edge, op);
  for (let i = 0; i < 6; i++)
    s += R(WIN.x + 13, WIN.y + 58 + i * 32, 18, 18, { fill: i === 3 ? C.peri : C.faint, r: 5, op: op * (i === 3 ? 0.95 : 0.5) });
  s += LINE(WIN.x + WIN.ribbon + WIN.tree, WIN.y + 38, WIN.x + WIN.ribbon + WIN.tree, WIN.y + WIN.h, C.edge, op);
  for (let i = 0; i < TREE.length; i++) {
    const folder = [0, 4, 7].includes(i), y = WIN.y + 62 + i * 25;
    if (i === activeTree) s += R(WIN.x + WIN.ribbon + 8, y - 13, WIN.tree - 16, 22, { fill: C.inner, r: 5, op: op * 0.9 });
    s += T(WIN.x + WIN.ribbon + (folder ? 16 : 26), y + 3, TREE[i], {
      fill: folder ? C.mut : i === activeTree ? C.body : C.dim, size: 11.5, weight: folder ? 600 : 400, op: op * 0.95,
    });
  }
  return s;
}
function editorBody(o = {}) {
  const { reveal = 1, dim = 1, op = 1, link = 0 } = o;
  const x = WIN.ed.x + 34;
  let s = T(x, WIN.y + 84, "Backprop als Feedback", { size: 21, weight: 600, op: op * dim });
  let y = WIN.y + 118;
  for (let i = 0; i < BODY.length; i++) {
    if (BODY[i] === 0) { y += 14; continue; }
    const k = cl((reveal - i * 0.06) / 0.2);
    s += R(x, y, (WIN.ed.w - 90) * BODY[i] * k, 7, { fill: C.faint, r: 3.5, op: op * 0.75 * dim });
    // the inline link suggestion: one run of the line lights up and gets underlined
    if (i === 5 && link > 0.004) {
      const lw = 116 * ease(link);
      s += R(x + 62, y, lw, 7, { fill: C.peri, r: 3.5, op: op * 0.9 * ease(link) });
      s += LINE(x + 62, y + 13, x + 62 + lw, y + 13, C.peri, op * ease(link), 1.6);
    }
    y += 19;
  }
  return s;
}
function panelShell(op = 1, sub = "Based on Backprop als Feedback") {
  return LINE(WIN.pn.x, WIN.y + 38, WIN.pn.x, WIN.y + WIN.h, C.edge, op)
    + T(WIN.pn.x + 20, WIN.y + 68, "Smart related notes", { size: 13.5, weight: 600, op })
    + T(WIN.pn.x + 20, WIN.y + 86, sub, { fill: C.dim, size: 11, op: op * 0.9 });
}
const CARDS = [
  { t: "Backpropagation", pct: 49, pill: ["Linked", "rec"] },
  { t: "Gradient Descent", pct: 35, pill: ["Related", "rel"] },
  { t: "Machine Learning MOC", pct: 32, pill: ["via Backpropagation", "rec"] },
  { t: "Recurrent Neural Network", pct: 16, pill: ["via Backpropagation", "rec"] },
];
const pillW = (s) => Math.round(s.length * 6.9) + 20;
function pill(x, y, label, kind, op) {
  const w = pillW(label);
  const st = kind === "rec" ? [C.recFill, C.recEdge, C.recInk] : [C.inner, C.innerEdge, C.mut];
  return R(x, y, w, 19, { fill: st[0], stroke: st[1], r: 6, op }) + T(x + w / 2, y + 13.5, label, { fill: st[2], size: 11, anchor: "middle", op });
}
function card(i, k, op = 1, hi = false) {
  if (k <= 0.004) return "";
  const c = CARDS[i];
  const y = WIN.y + 104 + i * 74 + (1 - k) * 14;
  const x = WIN.pn.x + 14, w = WIN.pn.w - 28;
  let s = R(x, y, w, 62, { fill: C.inner, stroke: hi ? C.recEdge : C.innerEdge, r: 9, op, sw: hi ? 1.6 : 1 });
  s += T(x + 16, y + 24, c.t, { fill: C.body, size: 13.5, weight: 500, op });
  s += R(x + w - 56, y + 11, 42, 20, { fill: c.pct >= 30 ? C.peri : C.innerEdge, r: 10, op });
  s += T(x + w - 35, y + 25, `${c.pct}%`, { fill: "#15151b", size: 11.5, weight: 600, anchor: "middle", op });
  s += pill(x + 16, y + 34, c.pill[0], c.pill[1], op);
  return s;
}
// Captions sit in the same place in every scene, app or concept.
const cap = (s, sub, op = 1) =>
  T(W / 2, 38, s, { size: 23, weight: 600, anchor: "middle", ls: -0.3, op })
  + (sub ? T(W / 2, 58, sub, { fill: C.dim, size: 13, anchor: "middle", op: op * 0.9 }) : "");

// ------------------------------------------------------------------ app scenes
function aOpen(t) {
  let s = cap("This is it, in Obsidian.", "A panel in the sidebar. Nothing else changes.", ease((t - 0.3) / 0.5));
  s += windowChrome();
  s += editorBody({ reveal: ease((t - 0.15) / 0.9) });
  s += panelShell(ease((t - 0.5) / 0.5));
  return s;
}
function aRank(t) {
  let s = cap("The sidebar ranks every other note.", "Real output, on a 494-note vault.");
  s += windowChrome();
  s += editorBody({ dim: 0.5 });
  s += panelShell();
  for (let i = 0; i < CARDS.length; i++) s += card(i, ease((t - 0.25 - i * 0.16) / 0.5));
  return s;
}
function aVia(t) {
  let s = cap("Click a glowing phrase to link it.", "And every card says why it is there.");
  s += windowChrome();
  s += editorBody({ dim: 0.75, link: ease((t - 0.3) / 0.6) });
  s += panelShell();
  for (let i = 0; i < CARDS.length; i++) s += card(i, 1, 1, i >= 2 && t > 1.0);
  const pk = ease((t - 0.9) / 0.5);
  if (pk > 0.004) {
    const px = WIN.ed.x + 150, py = WIN.y + 232;
    s += `<path d="M ${px} ${py} l 0 15 l 4 -4 l 3 7 l 3 -1 l -3 -7 l 6 0 Z" fill="${C.ink}" opacity="${(pk * 0.95).toFixed(3)}"/>`;
    const r = ((t - 0.9) % 0.9) / 0.9;
    s += RING(px, py, 6 + r * 22, C.peri, pk * 0.5 * (1 - r));
  }
  return s;
}
function aMap(t, map) {
  let s = cap("And a map of the whole vault, in a tab.", "Click a point to open it, a cluster to hide it.");
  s += windowChrome({ title: "Vault map", activeTree: -1 });
  s += panelShell(1, "Clusters");
  const G = { x: WIN.ed.x + 20, y: WIN.y + 54, w: WIN.ed.w - 40, h: WIN.h - 76 };
  s += R(G.x, G.y, G.w, G.h, { fill: C.card, stroke: C.cardEdge, r: 10 });
  const COL = [C.green, C.peri, C.gold, C.cyan, C.grey, C.rose];
  const { pts, xr, yr } = map;
  const sx = (x) => G.x + 14 + (x - xr[0]) / (xr[1] - xr[0]) * (G.w - 28);
  const sy = (y) => G.y + 14 + (1 - (y - yr[0]) / (yr[1] - yr[0])) * (G.h - 28);
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    const k = ease((t - 0.15 - p.k * 0.1 - (i % 30) * 0.003) / 0.4);
    if (k <= 0.004) continue;
    s += DOT(sx(p.x), sy(p.y), 2.7 * k, COL[p.k % 6], 0.74 * k);
  }
  let ly = WIN.y + 116;
  for (let li = 0; li < map.legend.length; li++) {
    const l = map.legend[li], k = ease((t - 0.5 - li * 0.1) / 0.4);
    s += DOT(WIN.pn.x + 26, ly - 4, 4.5, COL[l.k % 6], k);
    s += T(WIN.pn.x + 40, ly, l.name, { fill: C.body, size: 12.5, op: k });
    s += T(WIN.pn.x + 40, ly + 15, `${l.n} notes`, { fill: C.dim, size: 10.5, op: k });
    ly += 38;
  }
  return s;
}

// -------------------------------------------------------------- concept scenes
// Full bleed, no window. These answer "why does that work", using the whole frame.
const WORDS = ["gradient", "loss", "chain rule", "backprop"];
const DEWORDS = ["Fehler", "rückwärts", "Kettenregel"];
function cEmbed(t) {
  let s = cap("Underneath: each note becomes one point.", "Meaning decides where it lands, not the words.");
  const noteA = { x: 160, y: 210 }, noteB = { x: 160, y: 425 };
  const model = { x: 470, y: 315 };
  const F = { x: 640, y: 108, w: 520, h: 430 };
  s += R(F.x, F.y, F.w, F.h, { fill: C.card, stroke: C.cardEdge, r: 14 });
  s += T(F.x + F.w / 2, F.y + F.h + 26, "meaning space", { fill: C.dim, size: 12.5, anchor: "middle" });
  const rn = rng(5);
  for (let i = 0; i < 90; i++) s += DOT(F.x + 16 + rn() * (F.w - 32), F.y + 16 + rn() * (F.h - 32), 2.6, C.faint, 0.8);

  const noteCard = (p, title, lang, k) => k <= 0.004 ? "" :
    R(p.x - 112, p.y - 52, 224, 104, { fill: C.inner, stroke: C.innerEdge, r: 10, op: k })
    + T(p.x - 94, p.y - 24, title, { fill: C.body, size: 14, weight: 600, op: k })
    + T(p.x - 94, p.y - 4, lang, { fill: C.dim, size: 11, op: k });
  s += noteCard(noteA, "Backprop als Feedback", "written in English", ease(t / 0.4));
  s += noteCard(noteB, "2026-02-07", "written in German", ease((t - 2.1) / 0.4));

  const mk = ease((t - 0.2) / 0.4);
  s += R(model.x - 62, model.y - 32, 124, 64, { fill: C.inner, stroke: C.peri, r: 12, op: mk });
  s += T(model.x, model.y - 4, "embedding model", { fill: C.peri, size: 12, weight: 600, anchor: "middle", op: mk * (0.6 + 0.4 * Math.abs(Math.sin(t * 5))) });
  s += T(model.x, model.y + 15, "384 numbers, on your Mac", { fill: C.dim, size: 10, anchor: "middle", op: mk * 0.9 });

  const pA = { x: F.x + F.w * 0.44, y: F.y + F.h * 0.40 };
  const pB = { x: F.x + F.w * 0.53, y: F.y + F.h * 0.49 };
  const fly = (words, from, st, tone) => {
    let out = "";
    for (let i = 0; i < words.length; i++) {
      const k = ease((t - st - i * 0.11) / 0.65);
      if (k <= 0.004 || k >= 1) continue;
      const f = { x: from.x - 50 + (i % 2) * 74, y: from.y + 12 + i * 8 };
      const p = qbez(f, { x: (f.x + model.x) / 2, y: f.y - 70 }, model, k);
      const w = pillW(words[i]);
      out += R(p.x - w / 2, p.y - 10, w, 20, { fill: C.inner, stroke: tone, r: 6, op: 1 - k * k })
        + T(p.x, p.y + 4, words[i], { fill: tone, size: 11.5, anchor: "middle", op: 1 - k * k });
    }
    return out;
  };
  s += fly(WORDS, noteA, 0.5, C.body);
  s += fly(DEWORDS, noteB, 2.5, C.gold);

  const land = (target, st, tone) => {
    const k = ease((t - st) / 0.7);
    if (k <= 0.004) return "";
    const p = qbez(model, { x: (model.x + target.x) / 2, y: model.y - 60 }, target, k);
    let out = DOT(p.x, p.y, L(11, 7, k), tone);
    if (k >= 1) { const r = ease((t - st - 0.7) / 0.5); out += RING(target.x, target.y, 8 + r * 30, tone, 0.5 * (1 - r), 2); }
    return out;
  };
  s += land(pA, 1.35, C.peri);
  s += land(pB, 3.35, C.gold);

  const nk = ease((t - 4.05) / 0.45);
  if (nk > 0.004) {
    s += LINE(pA.x, pA.y, pB.x, pB.y, C.green, nk * 0.95, 2.4);
    s += T((pA.x + pB.x) / 2, (pA.y + pB.y) / 2 + 44, "different words, same idea", { fill: C.recInk, size: 13.5, anchor: "middle", op: nk });
  }
  return s;
}

function cGraph(t) {
  let s = cap("And your links reach where wording cannot.", "recall 0.66 → 0.75 on links held back from the model");
  const A = { x: 300, y: 260 }, B = { x: 900, y: 260 }, M = { x: 600, y: 452 };
  const node = (p, label, sub, k, tone) => {
    if (k <= 0.004) return "";
    const w = 210, h = 62, r = back(k);
    return R(p.x - w / 2 * r, p.y - h / 2 * r, w * r, h * r, { fill: C.inner, stroke: tone, r: 11, op: k })
      + T(p.x, p.y - 2, label, { fill: C.body, size: 15, weight: 600, anchor: "middle", op: k })
      + T(p.x, p.y + 18, sub, { fill: C.dim, size: 11, anchor: "middle", op: k });
  };
  s += node(A, "Entropy", "information theory", ease((t - 0.1) / 0.4), C.innerEdge);
  s += node(B, "Hash Table", "data structures", ease((t - 0.25) / 0.4), C.innerEdge);
  const xk = ease((t - 0.75) / 0.4);
  if (xk > 0.004) {
    s += LINE(A.x + 108, A.y, B.x - 108, B.y, C.innerEdge, xk * 0.9, 1.6, "6 6");
    const mx = (A.x + B.x) / 2;
    s += LINE(mx - 10, A.y - 10, mx + 10, A.y + 10, C.rose, xk, 2.4);
    s += LINE(mx + 10, A.y - 10, mx - 10, A.y + 10, C.rose, xk, 2.4);
    s += T(mx, A.y - 28, "no words in common", { fill: C.dim, size: 12.5, anchor: "middle", op: xk });
  }
  s += node(M, "Hash Function", "you linked both of these", ease((t - 1.35) / 0.4), C.recEdge);
  const ek = ease((t - 1.7) / 0.6);
  if (ek > 0.004) {
    s += LINE(M.x - 70, M.y - 28, L(M.x - 70, A.x, ek), L(M.y - 28, A.y + 32, ek), C.green, 0.95, 2.6);
    s += LINE(M.x + 70, M.y - 28, L(M.x + 70, B.x, ek), L(M.y - 28, B.y + 32, ek), C.green, 0.95, 2.6);
  }
  const rk = ease((t - 2.4) / 0.5);
  if (rk > 0.004) {
    s += LINE(A.x, A.y - 48, B.x, B.y - 48, C.recInk, rk * 0.85, 2.4);
    s += T((A.x + B.x) / 2, A.y - 60, "so these two are related", { fill: C.recInk, size: 14, anchor: "middle", op: rk });
  }
  return s;
}

function cDecrowd(t) {
  let s = cap("Notes from one template stop hiding each other.", "The shape they share is subtracted, the content is left alone.");
  const cx = W / 2, cy = 330;
  const k = easeIO(cl((t - 0.6) / 1.3));
  const rn = rng(31);
  for (let i = 0; i < 26; i++) {
    const ang = rn() * Math.PI * 2, rad = 70 + rn() * 210;
    const x = L(cx + Math.cos(ang) * 8, cx + Math.cos(ang) * rad, k);
    const y = L(cy + Math.sin(ang) * 7, cy + Math.sin(ang) * rad * 0.55, k);
    s += DOT(x, y, 8, C.recInk, L(0.4, 0.95, k));
  }
  const ak = ease((t - 0.6) / 0.45) * (1 - ease((t - 1.75) / 0.4));
  if (ak > 0.004) {
    s += LINE(cx - 150, cy + 130, cx + 150, cy - 130, C.rose, ak, 2.2, "7 6");
    s += T(cx + 168, cy - 136, "shared template direction, removed", { fill: C.rose, size: 12.5, op: ak });
  }
  s += T(cx, 545, `${L(8.2, 3.4, k).toFixed(1)} of a daily note's top 10 were other daily notes`, {
    fill: k > 0.5 ? C.recInk : C.mut, size: 15, anchor: "middle",
  });
  return s;
}

function cOutro(t) {
  const a = ease(t / 0.4);
  let s = T(W / 2 - 34, 274, "Smart Related Notes", { size: 42, weight: 600, anchor: "middle", ls: -1, op: a });
  s += T(W / 2 + 236, 274, "3.0", { size: 42, weight: 600, anchor: "middle", fill: C.green, op: a });
  s += T(W / 2, 314, "Community plugins  ›  Browse  ›  Smart Related Notes", { fill: C.mut, size: 16, anchor: "middle", op: ease((t - 0.3) / 0.4) });
  const c = ease((t - 0.55) / 0.4);
  s += R(W / 2 - 176, 342, 352, 40, { fill: C.recFill, stroke: C.recEdge, r: 10, op: c });
  s += T(W / 2, 368, "Local, offline, and measured before it ships", { fill: C.recInk, size: 14, anchor: "middle", op: c });
  return s;
}

// ------------------------------------------------------------------ timeline
// app and concept alternate, so neither the UI nor the theory runs on too long.
const SCENES = [
  { id: "aOpen",    dur: 2.2 },
  { id: "cEmbed",   dur: 5.0 },
  { id: "aRank",    dur: 3.0 },
  { id: "cGraph",   dur: 3.4 },
  { id: "aVia",     dur: 2.8 },
  { id: "cDecrowd", dur: 3.0 },
  { id: "aMap",     dur: 3.2 },
  { id: "cOutro",   dur: 2.0 },
];
const DRAW = { aOpen, aRank, aVia, aMap, cEmbed, cGraph, cDecrowd, cOutro };

// Scenes overlap by XF, so each cut is a dissolve rather than a switch.
let acc = 0;
for (const sc of SCENES) { sc.start = acc; acc += sc.dur - XF; }
const TOTAL = acc + XF;

let map = { pts: [], legend: [], xr: [-1, 1], yr: [-1, 1] };
if (MAPDATA) {
  const raw = JSON.parse(readFileSync(MAPDATA, "utf8"));
  const NAMES = { 0: "daily notes", 1: "machine learning", 2: "patterns / databases", 3: "linear algebra", 4: "drafts and stubs", 5: "algorithms / search" };
  const xs = raw.pts.map((p) => p.x), ys = raw.pts.map((p) => p.y);
  const counts = new Map();
  for (const p of raw.pts) counts.set(p.k, (counts.get(p.k) ?? 0) + 1);
  map = {
    pts: raw.pts, xr: [Math.min(...xs), Math.max(...xs)], yr: [Math.min(...ys), Math.max(...ys)],
    legend: [...counts.entries()].sort((a, b) => b[1] - a[1]).map(([k, n]) => ({ k, n, name: NAMES[k] ?? `cluster ${k + 1}` })),
  };
}

const total = Math.round(TOTAL * FPS);
for (let n = 0; n < total; n++) {
  const gt = n / FPS;
  const layers = [];
  for (let i = 0; i < SCENES.length; i++) {
    const sc = SCENES[i];
    const t = gt - sc.start;
    if (t < -0.001 || t > sc.dur) continue;
    // alpha and blur both come from position inside this scene's fade regions, so a
    // cut is two layers on screen at once, one sharpening and one dissolving.
    let a = 1, bl = 0;
    if (i > 0 && t < XF) { const u = cl(t / XF); a = easeIO(u); bl = BLUR * (1 - u); }
    if (i < SCENES.length - 1 && t > sc.dur - XF) {
      const u = cl((t - (sc.dur - XF)) / XF);
      a = Math.min(a, easeIO(1 - u)); bl = Math.max(bl, BLUR * u);
    }
    if (a <= 0.004) continue;
    layers.push({ i, body: DRAW[sc.id](t, map), a, bl });
  }
  let defs = "", g = "";
  for (const l of layers) {
    let f = "";
    if (l.bl > 0.05) {
      const id = `b${l.i}`;
      defs += `<filter id="${id}" x="-8%" y="-8%" width="116%" height="116%"><feGaussianBlur stdDeviation="${l.bl.toFixed(2)}"/></filter>`;
      f = ` filter="url(#${id})"`;
    }
    g += `<g opacity="${l.a.toFixed(3)}"${f}>${l.body}</g>`;
  }
  g += R(0, H - 3, W * (n / total), 3, { fill: C.green, r: 0, op: 0.7 });
  writeFileSync(join(OUT, `f${String(n).padStart(5, "0")}.svg`),
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" font-family="${FONT}">`
    + (defs ? `<defs>${defs}</defs>` : "")
    + `<rect width="${W}" height="${H}" fill="${C.page}"/>${g}</svg>`);
}
console.log(`${total} frames, ${TOTAL.toFixed(1)}s at ${FPS}fps, ${XF}s crossfades -> ${OUT}`);
