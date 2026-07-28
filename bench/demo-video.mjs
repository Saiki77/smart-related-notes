// Generates the README showcase video as SVG frames, rendered to PNG and muxed by
// ffmpeg. Not a screen recording: every frame is drawn, so the video is a diff and
// can be rebuilt when a number changes.
//
//   node bench/demo-video.mjs <frames-dir> [mapdata.json]
//
// It tells one story rather than listing features:
//
//   1 you wrote something relevant months ago and will not remember it
//   2 search cannot help, because it needs the words you happened to use
//   3 so every note is turned into a point, and near comes to mean related
//   4 which is why a sidebar can answer without being asked
//   5 but meaning alone misses a third of the links you actually make
//   6 so your own link graph is fused in, and each card shows its work
//   7 and the same points, stepped back from, are a map of the vault
//
// Two rules hold it together. Scenes alternate between the app and the idea, so
// the mechanism gets the whole frame instead of being mimed at UI scale. And every
// beat is given time to land: things appear, then rest, before the next beat.
//
// The content is real. Panel cards are the rankings this plugin returned for
// "Backprop als Feedback beim Schreiben" in the lab vault, and the points in
// scenes 3 and 7 are the same projected vectors, from bench/vault-map-figure.mjs.
import { writeFileSync, readFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";

const OUT = process.argv[2];
const MAPDATA = process.argv[3];
mkdirSync(OUT, { recursive: true });

const W = 1200, H = 620, FPS = 25;
const XF = 0.7;        // crossfade length, seconds
const BLUR = 9;        // peak blur during a cut, px

const C = {
  page: "#08080c", bg: "#16161d", edge: "#3d3d4c",
  card: "#20202a", cardEdge: "#3d3d4c",
  inner: "#2b2b38", innerEdge: "#505062",
  ink: "#ffffff", body: "#e6e9f0", mut: "#b0b7c6", dim: "#8d95a6", faint: "#565669",
  green: "#5fe3a3", recFill: "#183328", recEdge: "#4f9c74", recInk: "#9df3c6",
  peri: "#a6b0f5", gold: "#f7d68a", cyan: "#6fe0ea", grey: "#c3c7d4", rose: "#e09aab",
};
const FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif";

const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const cl = (t) => Math.max(0, Math.min(1, t));
const ease = (t) => 1 - Math.pow(1 - cl(t), 3);
const easeIO = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const back = (t) => { const c = 1.70158, u = cl(t) - 1; return 1 + (c + 1) * u * u * u + c * u * u; };
const L = (a, b, t) => a + (b - a) * cl(t);
// A caret that is always on is not a caret. 0.62s lit, 0.44s dark.
const blink = (t) => ((t % 1.06) < 0.62 ? 1 : 0);
// Appears, holds, leaves. The hold is the point: a beat that never rests cannot be read.
const hold = (t, inAt, outAt, f = 0.45) => cl(ease((t - inAt) / f)) * (outAt == null ? 1 : 1 - ease((t - outAt) / f));

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
  op <= 0.004 || r <= 0 ? "" : `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r.toFixed(1)}" fill="none" stroke="${stroke}" stroke-width="${sw}" opacity="${op.toFixed(3)}"/>`;
const LINE = (x1, y1, x2, y2, stroke, op = 1, sw = 1.5, dash = "") =>
  op <= 0.004 ? "" : `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="${stroke}" stroke-width="${sw}" opacity="${op.toFixed(3)}"${dash ? ` stroke-dasharray="${dash}"` : ""}/>`;
const qbez = (p0, p1, p2, t) => ({
  x: (1 - t) * (1 - t) * p0.x + 2 * (1 - t) * t * p1.x + t * t * p2.x,
  y: (1 - t) * (1 - t) * p0.y + 2 * (1 - t) * t * p1.y + t * t * p2.y,
});
function rng(seed) { let s = seed; return () => ((s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff); }

// Caption track. Lines cross-dissolve in place, so the top of the frame narrates
// continuously instead of one label sitting there for the whole scene.
const IN_T = 0.34, OUT_T = 0.28;
function says(t, beats) {
  let s = "";
  for (let i = 0; i < beats.length; i++) {
    const b = beats[i], next = beats[i + 1];
    // out completes exactly when in begins, so the two never share the baseline
    const inK = easeIO(cl((t - b.at) / IN_T));
    const outK = next ? easeIO(cl((t - (next.at - OUT_T)) / OUT_T)) : 0;
    const op = inK * (1 - outK);
    if (op <= 0.004) continue;
    const dy = (1 - inK) * 9 - outK * 9;
    s += T(W / 2, 40 + dy, b.head, { size: 31, weight: 600, anchor: "middle", ls: -0.4, op });
    if (b.sub) s += T(W / 2, 66 + dy, b.sub, { fill: C.mut, size: 16.5, anchor: "middle", op: op * 0.92 });
  }
  return s;
}

// ------------------------------------------------------------------ the window
const WIN = { x: 40, y: 84, w: 1120, h: 486, ribbon: 46, tree: 146, panel: 352 };
WIN.ed = { x: WIN.x + WIN.ribbon + WIN.tree };
WIN.ed.w = WIN.w - WIN.ribbon - WIN.tree - WIN.panel;
WIN.pn = { x: WIN.x + WIN.w - WIN.panel, w: WIN.panel };

const TREE = ["Concepts", "Backpropagation", "Gradient Descent", "Loss Function", "Ideas", "Backprop as feedb…", "Recursion in stor…", "Daily", "2026-02-07"];
const LONGTREE = ["Attention", "Autoencoder", "Backpropagation", "Bayes", "Chain Rule", "Convolution", "Cross Entropy", "Dropout", "Eigenvalues", "Entropy", "Fourier", "Gradient Descent", "Hash Function", "Hash Table", "Jacobian", "Chain Rule", "Loss Function", "Markov", "Matrix Product", "Momentum", "Normalisation", "Optimizer", "Overfitting", "Perceptron", "Regularisation", "RNN", "Softmax", "Transformer", "Variance", "Vector Space"];
const BODY = [0.94, 0.86, 0.55, 0, 0.91, 0.72, 0.88, 0.42];

function windowChrome(o = {}) {
  const { title = "Backprop as feedback when writing", activeTree = 5, op = 1, scroll = null } = o;
  let s = R(WIN.x, WIN.y, WIN.w, WIN.h, { fill: C.bg, stroke: C.edge, r: 12, op });
  s += LINE(WIN.x, WIN.y + 38, WIN.x + WIN.w, WIN.y + 38, C.edge, op);
  for (let i = 0; i < 3; i++) s += DOT(WIN.x + 20 + i * 17, WIN.y + 19, 5, [C.rose, C.gold, C.green][i], op * 0.85);
  s += T(WIN.x + WIN.w / 2, WIN.y + 24, title, { fill: C.mut, size: 14, anchor: "middle", op: op * 0.9 });
  s += LINE(WIN.x + WIN.ribbon, WIN.y + 38, WIN.x + WIN.ribbon, WIN.y + WIN.h, C.edge, op);
  for (let i = 0; i < 6; i++)
    s += R(WIN.x + 13, WIN.y + 58 + i * 32, 18, 18, { fill: i === 3 ? C.peri : C.faint, r: 5, op: op * (i === 3 ? 0.95 : 0.5) });
  s += LINE(WIN.x + WIN.ribbon + WIN.tree, WIN.y + 38, WIN.x + WIN.ribbon + WIN.tree, WIN.y + WIN.h, C.edge, op);

  if (scroll != null) {
    // the tree racing past: 494 notes, none of which you are going to remember
    const rowH = 25, span = LONGTREE.length * rowH;
    const off = (scroll * span * 3) % span;
    s += `<g clip-path="url(#treeclip)">`;
    for (let pass = 0; pass < 2; pass++) {
      for (let i = 0; i < LONGTREE.length; i++) {
        const y = WIN.y + 56 + i * rowH - off + pass * span;
        if (y < WIN.y + 42 || y > WIN.y + WIN.h - 4) continue;
        s += T(WIN.x + WIN.ribbon + 22, y, LONGTREE[i], { fill: C.dim, size: 13, op: op * 0.9 });
      }
    }
    s += `</g>`;
  } else {
    for (let i = 0; i < TREE.length; i++) {
      const folder = [0, 4, 7].includes(i), y = WIN.y + 62 + i * 25;
      if (i === activeTree) s += R(WIN.x + WIN.ribbon + 8, y - 13, WIN.tree - 16, 22, { fill: C.inner, r: 5, op: op * 0.9 });
      s += T(WIN.x + WIN.ribbon + (folder ? 16 : 26), y + 3, TREE[i], {
        fill: folder ? C.mut : i === activeTree ? C.body : C.dim, size: 13, weight: folder ? 600 : 400, op: op * 0.95,
      });
    }
  }
  return s;
}
function editorBody(o = {}) {
  const { reveal = 1, dim = 1, op = 1, link = 0, caret = -1, t = 0, linked = 0 } = o;
  const x = WIN.ed.x + 34;
  let s = T(x, WIN.y + 84, "Backprop as feedback", { size: 25, weight: 600, op: op * dim });
  let y = WIN.y + 118;
  for (let i = 0; i < BODY.length; i++) {
    if (BODY[i] === 0) { y += 14; continue; }
    const k = cl((reveal - i * 0.06) / 0.2);
    const wpx = (WIN.ed.w - 90) * BODY[i] * k;
    s += R(x, y, wpx, 7, { fill: C.faint, r: 3.5, op: op * 0.75 * dim });
    if (i === 5 && link > 0.004) {
      const lw = 116 * ease(link);
      const tone = linked > 0.5 ? C.cyan : C.peri;
      s += R(x + 62, y, lw, 7, { fill: tone, r: 3.5, op: op * 0.9 * ease(link) });
      // dotted while it is only a suggestion, solid once the click has made it a link
      s += LINE(x + 62, y + 13, x + 62 + lw, y + 13, tone, op * ease(link), linked > 0.5 ? 2.2 : 1.6, linked > 0.5 ? "" : "3 3");
      if (linked > 0.01) {
        s += T(x + 56, y + 7, "[[", { fill: C.cyan, size: 15, weight: 600, op: op * linked });
        s += T(x + 62 + lw + 4, y + 7, "]]", { fill: C.cyan, size: 15, weight: 600, op: op * linked });
      }
    }
    if (i === caret && wpx > 1) s += R(x + wpx + 6, y - 5, 2.5, 17, { fill: C.ink, r: 1, op: op * blink(t) });
    y += 19;
  }
  return s;
}
function panelShell(op = 1, sub = "Based on Backprop as feedback") {
  return LINE(WIN.pn.x, WIN.y + 38, WIN.pn.x, WIN.y + WIN.h, C.edge, op)
    + T(WIN.pn.x + 20, WIN.y + 68, "Smart related notes", { size: 15.5, weight: 600, op })
    + T(WIN.pn.x + 20, WIN.y + 86, sub, { fill: C.mut, size: 12.5, op: op * 0.9 });
}
const CARDS = [
  { t: "Backpropagation", pct: 49, pill: ["Linked", "rec"] },
  { t: "Gradient Descent", pct: 35, pill: ["Related", "rel"] },
  { t: "Machine Learning MOC", pct: 32, pill: ["via Backpropagation", "rec"] },
  { t: "Recurrent Neural Network", pct: 16, pill: ["via Backpropagation", "rec"] },
];
const pillW = (s) => Math.round(s.length * 7.6) + 22;
function pill(x, y, label, kind, op) {
  const w = pillW(label);
  const st = kind === "rec" ? [C.recFill, C.recEdge, C.recInk] : [C.card, C.innerEdge, C.body];
  return R(x, y, w, 19, { fill: st[0], stroke: st[1], r: 6, op }) + T(x + w / 2, y + 13.5, label, { fill: st[2], size: 12.5, anchor: "middle", op });
}
function card(i, k, op = 1, hi = 0) {
  if (k <= 0.004) return "";
  const c = CARDS[i];
  const y = WIN.y + 106 + i * 78 + (1 - k) * 14;
  const x = WIN.pn.x + 14, w = WIN.pn.w - 28;
  let s = R(x, y, w, 66, { fill: C.inner, stroke: hi > 0.01 ? C.recEdge : C.innerEdge, r: 9, op, sw: 1 + hi * 0.8 });
  if (hi > 0.01) s += RING(x + w / 2, y + 31, 0, C.recEdge, 0);
  s += T(x + 16, y + 24, c.t, { fill: C.body, size: 15, weight: 500, op });
  s += R(x + w - 58, y + 11, 46, 22, { fill: c.pct >= 30 ? C.peri : C.grey, r: 11, op });
  s += T(x + w - 35, y + 26, `${c.pct}%`, { fill: "#0d0d12", size: 13, weight: 700, anchor: "middle", op });
  s += pill(x + 16, y + 34, c.pill[0], c.pill[1], op);
  return s;
}

// =================================================================== 1. problem
function sProblem(t) {
  let s = says(t, [
    { at: 0.4, head: "You already wrote about this.", sub: "Months ago, in a note you have forgotten." },
    { at: 3.4, head: "494 notes in.", sub: "You are not going to find it by scrolling." },
  ]);
  s += windowChrome({ scroll: t > 3.0 ? ease((t - 3.0) / 2.4) * 0.9 : null });
  s += editorBody({ reveal: ease((t - 0.2) / 1.4), caret: 7, t });
  s += panelShell(hold(t, 0.9, null, 0.6) * 0.55);
  return s;
}

// ============================================================ 2. search cannot
function sSearch(t) {
  let s = says(t, [
    { at: 0.3, head: "Search needs the words you happened to use.", sub: "" },
    { at: 3.2, head: "The note that matters used different ones.", sub: "Sometimes in a different language." },
  ]);
  const bx = 300, bw = 600;
  // a search field with a query typed into it
  const q = "gradient descent";
  const typed = q.slice(0, Math.floor(cl((t - 0.5) / 1.1) * q.length));
  s += R(bx, 130, bw, 52, { fill: C.card, stroke: C.cardEdge, r: 26, op: hold(t, 0.2, null, 0.4) });
  s += `<g opacity="${hold(t, 0.2, null, 0.4).toFixed(3)}"><circle cx="${bx + 32}" cy="156" r="8" fill="none" stroke="${C.dim}" stroke-width="2"/><line x1="${bx + 38}" y1="162" x2="${bx + 45}" y2="169" stroke="${C.dim}" stroke-width="2"/></g>`;
  s += T(bx + 58, 162, typed || "search…", { fill: typed ? C.body : C.dim, size: 20, op: hold(t, 0.2, null, 0.4) });
  if (t > 0.5 && t < 2.4) s += R(bx + 60 + typed.length * 10.4, 143, 2.5, 24, { fill: C.ink, r: 1, op: blink(t) });

  // the note you would have wanted, which shares no words with the query
  const nk = hold(t, 3.4, null, 0.5);
  const noteCard = (x, title, lines, tone, k, gloss) => {
    if (k <= 0.004) return "";
    let o = R(x, 240, 380, 206, { fill: C.card, stroke: C.cardEdge, r: 12, op: k });
    o += T(x + 24, 274, title, { fill: C.body, size: 17, weight: 600, op: k });
    for (let i = 0; i < lines.length; i++) o += T(x + 24, 306 + i * 26, lines[i], { fill: i === 0 ? tone : C.mut, size: 15.5, op: k });
    if (gloss) o += T(x + 24, 420, gloss, { fill: C.dim, size: 13, op: k * 0.95 });
    return o;
  };
  const hit = hold(t, 1.9, null, 0.5);
  s += noteCard(150, "What you are writing now", ["“gradient descent”", "the optimizer follows the slope", "downhill, step by step"], C.peri, hit, "English");
  s += noteCard(670, "2026-02-07, six months ago", ["“den Fehler rückwärts schieben”", "Kettenregel per Hand gerechnet,", "drei Seiten Papier"], C.gold, nk, "German, and it means the same thing");
  const xk = hold(t, 4.3, null, 0.45);
  if (xk > 0.004) {
    s += LINE(532, 343, 668, 343, C.innerEdge, xk * 0.8, 1.6, "6 6");
    s += LINE(590, 333, 610, 353, C.rose, xk, 2.4);
    s += LINE(610, 333, 590, 353, C.rose, xk, 2.4);
    s += T(600, 486, "Not one word in common. Search will never join these.", { fill: C.body, size: 16.5, anchor: "middle", op: xk });
  }
  return s;
}

// ================================================== 3. the idea, given its time
const WORDS = ["gradient", "loss", "chain rule", "backprop"];
const DEWORDS = ["Fehler", "rückwärts", "Kettenregel"];
function sIdea(t, map) {
  let s = says(t, [
    { at: 0.3, head: "So do not match words. Place them.", sub: "A small model reads a note and returns one position." },
    { at: 4.6, head: "Every note gets a place.", sub: "Nothing is tagged, nothing is filed." },
    { at: 7.4, head: "Close together means the same idea.", sub: "Which is why the two notes above end up neighbours." },
  ]);
  const note = { x: 165, y: 250 };
  const model = { x: 452, y: 300 };
  const F = { x: 632, y: 106, w: 528, h: 424 };
  s += R(F.x, F.y, F.w, F.h, { fill: C.card, stroke: C.cardEdge, r: 14, op: hold(t, 0.5, null, 0.5) });

  const noteCard = (p, title, lang, k) => k <= 0.004 ? "" :
    R(p.x - 115, p.y - 48, 230, 96, { fill: C.inner, stroke: C.innerEdge, r: 10, op: k })
    + T(p.x - 97, p.y - 20, title, { fill: C.body, size: 15.5, weight: 600, op: k })
    + T(p.x - 97, p.y, lang, { fill: C.mut, size: 12.5, op: k });
  s += noteCard(note, "Backprop as feedback", "written in English", hold(t, 0.15, null, 0.4));
  s += noteCard({ x: note.x, y: note.y + 190 }, "2026-02-07", "written in German", hold(t, 6.2, null, 0.4));

  const mk = hold(t, 0.35, null, 0.4);
  s += R(model.x - 66, model.y - 34, 132, 68, { fill: C.inner, stroke: C.peri, r: 12, op: mk });
  s += T(model.x, model.y - 4, "embedding model", { fill: C.peri, size: 13.5, weight: 600, anchor: "middle", op: mk * (0.62 + 0.38 * Math.abs(Math.sin(t * 4))) });
  s += T(model.x, model.y + 16, "runs on your Mac", { fill: C.mut, size: 11.5, anchor: "middle", op: mk * 0.9 });

  const pA = { x: F.x + F.w * 0.42, y: F.y + F.h * 0.38 };
  const pB = { x: F.x + F.w * 0.50, y: F.y + F.h * 0.46 };
  const fly = (words, from, st, tone) => {
    let out = "";
    for (let i = 0; i < words.length; i++) {
      const k = ease((t - st - i * 0.16) / 0.8);
      if (k <= 0.004 || k >= 1) continue;
      const f = { x: from.x - 46 + (i % 2) * 70, y: from.y + 8 + i * 7 };
      const p = qbez(f, { x: (f.x + model.x) / 2, y: f.y - 76 }, model, k);
      const w = pillW(words[i]);
      out += R(p.x - w / 2, p.y - 10, w, 20, { fill: C.inner, stroke: tone, r: 6, op: 1 - k * k })
        + T(p.x, p.y + 4, words[i], { fill: tone, size: 13, anchor: "middle", op: 1 - k * k });
    }
    return out;
  };
  s += fly(WORDS, note, 0.9, C.body);
  s += fly(DEWORDS, { x: note.x, y: note.y + 190 }, 6.6, C.gold);

  const land = (target, st, tone) => {
    const k = ease((t - st) / 0.8);
    if (k <= 0.004) return "";
    const p = qbez(model, { x: (model.x + target.x) / 2, y: model.y - 70 }, target, k);
    let out = DOT(p.x, p.y, L(12, 7.5, k), tone);
    if (k >= 1) { const r = ease((t - st - 0.8) / 0.7); out += RING(target.x, target.y, 9 + r * 34, tone, 0.55 * (1 - r), 2); }
    return out;
  };
  s += land(pA, 2.0, C.peri);

  // then the rest of the vault streams in behind it: these are the real points
  const rest = map.pts;
  const sx = (x) => F.x + 18 + (x - map.xr[0]) / (map.xr[1] - map.xr[0]) * (F.w - 36);
  const sy = (y) => F.y + 18 + (1 - (y - map.yr[0]) / (map.yr[1] - map.yr[0])) * (F.h - 36);
  for (let i = 0; i < rest.length; i++) {
    const k = ease((t - 4.3 - (i % 120) * 0.012) / 0.5);
    if (k <= 0.004) continue;
    s += DOT(sx(rest[i].x), sy(rest[i].y), 2.5 * k, C.faint, 0.85 * k);
  }
  s += land(pB, 8.0, C.gold);
  const nk = hold(t, 8.9, null, 0.45);
  if (nk > 0.004) {
    s += LINE(pA.x, pA.y, pB.x, pB.y, C.green, nk * 0.95, 2.4);
    s += T((pA.x + pB.x) / 2, (pA.y + pB.y) / 2 + 46, "different words, one place", { fill: C.recInk, size: 15.5, anchor: "middle", op: nk });
  }
  return s;
}

// ========================================================== 4. so, the sidebar
function sSidebar(t) {
  let s = says(t, [
    { at: 0.3, head: "So the sidebar can answer before you ask.", sub: "It is just the nearest points, kept up to date." },
    { at: 3.6, head: "There is your six-month-old note.", sub: "You never searched for it." },
  ]);
  s += windowChrome();
  s += editorBody({ dim: 0.55 });
  s += panelShell();
  for (let i = 0; i < CARDS.length; i++) s += card(i, ease((t - 0.5 - i * 0.28) / 0.6));
  const hi = hold(t, 3.8, null, 0.4);
  if (hi > 0.004) {
    const y = WIN.y + 106 + 3 * 78, x = WIN.pn.x + 14, w = WIN.pn.w - 28;
    s += R(x - 3, y - 3, w + 6, 72, { fill: "none", stroke: C.recEdge, r: 11, op: hi, sw: 2 });
  }
  return s;
}

// ================================================== 5. where meaning runs out
function sLimit(t) {
  let s = says(t, [
    { at: 0.3, head: "But meaning alone has a blind spot.", sub: "" },
    { at: 2.6, head: "These two belong together.", sub: "They share no wording at all, so similarity cannot see it." },
    { at: 5.6, head: "You already knew. You linked them both to a third note.", sub: "" },
  ]);
  const A = { x: 300, y: 275 }, B = { x: 900, y: 275 }, M = { x: 600, y: 460 };
  const node = (p, label, sub, k, tone) => {
    if (k <= 0.004) return "";
    const w = 218, h = 64, r = back(k);
    return R(p.x - w / 2 * r, p.y - h / 2 * r, w * r, h * r, { fill: C.inner, stroke: tone, r: 11, op: k })
      + T(p.x, p.y - 2, label, { fill: C.body, size: 17, weight: 600, anchor: "middle", op: k })
      + T(p.x, p.y + 19, sub, { fill: C.mut, size: 12.5, anchor: "middle", op: k });
  };
  s += node(A, "Entropy", "information theory", hold(t, 0.5, null, 0.45), C.innerEdge);
  s += node(B, "Hash Table", "data structures", hold(t, 0.75, null, 0.45), C.innerEdge);
  const xk = hold(t, 2.9, null, 0.45);
  if (xk > 0.004) {
    s += LINE(A.x + 112, A.y, B.x - 112, B.y, C.innerEdge, xk * 0.9, 1.6, "6 6");
    const mx = (A.x + B.x) / 2;
    s += LINE(mx - 11, A.y - 11, mx + 11, A.y + 11, C.rose, xk, 2.4);
    s += LINE(mx + 11, A.y - 11, mx - 11, A.y + 11, C.rose, xk, 2.4);
    s += T(mx, A.y - 32, "no words in common", { fill: C.mut, size: 14.5, anchor: "middle", op: xk });
  }
  s += node(M, "Hash Function", "you linked both of these", hold(t, 5.8, null, 0.45), C.recEdge);
  const ek = ease((t - 6.2) / 0.7);
  if (ek > 0.004) {
    s += LINE(M.x - 74, M.y - 30, L(M.x - 74, A.x, ek), L(M.y - 30, A.y + 34, ek), C.green, 0.95, 2.6);
    s += LINE(M.x + 74, M.y - 30, L(M.x + 74, B.x, ek), L(M.y - 30, B.y + 34, ek), C.green, 0.95, 2.6);
  }
  const rk = hold(t, 7.0, null, 0.45);
  if (rk > 0.004) {
    s += LINE(A.x, A.y - 52, B.x, B.y - 52, C.recInk, rk * 0.85, 2.4);
    s += T((A.x + B.x) / 2, A.y - 64, "so 3.0 counts them as related", { fill: C.recInk, size: 16, anchor: "middle", op: rk });
    s += T(W / 2, 566, "A third of the links you make point where wording cannot look.  Held-out link recall 0.66 → 0.75.", {
      fill: C.body, size: 15.5, anchor: "middle", op: rk,
    });
  }
  return s;
}

// ============================================================== 6. the receipt
function sReceipt(t) {
  let s = says(t, [
    { at: 0.3, head: "And every card tells you why it is there.", sub: "Named, so you can judge it at a glance." },
    { at: 3.4, head: "Mentioned a concept you have a note for?", sub: "It glows. One click makes it a link." },
  ]);
  s += windowChrome();
  const CLICK = 4.5;
  s += editorBody({ dim: 0.8, link: ease((t - 3.5) / 0.6), t, linked: hold(t, CLICK, null, 0.3) });
  s += panelShell();
  for (let i = 0; i < CARDS.length; i++) s += card(i, 1, 1, i >= 2 ? hold(t, 0.7 + (i - 2) * 0.3, null, 0.4) : 0);
  // the phrase sits on body line 5; put the pointer ON it, and let it travel in
  const gy = WIN.y + 208, gx = WIN.ed.x + 34 + 62;
  const pk = hold(t, 3.9, null, 0.3);
  if (pk > 0.004) {
    const travel = ease((t - 3.9) / 0.6);
    const px = L(gx + 190, gx + 58, travel), py = L(gy + 66, gy + 6, travel);
    if (t >= CLICK) {
      const r = cl((t - CLICK) / 0.55);
      s += RING(px, py, 4 + r * 26, C.cyan, 0.75 * (1 - r), 2.2);
    }
    s += `<path d="M ${px.toFixed(1)} ${py.toFixed(1)} l 0 16 l 4.5 -4.5 l 3 7.5 l 3.5 -1.5 l -3 -7 l 6.5 0 Z" fill="${C.ink}" stroke="${C.page}" stroke-width="1" opacity="${(pk * 0.98).toFixed(3)}"/>`;
  }
  return s;
}

// ============================================================ 7. the whole vault
function sMap(t, map) {
  let s = says(t, [
    { at: 0.3, head: "Step back, and the same points are a map.", sub: "" },
    { at: 3.0, head: "It names the clusters from the notes themselves.", sub: "Purity 0.65, against 0.29 for a shuffled baseline." },
  ]);
  s += windowChrome({ title: "Vault map", activeTree: -1 });
  s += panelShell(1, "Clusters");
  const G = { x: WIN.ed.x + 20, y: WIN.y + 54, w: WIN.ed.w - 40, h: WIN.h - 76 };
  s += R(G.x, G.y, G.w, G.h, { fill: C.card, stroke: C.cardEdge, r: 10 });
  const COL = [C.green, C.peri, C.gold, C.cyan, C.grey, C.rose];
  const { pts, xr, yr } = map;
  const sx = (x) => G.x + 14 + (x - xr[0]) / (xr[1] - xr[0]) * (G.w - 28);
  const sy = (y) => G.y + 14 + (1 - (y - yr[0]) / (yr[1] - yr[0])) * (G.h - 28);
  // grey first (continuous with scene 3), then colour arrives cluster by cluster
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    const ck = ease((t - 2.6 - p.k * 0.34) / 0.6);
    s += DOT(sx(p.x), sy(p.y), 2.8, ck > 0.5 ? COL[p.k % 6] : C.faint, 0.78);
    if (ck > 0.004 && ck <= 0.5) s += DOT(sx(p.x), sy(p.y), 2.8, COL[p.k % 6], 0.78 * ck * 2);
  }
  let ly = WIN.y + 116;
  for (let li = 0; li < map.legend.length; li++) {
    const l = map.legend[li], k = hold(t, 2.9 + li * 0.34, null, 0.4);
    s += DOT(WIN.pn.x + 26, ly - 4, 4.5, COL[l.k % 6], k);
    s += T(WIN.pn.x + 40, ly, l.name, { fill: C.body, size: 14.5, op: k });
    s += T(WIN.pn.x + 40, ly + 15, `${l.n} notes`, { fill: C.mut, size: 12, op: k });
    ly += 42;
  }
  return s;
}

// ==================================================================== 8. close
function sClose(t) {
  const a = hold(t, 0.15, null, 0.5);
  let s = T(W / 2 - 52, 258, "Smart Related Notes", { size: 50, weight: 600, anchor: "middle", ls: -1.2, op: a });
  s += T(W / 2 + 262, 258, "3.0", { size: 50, weight: 600, anchor: "middle", fill: C.green, op: a });
  s += T(W / 2, 300, "Community plugins  ›  Browse  ›  Smart Related Notes", { fill: C.body, size: 18.5, anchor: "middle", op: hold(t, 0.7, null, 0.45) });
  const c = hold(t, 1.2, null, 0.45);
  s += R(W / 2 - 230, 330, 460, 46, { fill: C.recFill, stroke: C.recEdge, r: 10, op: c });
  s += T(W / 2, 360, "Local, offline, and measured before it ships", { fill: C.recInk, size: 16.5, anchor: "middle", op: c });
  return s;
}

// ------------------------------------------------------------------ timeline
const SCENES = [
  { id: "sProblem", dur: 6.0 },
  { id: "sSearch",  dur: 6.2 },
  { id: "sIdea",    dur: 10.2 },
  { id: "sSidebar", dur: 5.4 },
  { id: "sLimit",   dur: 8.4 },
  { id: "sReceipt", dur: 5.8 },
  { id: "sMap",     dur: 6.6 },
  { id: "sClose",   dur: 3.2 },
];
const DRAW = { sProblem, sSearch, sIdea, sSidebar, sLimit, sReceipt, sMap, sClose };

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

const CLIP = `<clipPath id="treeclip"><rect x="${WIN.x + WIN.ribbon}" y="${WIN.y + 39}" width="${WIN.tree}" height="${WIN.h - 40}"/></clipPath>`;
const total = Math.round(TOTAL * FPS);
for (let n = 0; n < total; n++) {
  const gt = n / FPS;
  const layers = [];
  for (let i = 0; i < SCENES.length; i++) {
    const sc = SCENES[i];
    const t = gt - sc.start;
    if (t < -0.001 || t > sc.dur) continue;
    let a = 1, bl = 0;
    if (i > 0 && t < XF) { const u = cl(t / XF); a = easeIO(u); bl = BLUR * (1 - u); }
    if (i < SCENES.length - 1 && t > sc.dur - XF) {
      const u = cl((t - (sc.dur - XF)) / XF);
      a = Math.min(a, easeIO(1 - u)); bl = Math.max(bl, BLUR * u);
    }
    if (a <= 0.004) continue;
    layers.push({ i, body: DRAW[sc.id](t, map), a, bl });
  }
  let defs = CLIP, g = "";
  for (const l of layers) {
    let f = "";
    if (l.bl > 0.05) {
      const id = `b${l.i}`;
      defs += `<filter id="${id}" x="-8%" y="-8%" width="116%" height="116%"><feGaussianBlur stdDeviation="${l.bl.toFixed(2)}"/></filter>`;
      f = ` filter="url(#${id})"`;
    }
    g += `<g opacity="${l.a.toFixed(3)}"${f}>${l.body}</g>`;
  }
  g += R(46, H - 16, W - 92, 4, { fill: C.faint, r: 2, op: 0.55 });
  g += R(46, H - 16, (W - 92) * (n / total), 4, { fill: C.green, r: 2, op: 0.95 });
  writeFileSync(join(OUT, `f${String(n).padStart(5, "0")}.svg`),
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" font-family="${FONT}">`
    + `<defs>${defs}</defs><rect width="${W}" height="${H}" fill="${C.page}"/>${g}</svg>`);
}
console.log(`${total} frames, ${TOTAL.toFixed(1)}s at ${FPS}fps, ${XF}s crossfades -> ${OUT}`);
