// Generates docs/feature-vault-map.svg from REAL vectors, so the figure is the
// output of the algorithm rather than a drawing of it. Same spherical k-means
// (k-means++, seed 20260727) and power-iteration PCA the shipped map view uses,
// over the cached lab-vault embeddings, restricted to the same real corpus every
// other figure in the README is measured on.
//
//   node bench/vault-map-figure.mjs /tmp/mapdata.json
//
import { readFileSync, writeFileSync } from "node:fs";
const CACHE = process.env.HOME + "/.cache/srn-lab/v3-Xenova_paraphrase_multilingual_MiniLM_L12_v2.json";
const cache = JSON.parse(readFileSync(CACHE, "utf8"));
const manifest = JSON.parse(readFileSync("/Users/justus/obsidian_atomized_intermediary/lab/corpus-manifest.json","utf8"));
// Restrict to the SAME 494 real notes every other figure in the README is measured
// on. The cache also holds the generated corpus; including it produced a 1,421-note
// map sitting next to a graphic that says "a 494-note vault".
const VAULT = "/Users/justus/obsidian_atomized_intermediary/lab/vault";
const stripFront = (r) => { const m = r.match(/^---\n[\s\S]*?\n---\n?/); return m ? r.slice(m[0].length) : r; };
const rows = [];
for (const rel of manifest.answer_paths) {
  const abs = VAULT + "/real/" + rel;
  let raw; try { raw = readFileSync(abs, "utf8"); } catch { continue; }
  const base = rel.slice(rel.lastIndexOf("/") + 1).replace(/\.md$/, "");
  const text = (base + "\n\n" + stripFront(raw)).slice(0, 8000);
  const v = cache[text];
  if (v) rows.push({ title: base, v: Float64Array.from(v) });
}
const D = rows[0].v.length, N = rows.length;
// mean-center then renormalise, exactly as the ranker does
const mu = new Float64Array(D);
for (const r of rows) for (let i=0;i<D;i++) mu[i]+=r.v[i]/N;
for (const r of rows){ let s=0; for(let i=0;i<D;i++){ r.v[i]-=mu[i]; s+=r.v[i]*r.v[i]; } const n=Math.sqrt(s)||1; for(let i=0;i<D;i++) r.v[i]/=n; }
// spherical k-means, k-means++ seeded, deterministic LCG
let seed=20260727; const rnd=()=>((seed=(seed*1664525+1013904223)>>>0)/4294967296);
const K=6, cent=[];
cent.push(rows[Math.floor(rnd()*N)].v.slice());
while(cent.length<K){
  const d2=rows.map(r=>{let best=-1;for(const c of cent){let s=0;for(let i=0;i<D;i++)s+=r.v[i]*c[i];best=Math.max(best,s);}return Math.max(1e-9,(1-best)**2);});
  const tot=d2.reduce((a,b)=>a+b,0); let t=rnd()*tot,idx=0;
  for(;idx<N;idx++){t-=d2[idx]; if(t<=0)break;}
  cent.push(rows[Math.min(idx,N-1)].v.slice());
}
let assign=new Array(N).fill(0);
for(let it=0;it<40;it++){
  for(let n=0;n<N;n++){let bi=0,bs=-2;for(let k=0;k<K;k++){let s=0;for(let i=0;i<D;i++)s+=rows[n].v[i]*cent[k][i];if(s>bs){bs=s;bi=k;}}assign[n]=bi;}
  for(let k=0;k<K;k++){const acc=new Float64Array(D);let c=0;
    for(let n=0;n<N;n++) if(assign[n]===k){c++;for(let i=0;i<D;i++)acc[i]+=rows[n].v[i];}
    if(!c) continue; let s=0; for(let i=0;i<D;i++)s+=acc[i]*acc[i]; const nn=Math.sqrt(s)||1;
    for(let i=0;i<D;i++)cent[k][i]=acc[i]/nn;}
}
// power-iteration PCA to 2D
function topPC(vecs, exclude){
  let p=new Float64Array(D); for(let i=0;i<D;i++)p[i]=rnd()-0.5;
  for(let it=0;it<80;it++){
    const acc=new Float64Array(D);
    for(const v of vecs){let d=0;for(let i=0;i<D;i++)d+=v[i]*p[i];for(let i=0;i<D;i++)acc[i]+=d*v[i];}
    if(exclude){let d=0;for(let i=0;i<D;i++)d+=acc[i]*exclude[i];for(let i=0;i<D;i++)acc[i]-=d*exclude[i];}
    let s=0;for(let i=0;i<D;i++)s+=acc[i]*acc[i];const n=Math.sqrt(s)||1;
    for(let i=0;i<D;i++)p[i]=acc[i]/n;
  }
  return p;
}
const vecs=rows.map(r=>r.v);
const pc1=topPC(vecs,null), pc2=topPC(vecs,pc1);
const pts=rows.map((r,n)=>{let x=0,y=0;for(let i=0;i<D;i++){x+=r.v[i]*pc1[i];y+=r.v[i]*pc2[i];}return {x,y,k:assign[n],title:r.title};});
// c-TF-IDF-ish cluster labels from titles
const stop=new Set(["der","die","das","und","the","and","for","von","mit","ein","eine","des","dem","den","zu","im","in","of","a","an","is","to","on","als"]);
const labels=[];
for(let k=0;k<K;k++){
  const inC=pts.filter(p=>p.k===k), df=new Map(), gf=new Map();
  for(const p of pts){ for(const w of new Set(p.title.toLowerCase().split(/[^\p{L}\p{N}]+/u).filter(w=>w.length>3&&!stop.has(w)))) gf.set(w,(gf.get(w)||0)+1); }
  for(const p of inC){ for(const w of new Set(p.title.toLowerCase().split(/[^\p{L}\p{N}]+/u).filter(w=>w.length>3&&!stop.has(w)))) df.set(w,(df.get(w)||0)+1); }
  const scored=[...df].map(([w,c])=>[w, (c/Math.max(1,inC.length)) * Math.log(pts.length/(gf.get(w)||1))]).sort((a,b)=>b[1]-a[1]);
  labels.push({k, n:inC.length, words:scored.slice(0,2).map(x=>x[0])});
}
writeFileSync(process.argv[2], JSON.stringify({pts,labels},null,0));
console.log("clusters:", labels.map(l=>`${l.words.join("/")} (${l.n})`).join(", "));
