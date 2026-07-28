// "Surprising Connections" product output: pairs the fusion model scores highly
// that are (a) NOT already linked and (b) mutually OUTSIDE each other's content
// top-10 (non-obvious to pure similarity). This is the 3.0 magic-feature candidate.
import { readFileSync, writeFileSync } from "node:fs";
import { relative } from "node:path";
import { walk, stripFront, noteText } from "./jina-cache.mjs";
const VAULT = "/Users/justus/obsidian_atomized_intermediary/lab/vault";
const CACHE_PATH = process.env.JINA_CACHE || (process.env.HOME + "/.cache/srn-lab/jina-cache.json");
const PREFIX = "Document: ";
const fold = (s) => s.toLowerCase().normalize("NFC").replace(/ä/g,"ae").replace(/ö/g,"oe").replace(/ü/g,"ue").replace(/ß/g,"ss").replace(/\s+/g," ").trim();
const dot=(a,b)=>{let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s;};
const l2=(v)=>{let s=0;for(const x of v)s+=x*x;s=Math.sqrt(s)||1;return v.map(x=>x/s);};
const zmap=(arr)=>{const m=arr.reduce((s,x)=>s+x,0)/arr.length;const sd=Math.sqrt(arr.reduce((s,x)=>s+(x-m)**2,0)/arr.length)||1;return arr.map(x=>(x-m)/sd);};
const manifest=JSON.parse(readFileSync("/Users/justus/obsidian_atomized_intermediary/lab/corpus-manifest.json","utf8"));
const INDEXABLE=new Set(manifest.answer_paths.map(p=>"real/"+p));
const cache=JSON.parse(readFileSync(CACHE_PATH,"utf8"));
const vec=(t)=>cache[PREFIX+t]||null;
const notes=[];
for(const abs of walk(VAULT)){const rel=relative(VAULT,abs);if(rel.startsWith("real/")&&!INDEXABLE.has(rel))continue;const bn=rel.replace(/\.md$/,"").split("/").pop();if(/\bMOC\b/i.test(bn)||/(^|\/)Attachments\//.test(rel)||/\.dup$/.test(bn))continue;const body=stripFront(readFileSync(abs,"utf8"));notes.push({rel,bn,body,v:vec(noteText(bn,body))});}
const wv=notes.filter(n=>n.v);const N=wv.length,DIM=wv[0].v.length;
const idx=new Map(wv.map((n,i)=>[n,i]));const byFold=new Map(wv.map(n=>[fold(n.bn),n]));
const mean=new Array(DIM).fill(0);for(const n of wv)for(let i=0;i<DIM;i++)mean[i]+=n.v[i];for(let i=0;i<DIM;i++)mean[i]/=N;
const C=wv.map(n=>l2(n.v.map((x,i)=>x-mean[i])));
const adj=Array.from({length:N},()=>new Set());
for(const n of wv)for(const m of n.body.matchAll(/\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]/g)){const t=byFold.get(fold(m[1]));if(!t||t===n)continue;adj[idx.get(n)].add(idx.get(t));adj[idx.get(t)].add(idx.get(n));}
const AA=(s,t)=>{let a=0;for(const x of adj[s])if(adj[t].has(x))a+=1/Math.log(1+adj[x].size);return a;};
// content top-10 per node (obviousness set)
const contentTop10=[];for(let s=0;s<N;s++){const sc=[];for(let t=0;t<N;t++)if(t!==s)sc.push([dot(C[s],C[t]),t]);sc.sort((a,b)=>b[0]-a[0]);contentTop10.push(new Set(sc.slice(0,10).map(x=>x[1])));}
// fused suggestions: for each s, rank candidates by z(content)+z(AA); keep those NOT linked and mutually non-obvious
const seen=new Set(),out=[];
for(let s=0;s<N;s++){
  const cands=[];for(let t=0;t<N;t++){if(t===s||adj[s].has(t))continue;cands.push(t);}
  if(!cands.length)continue;
  const zc=zmap(cands.map(t=>dot(C[s],C[t]))),za=zmap(cands.map(t=>AA(s,t)));
  const ranked=cands.map((t,i)=>[zc[i]+za[i],t,dot(C[s],C[t]),AA(s,t)]).sort((a,b)=>b[0]-a[0]);
  for(const [score,t,cc,aa] of ranked.slice(0,5)){
    if(aa<=0)continue; // needs at least one shared neighbor (structural evidence)
    if(contentTop10[s].has(t)||contentTop10[t].has(s))continue; // must be non-obvious BOTH ways
    const key=[Math.min(s,t),Math.max(s,t)].join("-");if(seen.has(key))continue;seen.add(key);
    const shared=[...adj[s]].filter(x=>adj[t].has(x)).map(x=>wv[x].bn);
    out.push({score:+score.toFixed(2),contentCos:+cc.toFixed(3),adamicAdar:+aa.toFixed(2),a:wv[s].rel,b:wv[t].rel,sharedNeighbors:shared.slice(0,5)});
  }
}
out.sort((a,b)=>b.score-a.score);
writeFileSync("/Users/justus/obsidian_atomized_intermediary/lab/results/surprising-connections.json",JSON.stringify(out.slice(0,40),null,1));
console.log(`mined ${out.length} non-obvious unlinked suggestions (mutually outside content top-10, >=1 shared neighbor)`);
for(const e of out.slice(0,22))console.log(`  ${e.score}  cos ${e.contentCos}  AA ${e.adamicAdar}  ${e.a.split("/").pop().replace(".md","").slice(0,26).padEnd(27)} <> ${e.b.split("/").pop().replace(".md","").slice(0,26).padEnd(27)} via [${e.sharedNeighbors.slice(0,3).join(", ")}]`);
