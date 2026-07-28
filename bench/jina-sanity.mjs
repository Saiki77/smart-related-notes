import { pipeline, env } from "@huggingface/transformers";
env.allowLocalModels = false;
const MODEL = "jinaai/jina-embeddings-v5-text-nano-text-matching";
const t0 = Date.now();
const pipe = await pipeline("feature-extraction", MODEL, { dtype: "fp32" });
console.log("model loaded in", ((Date.now()-t0)/1000).toFixed(1), "s");
function lastTokenNorm(out) {
  const d = out.dims, data = out.data;
  const seq = d.length === 3 ? d[1] : d[0], dim = d.length === 3 ? d[2] : d[1];
  const v = Array.from(data.subarray((seq-1)*dim, seq*dim));
  const n = Math.sqrt(v.reduce((s,x)=>s+x*x,0)) || 1;
  return v.map(x=>x/n);
}
async function embed(t) {
  const o = await pipe("Document: " + t, { pooling: "none" });
  const v = lastTokenNorm(o); o.dispose?.(); return v;
}
const dot = (a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
const texts = {
  de: "Eine schlechtere Lösung wird trotzdem akzeptiert, und zwar mit Wahrscheinlichkeit exp(-ΔE/T), wobei T die aktuelle Temperatur ist.",
  en: "A worse candidate is still accepted with probability exp(-ΔE/T), where T is the current temperature.",
  unrel: "Der Feldspieler darf den Ball nicht mit der Hand spielen, sonst gibt es Freistoß.",
  cache: "Caching works because real workloads exhibit locality of reference: data used recently tends to be used again soon.",
  hier: "The memory hierarchy pays off because of temporal and spatial locality of programs.",
};
const v = {};
const t1 = Date.now();
for (const [k, t] of Object.entries(texts)) v[k] = await embed(t);
console.log("5 embeds in", ((Date.now()-t1)/1000).toFixed(1), "s, dim", v.de.length);
console.log("DE~EN twin      :", dot(v.de, v.en).toFixed(3));
console.log("DE~unrelated DE :", dot(v.de, v.unrel).toFixed(3));
console.log("EN~unrelated DE :", dot(v.en, v.unrel).toFixed(3));
console.log("cache~hierarchy :", dot(v.cache, v.hier).toFixed(3));
console.log("cache~annealing :", dot(v.cache, v.de).toFixed(3));
