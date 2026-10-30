// True when an embed failed because the input was too big for the runtime, rather
// than because the runtime is unavailable. Retrying with smaller inputs only helps
// for the former; for a missing model or a dead worker it would just repeat a slow
// failure once per tier per note. The wasm heap runs out in several different ways
// depending on which allocation overflows first, hence the several spellings.
export function isCapacityError(e: unknown): boolean {
  const msg = (e instanceof Error ? e.message : String(e)).toLowerCase();
  return (
    msg.includes("memory access out of bounds") ||
    msg.includes("bad_alloc") ||
    msg.includes("integer overflow") ||
    msg.includes("out of memory") ||
    msg.includes("cannot enlarge memory") ||
    msg.includes("maximum call stack")
  );
}
