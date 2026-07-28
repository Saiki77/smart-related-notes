// Fisher-Yates, seeded.
//
// Every harness in this directory used `[...xs].sort(() => rnd() - 0.5)` to make the
// held-out split. That is not a shuffle. A comparator that answers randomly is not a
// consistent ordering, so the sort's behaviour depends on its internal pivot choices
// and the result is a strongly biased permutation: measured over 200k trials on an
// 8-element array, the first element stayed in position 0 about 77% more often than
// chance, chi-square 21725 against an expected 7.
//
// The consequence was that "20% of links held out at random" was really "20% of links
// held out with a bias towards the order the edges happened to be collected in", which
// tracks folder and filename order. Every number the project published came through it.
export function fisherYates(xs, rnd) {
  const a = [...xs];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}
