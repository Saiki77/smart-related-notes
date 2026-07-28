# Reddit draft — r/ObsidianMD

Deliberately not the forum post. Different room, different expectations:

- **Shorter.** The forum tolerates a spec sheet; Reddit does not. Lead with the
  demo and one idea, not a feature table.
- **First person, no launch voice.** "I built this because X kept happening" beats
  "3.0 introduces". Anything that smells like marketing gets downvoted.
- **The video is the post.** Upload the MP4 natively rather than linking out; a
  link-only self-promo post reads as spam and Reddit throttles outbound links.
- **One number, not six.** Save the measurement table for people who ask in the
  comments, and they will.
- **Check the subreddit rules before posting.** r/ObsidianMD restricts plugin
  self-promotion (frequency, flair, sometimes a weekly thread). Read the sidebar
  first; if it requires a specific flair, use it.

Post the video as a native upload, title below, body below that.

---

## Title

I got tired of never finding the note I knew I'd already written, so I made the sidebar find it for me

## Body

The problem I actually had: I'd write something, and six months later write about the same thing again, having completely forgotten the first note existed. Search didn't help, because search needs the words I happened to use — and past me had used different ones. Sometimes in a different language.

So the plugin doesn't match words. It turns every note into a position using a small embedding model that runs **entirely on your machine**, and shows you the nearest ones in a sidebar panel. No query, no tags, no folders. After a one-time model download it's fully offline, and nothing is ever sent anywhere.

The bit I'm actually pleased with is in the new version. Similarity alone has a blind spot: I masked 20% of the links in my vault and checked whether the panel could recover them, and about **a third of the links I'd actually made pointed at notes that similarity structurally could not see** — no shared wording at all. But I'd already connected those pairs myself, through some third note. So 3.0 reads that link structure too and fuses it with meaning. Recall on the held-out links went 0.66 → 0.75.

And when the link graph is why something surfaced, the card tells you which note bridged them — `via Hash Function` — so you can dismiss it in a second if it's wrong.

There's also a map view that lays the whole vault out as points and names the clusters itself, which was mostly built for debugging and turned out to be the thing I use most.

Desktop only, Obsidian 1.7.2+. It's in Community plugins as **Smart Related Notes**.

Repo, with the architecture write-up and every measurement including the ideas I built and then cut for making things worse: https://github.com/Saiki77/smart-related-notes

Happy to get into the ranking details if anyone's curious, and genuinely interested in whether it holds up on vaults bigger than mine.

---

## If someone asks "how is this different from the built-in graph / other related-notes plugins"

Short answer to keep handy:

The graph view shows links you already made. This shows relationships you never
made, inferred from the text, and then uses your links as a second signal on top.
Other similarity plugins mostly stop at the first half; the measured jump from
0.66 to 0.75 recall is the second half doing work.
