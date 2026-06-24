# Related Notes

A left-sidebar card stack of notes ranked by **local semantic-embedding similarity** to the note you're reading. Browse your vault by relevance instead of folders.

Everything runs locally: a small multilingual embedding model (German + English and 100+ other languages) runs in your machine's GPU (WebGPU) or CPU (WASM). The ONNX runtime (`.wasm`) is **shipped inside the plugin**, so the only network traffic is the **one-time** download of the model weights from the Hugging Face Hub; after that the model is cached and the plugin works fully offline.

## What it does

- Adds a **Related notes** view to the **left sidebar** (ribbon icon + command).
- Embeds every Markdown note (title + the first ~1500 characters of cleaned body text) into a 384-dimensional vector.
- For the active note, ranks all others by cosine similarity and shows the top matches as cards: **title**, muted **folder path**, a short **snippet**, and a **similarity %** pill.
- Click a card to open that note.
- Updates (debounced) when you switch notes, and re-embeds changed/created/renamed notes incrementally.
- While the index is still building, it falls back to a cheap keyword / tag / link-overlap ranking (shown with a `~` pill) so the panel is never empty.

## How it works

| Piece | Detail |
| --- | --- |
| Model | `Xenova/multilingual-e5-small` (default) — best German + English quality at a small size. Alternative: `Xenova/paraphrase-multilingual-MiniLM-L12-v2` (no query/passage prefix needed). |
| Engine | [`@huggingface/transformers`](https://www.npmjs.com/package/@huggingface/transformers) feature-extraction pipeline, mean-pooled + L2-normalized. Notes are embedded in batches in a single pipeline call. |
| Device | Auto-detects WebGPU and falls back to WASM (CPU). Pin one in settings. |
| ONNX runtime | The matching `onnxruntime-web` `.wasm`/`.mjs` files are copied out of the resolved dependency at build time and **shipped in the plugin's `ort/` folder**, so the WASM build always matches the bundled JS glue and works offline. The plugin points the runtime at that local folder via the vault's resource path. |
| Storage | Vectors persist as compact JSON in the plugin's config dir (`index.json`). Model weights cache in the browser Cache API. |

## Settings

- **Model** — embedding model id (dropdown of vetted choices, or paste a custom one).
- **Compute device** — Auto / WebGPU / WASM.
- **Number of results** — how many cards to show (default 12).
- **Minimum similarity** — hide matches below this cosine score (default 0.3).
- **Embed character limit** — how much of each note's body to embed (default 1500).
- **Excluded folders** — folders to leave out of the index.
- **Show snippet** — toggle the per-card text preview.
- **Rebuild index** — force a full re-embed (also available from the command palette and the panel's refresh icon).

Changing the model or compute device transparently rebuilds the index; unrelated setting changes (sliders, toggles) never trigger a re-embed.

## First run

On first launch the plugin downloads the model weights (~110 MB for the quantized WASM build) from the Hugging Face Hub. The ONNX runtime itself ships with the plugin, so only the model weights are fetched. This happens once; a progress notice is shown, and the weights are cached. If the model fails to load (no connection on first run, or a firewall/CSP blocking the Hub), the panel shows an error in its status line and a Notice, and degrades to keyword ranking until it can load — it never silently shows empty results.

## Requirements

- Desktop only (the model and WebGPU need a desktop Electron environment).
- Obsidian 1.7.2 or newer.

## Building

```bash
npm install
npm run build   # gen-ort -> tsc --noEmit -> esbuild production bundle (main.js + ort/)
npm run dev     # watch build with inline sourcemaps
npm run lint    # eslint (typescript-eslint + eslint-plugin-obsidianmd)
```

The build step (`gen-ort.mjs`) writes `src/ort-version.ts` (the pinned runtime version + CDN fallback URL) and copies the matching `onnxruntime-web` wasm assets into `ort/`. Both are build artifacts and are gitignored. The release workflow packages `main.js`, `manifest.json`, `styles.css`, and the `ort/` folder into `related-notes.zip`; for a manual install, extract that zip into `.obsidian/plugins/related-notes/`.

## License

MIT © 2026 Saiki77
