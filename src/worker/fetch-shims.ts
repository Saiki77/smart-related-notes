// Fetch-API stand-ins for worker realms that lack them. Obsidian's in-app
// updates never replace the Electron runtime the INSTALLER shipped, so a
// current app can run on a years-old runtime whose worker scope has no
// fetch/Headers/Request/Response (seen in the wild on installer 1.6.7, app
// 1.13.x). transformers.js assumes they exist and dies constructing the model
// request. On runtimes that have the real globals, this module does nothing.
//
// Must stay dependency-free: it is evaluated inside the embed worker before
// @huggingface/transformers.

type G = Record<string, unknown>;
// self, not globalThis: this file runs in a worker realm (no window), and
// the popout-compatibility lint on globalThis does not apply there.
const g = self as unknown as G;

export const hadFetch = typeof g.fetch === "function";
export const hadHeaders = typeof g.Headers === "function";
export const hasCaches = typeof g.caches !== "undefined";

class HeadersShim {
  private readonly map = new Map<string, string>();

  constructor(init?: unknown) {
    if (init instanceof HeadersShim) {
      for (const [k, v] of init.map) this.map.set(k, v);
    } else if (Array.isArray(init)) {
      for (const pair of init as [string, string][]) this.append(pair[0], pair[1]);
    } else if (init && typeof init === "object") {
      for (const [k, v] of Object.entries(init as Record<string, string>)) this.append(k, v);
    }
  }

  append(name: string, value: string): void {
    const key = name.toLowerCase();
    const prev = this.map.get(key);
    this.map.set(key, prev !== undefined ? `${prev}, ${value}` : String(value));
  }

  set(name: string, value: string): void {
    this.map.set(name.toLowerCase(), String(value));
  }

  get(name: string): string | null {
    return this.map.get(name.toLowerCase()) ?? null;
  }

  has(name: string): boolean {
    return this.map.has(name.toLowerCase());
  }

  delete(name: string): void {
    this.map.delete(name.toLowerCase());
  }

  forEach(cb: (value: string, key: string, parent: HeadersShim) => void): void {
    for (const [k, v] of this.map) cb(v, k, this);
  }

  entries(): IterableIterator<[string, string]> {
    return this.map.entries();
  }

  keys(): IterableIterator<string> {
    return this.map.keys();
  }

  values(): IterableIterator<string> {
    return this.map.values();
  }

  [Symbol.iterator](): IterableIterator<[string, string]> {
    return this.map.entries();
  }
}

// A Response-shaped wrapper over one buffered body: what the bridge hands
// back, and enough surface for transformers.js' file loader (ok/status/
// headers/arrayBuffer/json/text/clone; body stays null, which its reader
// path treats as "no streaming, read whole").
export class BufferedResponse {
  readonly ok: boolean;
  readonly status: number;
  readonly statusText: string;
  readonly headers: HeadersShim;
  readonly url: string;
  readonly body: null = null;
  readonly redirected = false;
  readonly type = "basic";
  private readonly buf: ArrayBuffer;

  constructor(url: string, status: number, headers: Record<string, string>, buf: ArrayBuffer) {
    this.url = url;
    this.status = status;
    this.statusText = "";
    this.ok = status >= 200 && status < 300;
    this.headers = new HeadersShim(headers);
    this.buf = buf;
  }

  arrayBuffer(): Promise<ArrayBuffer> {
    return Promise.resolve(this.buf);
  }

  text(): Promise<string> {
    return Promise.resolve(new TextDecoder().decode(this.buf));
  }

  async json(): Promise<unknown> {
    return JSON.parse(await this.text());
  }

  blob(): Promise<unknown> {
    return Promise.resolve(new Blob([this.buf]));
  }

  clone(): BufferedResponse {
    return new BufferedResponse(
      this.url,
      this.status,
      Object.fromEntries(this.headers.entries()),
      this.buf,
    );
  }
}

class RequestShim {
  readonly url: string;
  readonly method: string;
  readonly headers: HeadersShim;

  constructor(input: unknown, init?: { method?: string; headers?: unknown }) {
    this.url =
      typeof input === "string"
        ? input
        : ((input as { url?: string }).url ?? String(input));
    this.method = init?.method ?? "GET";
    this.headers = new HeadersShim(init?.headers);
  }
}

if (!hadHeaders) g.Headers = HeadersShim;
if (typeof g.Request === "undefined") g.Request = RequestShim;
if (typeof g.Response === "undefined") g.Response = BufferedResponse;
