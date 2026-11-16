// The reader's model download, driven through the REAL src/reader/reader-assets.ts
// with a fake requestUrl that serves byte ranges from memory: pieces are joined in
// order, a partial file resumes where it stopped, a damaged partial file is caught
// by the checksum, a server that ignores ranges still works, and an unknown size
// is read from Content-Range.
//   node --experimental-transform-types --import ./bench/register-stub.mjs --test bench/reader-download-check.mjs
import assert from "node:assert/strict";
import { createHash, randomBytes } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, beforeEach, test } from "node:test";
import { __setRequestUrl } from "./obsidian-stub.mjs";

const home = mkdtempSync(join(process.env.TMPDIR || tmpdir(), "srn-download-"));
process.env.SRN_READER_HOME = home;
const { downloadGguf, modelPath, GGUF_CHUNK_BYTES } = await import("../src/reader/reader-assets.ts");

const rung = { id: "small", label: "Test model", repo: "test/repo", file: "test-model.gguf", sizeMb: 1, minRamGb: 0, judge: false, noThink: true };
const body = randomBytes(2 * GGUF_CHUNK_BYTES + 12345); // three pieces, the last one short
const sha = createHash("sha256").update(body).digest("hex");
const dest = modelPath(rung);
const part = `${dest}.part`;
const toArrayBuffer = (buf) => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);

function server({ pointerSize = true, ignoreRange = false } = {}) {
  const ranges = [];
  __setRequestUrl(async ({ url, headers = {} }) => {
    if (url.includes("/raw/main/")) {
      const text = `version https://git-lfs.github.com/spec/v1\noid sha256:${sha}\n${pointerSize ? `size ${body.length}\n` : ""}`;
      return { status: 200, headers: {}, text, arrayBuffer: toArrayBuffer(Buffer.from(text)) };
    }
    assert.ok(url.endsWith("/resolve/main/test-model.gguf"), `unexpected url ${url}`);
    const m = /^bytes=(\d+)-(\d+)$/.exec(headers.Range ?? "");
    if (ignoreRange || !m) return { status: 200, headers: {}, arrayBuffer: toArrayBuffer(body) };
    const start = Number(m[1]);
    const end = Math.min(Number(m[2]), body.length - 1);
    ranges.push([start, end]);
    if (start >= body.length) return { status: 416, headers: { "content-range": `bytes */${body.length}` }, arrayBuffer: new ArrayBuffer(0) };
    return {
      status: 206,
      headers: { "Content-Range": `bytes ${start}-${end}/${body.length}` },
      arrayBuffer: toArrayBuffer(body.subarray(start, end + 1)),
    };
  });
  return ranges;
}

beforeEach(() => {
  rmSync(join(home, "models"), { recursive: true, force: true });
  mkdirSync(join(home, "models"), { recursive: true });
});
after(() => rmSync(home, { recursive: true, force: true }));

test("a fresh download arrives in pieces and is complete", async () => {
  const ranges = server();
  await downloadGguf(rung, () => {});
  assert.ok(readFileSync(dest).equals(body));
  assert.equal(existsSync(part), false);
  assert.deepEqual(ranges.map(([s]) => s), [0, GGUF_CHUNK_BYTES, 2 * GGUF_CHUNK_BYTES]);
  assert.ok(ranges.every(([s, e]) => e - s + 1 <= GGUF_CHUNK_BYTES), "no piece is larger than the chunk size");
});

test("a partial download resumes where it stopped", async () => {
  const ranges = server();
  const have = 10 * 1024 * 1024;
  writeFileSync(part, body.subarray(0, have));
  await downloadGguf(rung, () => {});
  assert.ok(readFileSync(dest).equals(body));
  assert.equal(ranges[0][0], have);
});

test("a damaged partial file fails the checksum and is removed", async () => {
  server();
  const bad = Buffer.from(body.subarray(0, 1024 * 1024));
  bad[100] ^= 0xff;
  writeFileSync(part, bad);
  await assert.rejects(downloadGguf(rung, () => {}), /checksum mismatch/);
  assert.equal(existsSync(dest), false);
  assert.equal(existsSync(part), false);
});

test("a server that ignores ranges still delivers the file", async () => {
  server({ ignoreRange: true });
  await downloadGguf(rung, () => {});
  assert.ok(readFileSync(dest).equals(body));
});

test("an unknown size is taken from Content-Range", async () => {
  const ranges = server({ pointerSize: false });
  await downloadGguf(rung, () => {});
  assert.ok(readFileSync(dest).equals(body));
  assert.equal(ranges.length, 3);
});

test("an existing model is not downloaded again", async () => {
  const ranges = server();
  writeFileSync(dest, body);
  await downloadGguf(rung, () => {});
  assert.equal(ranges.length, 0);
});
