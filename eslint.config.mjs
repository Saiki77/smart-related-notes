import tsparser from "@typescript-eslint/parser";
import { defineConfig } from "eslint/config";
import obsidianmd from "eslint-plugin-obsidianmd";

export default defineConfig([
  {
    ignores: ["main.js", "ort/**", "node_modules/**", "*.mjs", "src/reader/engine-entry.mjs", "src/ort-version.ts"],
  },
  ...obsidianmd.configs.recommended,
  {
    files: ["**/*.ts"],
    languageOptions: {
      parser: tsparser,
      parserOptions: { project: "./tsconfig.json" },
    },
    rules: {
      // Product copy ("Related notes", "WebGPU", "WASM", "Auto") is deliberate and
      // would be mangled by enforced sentence-case, the way Hub Sidebar disables it.
      "obsidianmd/ui/sentence-case": "off",
    },
  },
  {
    // The reader subsystem is desktop-only Node territory by design (the plugin
    // is isDesktopOnly): node builtins are required lazily at runtime, downloads
    // stream via fetch because requestUrl buffers whole responses (a 5 GB GGUF
    // cannot live in RAM), and the engine bundle must be loaded with a REAL
    // dynamic import() hidden from esbuild's CJS lowering (it has top-level
    // await; a lowered require() would throw).
    files: ["src/reader/*.ts"],
    rules: {
      "@typescript-eslint/no-require-imports": "off",
      "no-restricted-globals": "off",
      "@typescript-eslint/no-implied-eval": "off",
    },
  },
]);
