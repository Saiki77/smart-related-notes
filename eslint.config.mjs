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
    // is isDesktopOnly): node builtins are required lazily at runtime, and the
    // engine bundle must be loaded with a REAL dynamic import() hidden from
    // esbuild's CJS lowering (it has top-level await; a lowered require() would
    // throw). Downloads use requestUrl like the rest of the plugin.
    files: ["src/reader/*.ts"],
    rules: {
      "@typescript-eslint/no-require-imports": "off",
      "@typescript-eslint/no-implied-eval": "off",
    },
  },
]);
