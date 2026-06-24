import tsparser from "@typescript-eslint/parser";
import { defineConfig } from "eslint/config";
import obsidianmd from "eslint-plugin-obsidianmd";

export default defineConfig([
  {
    ignores: ["main.js", "ort/**", "node_modules/**", "*.mjs", "src/ort-version.ts"],
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
]);
