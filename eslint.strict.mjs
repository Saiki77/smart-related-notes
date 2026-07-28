import tsparser from "@typescript-eslint/parser";
import tseslint from "typescript-eslint";
export default [
  { ignores: ["main.js","ort/**","node_modules/**","src/ort-version.ts"] },
  ...tseslint.configs.strictTypeChecked,
  { files: ["**/*.ts","**/*.mjs"], languageOptions: { parser: tsparser, parserOptions: { project: "./tsconfig.json", allowDefaultProject: true } } },
];
