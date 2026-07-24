import js from "@eslint/js";
import { FlatCompat } from "@eslint/eslintrc";

const compat = new FlatCompat({
  baseDirectory: import.meta.dirname
});

const config = [
  {
    ignores: [".next/**", "node_modules/**", "coverage/**"]
  },
  js.configs.recommended,
  ...compat.extends("next/core-web-vitals"),
  {
    rules: {
      "no-unused-vars": "off",
      "react/no-unescaped-entities": "off"
    }
  }
];

export default config;
