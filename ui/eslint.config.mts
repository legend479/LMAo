import js from "@eslint/js";
import globals from "globals";
import tseslint from "typescript-eslint";
import pluginReact from "eslint-plugin-react";

export default tseslint.config(
  // Base JavaScript recommendations
  js.configs.recommended,
  // TypeScript recommended rules
  ...tseslint.configs.recommended,
  // React recommended rules
  pluginReact.configs.flat.recommended,
  // Project-specific configuration
  {
    files: ["**/*.{js,mjs,cjs,ts,mts,cts,jsx,tsx}"],
    languageOptions: {
      globals: globals.browser
    }
  }
);
