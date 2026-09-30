import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],

  // Vite options tailored for Tauri development
  clearScreen: false,

  // Tauri expects a fixed port, fail if that port is not available
  server: {
    port: 5173,
    strictPort: true,
    watch: {
      // tell vite to ignore watching `src-tauri` and the Cargo `target` dir.
      // The workspace `target/` lives at the project root and contains
      // `*.dll` artifacts that get locked by cargo during builds, which
      // crashes Vite's file watcher with EBUSY on Windows.
      ignored: ["**/src-tauri/**", "**/target/**"],
    },
  },

  // Env variables starting with VITE_ are exposed to the frontend
  envPrefix: ["VITE_", "TAURI_"],

  build: {
    // Tauri uses Chromium on Windows and WebKit on macOS and Linux
    target: process.env.TAURI_PLATFORM == "windows" ? "chrome105" : "safari13",
    // don't minify for debug builds
    minify: !process.env.TAURI_DEBUG ? "esbuild" : false,
    // produce sourcemaps for debug builds
    sourcemap: !!process.env.TAURI_DEBUG,
    outDir: "dist",
    rollupOptions: {
      output: {
        // Split heavy vendors into separate chunks so the entry chunk
        // parses fast on first launch
        manualChunks: {
          react: ["react", "react-dom"],
          fluentui: ["@fluentui/react-components", "@fluentui/react-icons"],
        },
      },
    },
  },
});
