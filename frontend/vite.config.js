import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies /api to the FastAPI backend (override with VITE_API_PROXY).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
    proxy: { "/api": { target: process.env.VITE_API_PROXY || "http://localhost:8000", changeOrigin: true } },
  },
});
