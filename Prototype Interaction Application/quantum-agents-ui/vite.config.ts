import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies /v1/* requests to the locally-running agent_proxy
// FastAPI server. This lets the React app use relative URLs that resolve
// against the same origin, while the model server stays bound to localhost.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
    proxy: {
      "/v1": {
        target: "http://localhost:8080",
        changeOrigin: true,
        // SSE streaming responses must not be buffered — Vite's default
        // http-proxy handles streaming bodies correctly as long as we
        // don't enable response transforms.
      },
    },
  },
});
