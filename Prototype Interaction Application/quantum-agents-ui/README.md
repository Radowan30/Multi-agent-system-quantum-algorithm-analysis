# Quantum Agents — UI

A chat-like React interface for the Phase 4 multi-agent system. Drop a
Grover circuit (`.qasm`) and watch the three agents reason through it
token-by-token.

The UI talks to the OpenAI-compatible proxy (`agent_proxy.py`) that wraps
the Phase 4 merged model — no IDE-side integration required, just point
the UI at the proxy's base URL.

## Setup

```bash
cd quantum-agents-ui
npm install
npm run dev
```

The dev server runs on `http://localhost:5173`. The UI defaults to talking
to a proxy at `http://localhost:8080`; you can change this at runtime via
the settings panel (gear icon in the header).

## Start the backend first

In a separate shell, launch the proxy + vLLM:

```bash
bash ../agent_proxy_backend/run_agent_proxy.sh
```

This boots vLLM on port 8000 (serving the Phase 4 merged model) and the
agent proxy on port 8080. Wait for "ready" lines in the console, then load
the UI.

See the top-level `README.md` of this folder for the full step-by-step setup
guide (model paths, Python venv, etc.).

## What you can do

- **Drag a `.qasm` file** onto the dropzone at the bottom of the page
  (or click to browse) — the file content is sent to the proxy, which
  runs Agent 1 → Agent 2 → Agent 3 and streams the combined reasoning
  back as one OpenAI chat completion.
- **No typing.** The chat is read-only beyond file uploads — the proxy
  only accepts Grover circuits in OpenQASM 3.0 anyway.
- **Settings panel** lets you override the proxy URL and the advertised
  model name if you've started the proxy with non-default arguments.

## Build

```bash
npm run build
```

Outputs a static SPA into `dist/`. Serve it from any static file host
(or `npm run preview` to inspect locally on port 5173).

## Files

- `src/App.tsx` — top-level state + page layout
- `src/components/` — Header, Dropzone, ChatMessage, EmptyState
- `src/lib/api.ts` — SSE streaming client for the OpenAI chat-completions
  endpoint
- `src/lib/types.ts` — Message / ApiSettings types
