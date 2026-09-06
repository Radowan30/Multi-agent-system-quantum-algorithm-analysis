# Prototype Interaction Application

A self-contained chat-style interface for the Phase 4 multi-agent system from
the thesis *"LLM Based Multi-Agent System for Improving Quantum Algorithm Symbolic Analysis"*.

You drop a Grover-algorithm QASM circuit into the upload zone, and the
interface streams back the live per-agent reasoning of the three specialised
agents (Oracle Extractor → Marked-State Identifier → Probability Distribution),
finishing with the final probability distribution. Long agent outputs are
truncated during streaming behind a **Show more** button and collapsed to a
one-line "Analysis completed" summary once finished, and the whole session
can be exported as a professionally formatted PDF report with one click.

Everything in this folder runs **locally on your own computer**. Nothing is
sent off-machine; the model, the proxy, and the UI all communicate over
`localhost`.

## Folder layout

```
Prototype Interaction Application/
├── README.md                          — this file (start here)
├── agent_proxy_backend/               — OpenAI-compatible facade in front of vLLM
│   ├── agent_proxy.py                 — FastAPI server (port 8080)
│   ├── prompts.py                     — minimal Phase 4 prompt builders
│   ├── parsers.py                     — output parsers for the three agents
│   └── run_agent_proxy.sh             — convenience launcher (vLLM + proxy together)
└── quantum-agents-ui/                 — React chat-style frontend
    ├── package.json                   — Node dependencies
    ├── vite.config.ts                 — dev-server config (proxies /v1 → :8080)
    ├── src/                           — TypeScript + React sources
    └── README.md                      — frontend-specific notes
```

## How the three pieces fit together

```
              Browser (you)
                  │
                  ▼  HTTP :5173
        quantum-agents-ui  ── Vite dev server / static build
                  │  (the Vite proxy forwards every /v1/* call)
                  ▼  HTTP :8080
              agent_proxy.py  ── FastAPI, OpenAI-compatible facade
                  │  (runs the three-agent chain internally)
                  ▼  HTTP :8000
                 vLLM   ── serves the merged Phase 4 model on the GPU
```

## Prerequisites

### Hardware

- **NVIDIA GPU** with ≥ 80 GB VRAM (the Phase 4 merged model needs ~30 GB at
  rest plus headroom for the long-context KV-cache). The thesis ran on an
  RTX PRO 6000 Blackwell 96 GB.
- **≥ 32 GB system RAM** minimum for vLLM + the agent proxy; **64 GB recommended** for headroom alongside a desktop session. Host-RAM use per vLLM instance is roughly 25 – 35 GB; the model weights themselves live on the GPU.

### Software

- **Linux** (the thesis used Ubuntu 24.04 LTS inside WSL2; native Linux works
  the same).
- **NVIDIA driver** supporting CUDA 12.8 (driver ≥ 555).
- **Python 3.11** with a working pip.
- **Node.js 22** with npm.
- A **fine-tuned Phase 4 merged model** sitting on disk. Build one yourself by
  following the steps in `Reproduce Results/README.md` (Phase 4 section) of
  this repository, or point at any other merged checkpoint you already have.

## 1. Set up the Python environment for the backend

Create a Python virtual environment for the proxy + vLLM, using the same
pinned dependencies as our `venv-inference` (full requirements file ships
inside `Reproduce Results/envs/venv-inference_requirements.txt`):

```bash
python3.11 -m venv venv-inference
source venv-inference/bin/activate
pip install --upgrade pip wheel setuptools
pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
pip install -r ../Reproduce\ Results/envs/venv-inference_requirements.txt
deactivate
```

Verify it works:

```bash
./venv-inference/bin/python -c "import torch, vllm; print(torch.__version__, vllm.__version__)"
# expected: 2.11.0+cu128 0.20.2
```

## 2. Configure the backend launcher

Open `agent_proxy_backend/run_agent_proxy.sh` and adjust the three environment
variables at the top to match your local paths:

```bash
PROJ=<path to the folder containing this repo>          # parent dir
PHASE4_DIR=<absolute path to agent_proxy_backend>       # this folder
PY_EVAL=<absolute path to your venv-inference/bin/python>
```

You can also override these from the command line:

```bash
MERGED_MODEL=/abs/path/to/your/Phase4_alpha32 \
PROXY_PORT=8080 VLLM_PORT=8000 \
bash agent_proxy_backend/run_agent_proxy.sh
```

The script defaults assume `MERGED_MODEL` points at a Phase 4 merged
checkpoint produced by `Reproduce Results/Phase 4`; override it if yours
lives elsewhere.

## 3. Start the backend

Open a terminal and run:

```bash
bash agent_proxy_backend/run_agent_proxy.sh
```

This brings up two long-running processes:

1. **vLLM** on `http://127.0.0.1:8000` — serves the Phase 4 merged model directly.
2. **agent_proxy** on `http://0.0.0.0:8080` — the multi-agent facade.

Wait until the console prints the "Phase 4 multi-agent system is ready" banner
(typically 30 – 90 seconds after vLLM starts loading weights).

Press **Ctrl-C** in this terminal to stop both processes cleanly when done.

## 4. Set up the frontend

In a **second terminal**:

```bash
cd quantum-agents-ui
npm install        # one-time — installs React, Vite, Tailwind, etc.
npm run dev
```

Vite will print a line such as:

```
  ➜  Local:   http://localhost:5173/
```

Open that URL in your browser.

## 5. Use the interface

- **Drag a `.qasm` file** onto the dropzone at the bottom of the page (or
  click to browse). The file contents are sent to the proxy, which runs
  Agent 1 → Agent 2 → Agent 3 sequentially and streams the combined per-agent
  reasoning back as one OpenAI-style chat completion.
- **No typing required.** The interface is upload-driven; the proxy only
  accepts Grover circuits in OpenQASM 3.0 anyway.
- **Progressive show / collapse of each agent's output.** While an agent is
  streaming, only the first few lines of its output are shown alongside a
  **Show more** button so the page never grows into a long scroll. Once the
  agent has finished, its section collapses to a single "Analysis completed"
  strip with an **Expand** button; click **Expand** to see the full trace
  and **Collapse** to fold it back. If you expanded an agent mid-stream, it
  stays open after streaming ends, with a **Collapse** button ready.
- **Download a session report as PDF.** The file-download icon in the header
  becomes active as soon as at least one circuit has been analysed. Clicking
  it opens an in-page preview of a professionally formatted PDF report that
  includes, for every uploaded circuit in the current session, the filename,
  the number of qubits, the number of marked states (and the state strings
  themselves), and the raw output of each of the three agents in its per-
  agent colour. The preview and the downloaded file are byte-identical.
- **Stop response.** A stop button appears in the dropzone area while a
  response is streaming; click it to cancel the in-flight request mid-stream.
- **Settings panel** (gear icon in the header). Override the proxy base URL
  and the advertised model name if you started the proxy on non-default
  ports.
- **Theme toggle** (sun / moon icon in the header). Switch between light and
  dark themes; the choice is remembered across page loads.

## 6. Optional — build the UI for production

```bash
cd quantum-agents-ui
npm run build       # outputs to dist/
npm run preview     # serves dist/ on http://localhost:5173 for inspection
```

You can then deploy `dist/` to any static-file host on your local network. The
UI uses **relative URLs** for the chat endpoint, so as long as it is served
from the same origin as the proxy (or via a same-origin reverse proxy), it
will Just Work.

## Troubleshooting

- **vLLM does not start (CUDA OOM).** Lower `--gpu-memory-utilization` in
  `run_agent_proxy.sh` (e.g. from 0.90 → 0.80), or reduce `MAX_MODEL_LEN`.
- **`npm install` fails on a non-Linux host.** Make sure you are using Node 22
  and a clean working tree. Delete `node_modules` and `package-lock.json` and
  retry if needed.
- **The UI loads but shows "Offline".** The proxy is not reachable. Confirm
  step 3 succeeded (check `/tmp/agent_proxy_vllm.log` for vLLM startup
  errors), and confirm port 8080 is not blocked or taken by another process.
- **Requests time out for large circuits.** Increase `MAX_MODEL_LEN` in
  `run_agent_proxy.sh` if your GPU has enough VRAM.

## What this prototype is **not**

This is a research-grade local prototype for booth demos and personal
experimentation. It is intentionally minimal:

- No authentication. Anyone with access to the proxy port can use the model.
- No multi-tenant support — one user, one model, one GPU.
- No production hardening — the proxy is designed for local use.

If you want to expose this beyond your own machine, set up the appropriate
network controls and authentication layer yourself.
