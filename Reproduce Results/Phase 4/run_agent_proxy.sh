#!/bin/bash
# Start the Phase 4 agent proxy in front of a vLLM server.
#
# This brings up two long-running processes:
#   1. vLLM serving the Phase 4 merged model on port 8000 (OpenAI-compatible)
#   2. agent_proxy.py listening on port 8080 (OpenAI-compatible facade in
#      front of the three-agent pipeline)
#
# Chat clients (Theia IDE, Open WebUI, etc.) should be pointed at port 8080.
#
# Both processes are killed on script exit (SIGINT/SIGTERM/normal).
#
# Run:
#   bash "Multi Agent System Phase 4/run_agent_proxy.sh"
#
# Optional environment overrides:
#   MERGED_MODEL  — path to a merged model dir (default: Phase 4 Gradient)
#   VLLM_PORT     — vLLM port (default 8000)
#   PROXY_PORT    — proxy port (default 8080)
#   MAX_MODEL_LEN — vLLM max context (default 150000)

set -euo pipefail

PROJ=/home/quantum-user/radowan/final_year_project
PHASE4_DIR="$PROJ/Multi Agent System Phase 4"
PY_EVAL="$PROJ/venv-eval/bin/python"

MERGED="${MERGED_MODEL:-$PROJ/saves/Llama-3-8B-Instruct-262k/merged/Phase4_alpha32}"
VLLM_PORT="${VLLM_PORT:-8000}"
PROXY_PORT="${PROXY_PORT:-8080}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-150000}"

VLLM_MODEL_NAME="Phase4_alpha32"
PROXY_MODEL_NAME="Phase4_Quantum_Agents"

VLLM_PID=""
PROXY_PID=""

cleanup() {
    echo
    echo "[run_agent_proxy] Shutting down…"
    if [[ -n "$PROXY_PID" ]] && kill -0 "$PROXY_PID" 2>/dev/null; then
        echo "[run_agent_proxy] Stopping proxy (PID $PROXY_PID)"
        kill -TERM "$PROXY_PID" 2>/dev/null || true
        wait "$PROXY_PID" 2>/dev/null || true
    fi
    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[run_agent_proxy] Stopping vLLM (PID $VLLM_PID)"
        kill -TERM "$VLLM_PID" 2>/dev/null || true
        # Allow vLLM ~10s to flush before SIGKILL.
        for i in {1..10}; do
            kill -0 "$VLLM_PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
    echo "[run_agent_proxy] Done."
}
trap cleanup EXIT INT TERM

# ── 1. Start vLLM ─────────────────────────────────────────────────────────────
echo "[run_agent_proxy] Starting vLLM on port $VLLM_PORT with $MERGED"
"$PY_EVAL" -m vllm.entrypoints.openai.api_server \
    --model "$MERGED" \
    --served-model-name "$VLLM_MODEL_NAME" \
    --host 127.0.0.1 --port "$VLLM_PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization 0.39 \
    --dtype bfloat16 \
    --enable-prefix-caching \
    --disable-uvicorn-access-log \
    > /tmp/agent_proxy_vllm.log 2>&1 &
VLLM_PID=$!
echo "[run_agent_proxy] vLLM PID: $VLLM_PID"

# Wait for vLLM /health to respond
echo "[run_agent_proxy] Waiting for vLLM to be ready…"
for i in {1..120}; do
    if curl -sf "http://127.0.0.1:$VLLM_PORT/health" > /dev/null 2>&1; then
        echo "[run_agent_proxy] vLLM ready (after ${i}s)"
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[run_agent_proxy] vLLM died during startup. See /tmp/agent_proxy_vllm.log"
        exit 1
    fi
    sleep 1
done

if ! curl -sf "http://127.0.0.1:$VLLM_PORT/health" > /dev/null 2>&1; then
    echo "[run_agent_proxy] vLLM did not become ready within 120s — aborting."
    exit 1
fi

# ── 2. Start agent proxy ──────────────────────────────────────────────────────
echo "[run_agent_proxy] Starting agent proxy on port $PROXY_PORT"
"$PY_EVAL" "$PHASE4_DIR/agent_proxy.py" \
    --vllm_url "http://127.0.0.1:$VLLM_PORT/v1" \
    --vllm_model "$VLLM_MODEL_NAME" \
    --proxy_model "$PROXY_MODEL_NAME" \
    --host 0.0.0.0 \
    --port "$PROXY_PORT" &
PROXY_PID=$!
echo "[run_agent_proxy] Proxy PID: $PROXY_PID"

cat <<EOF

================================================================
  Phase 4 multi-agent system is ready
================================================================
  vLLM    : http://127.0.0.1:$VLLM_PORT   (internal)
  Proxy   : http://0.0.0.0:$PROXY_PORT/v1  (point chat clients here)
  Model   : $PROXY_MODEL_NAME
================================================================

Connect your chat client (Theia IDE, Open WebUI, LibreChat, …) by setting
the OpenAI-compatible base URL to:

    http://<this-host>:$PROXY_PORT/v1

…and the model name to:

    $PROXY_MODEL_NAME

The proxy ignores any system prompt the client sends. Paste a Grover
circuit in OpenQASM 3.0 (raw or inside a markdown code fence) and the
proxy will run Agent 1 → Agent 2 → Agent 3 sequentially, streaming the
combined output back to the chat UI.

Press Ctrl-C to stop both processes.
EOF

# Wait on whichever child exits first; cleanup() handles the other.
wait -n
