"""
Helper for managing a `vllm serve` subprocess from run_eval.py.

`vllm serve` launches vLLM's OpenAI-compatible HTTP API server, which internally
runs AsyncLLMEngine. This is the production-grade async-native vLLM deployment
pattern, and is what both Phase 3 orchestrators talk to over HTTP. It avoids
the deadlock problem inherent to wrapping the sync `vllm.LLM` class in
`asyncio.to_thread`.

Public API:
    async with VLLMServer(model_path, ...) as server:
        # server.base_url is the URL of the live server
        # server.served_model_name is the name vLLM exposed it under
        ...
    # subprocess is terminated when the `async with` block exits.
"""

import asyncio
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx


def _pick_free_port(start: int = 8000) -> int:
    """Find a free TCP port starting from `start`."""
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in range [{start}, {start+100})")


class VLLMServer:
    """Manages a `vllm serve` subprocess as a context manager.

    Usage:
        async with VLLMServer(model_path="/path/to/model", ...) as server:
            base_url = server.base_url  # e.g. "http://127.0.0.1:8000/v1"
            served_model_name = server.served_model_name
            # ... call the server ...
    """

    def __init__(
        self,
        model_path: str,
        served_model_name: str = None,
        max_model_len: int = 122880,
        gpu_memory_utilization: float = 0.90,
        dtype: str = "bfloat16",
        port: int = None,
        host: str = "127.0.0.1",
        startup_timeout_s: float = 600.0,
    ):
        self.model_path = model_path
        self.served_model_name = served_model_name or Path(model_path).name
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.port = port or _pick_free_port(8000)
        self.host = host
        self.startup_timeout_s = startup_timeout_s
        self.proc = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    async def __aenter__(self):
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--served-model-name", self.served_model_name,
            "--host", self.host,
            "--port", str(self.port),
            "--max-model-len", str(self.max_model_len),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--dtype", self.dtype,
            "--enable-prefix-caching",
            # Suppress request access logs (one per LLM call would be very
            # noisy across 100s of agent calls per run).
            "--disable-uvicorn-access-log",
        ]
        print(f"[vllm_server] Starting: {' '.join(cmd)}", flush=True)
        # Capture stderr to a log file so startup failures are diagnosable.
        self._stderr_log = open(f"/tmp/vllm_serve_{self.port}.stderr.log", "w")
        # Spawn in its own process group so we can SIGKILL the whole group on
        # teardown (vllm forks workers that ignore the parent's SIGTERM).
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=self._stderr_log,
            start_new_session=True,
            env={**os.environ, "VLLM_LOGGING_LEVEL": "WARNING"},
        )
        # Poll /health until ready, or fail on timeout.
        health_url = f"http://{self.host}:{self.port}/health"
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=5.0) as client:
            while True:
                if self.proc.poll() is not None:
                    self._stderr_log.flush()
                    log_path = self._stderr_log.name
                    try:
                        tail = open(log_path).read()[-3000:]
                    except Exception:
                        tail = "(could not read stderr log)"
                    raise RuntimeError(
                        f"vllm serve died during startup (exit code "
                        f"{self.proc.returncode}). stderr tail:\n{tail}"
                    )
                try:
                    r = await client.get(health_url)
                    if r.status_code == 200:
                        elapsed = time.perf_counter() - t0
                        print(f"[vllm_server] Ready in {elapsed:.1f}s at {self.base_url}",
                              flush=True)
                        return self
                except (httpx.ConnectError, httpx.ReadTimeout):
                    pass
                if time.perf_counter() - t0 > self.startup_timeout_s:
                    self._terminate()
                    raise TimeoutError(
                        f"vllm serve did not become ready in {self.startup_timeout_s}s"
                    )
                await asyncio.sleep(2.0)

    async def __aexit__(self, exc_type, exc, tb):
        self._terminate()

    def _terminate(self):
        if self.proc is None:
            return
        if self.proc.poll() is not None:
            return  # already dead
        print(f"[vllm_server] Terminating PID {self.proc.pid} ...", flush=True)
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            self.proc.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            print("[vllm_server] SIGTERM timed out, sending SIGKILL", flush=True)
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.proc.wait(timeout=15.0)
        print(f"[vllm_server] Stopped.", flush=True)
