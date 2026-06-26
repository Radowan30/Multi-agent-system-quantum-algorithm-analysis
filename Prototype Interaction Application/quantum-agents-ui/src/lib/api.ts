/**
 * SSE client for the OpenAI-compatible chat-completions endpoint exposed by
 * agent_proxy.py. Yields token-level content deltas as the upstream stream
 * progresses; throws on network / HTTP / parse errors so the caller can
 * surface them in the UI.
 */
export async function* streamChat(
  baseUrl: string,
  modelName: string,
  qasm: string,
  signal?: AbortSignal,
): AsyncGenerator<string, void, unknown> {
  const url = `${baseUrl.replace(/\/+$/, "")}/v1/chat/completions`;
  const body = {
    model: modelName,
    messages: [{ role: "user", content: qasm }],
    stream: true,
  };

  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      // Some clients require Accept: text/event-stream to keep
      // intermediaries from buffering.
      Accept: "text/event-stream",
    },
    body: JSON.stringify(body),
    signal,
  });

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(
      `Upstream returned HTTP ${response.status}${text ? `: ${text.slice(0, 200)}` : ""}`,
    );
  }
  if (!response.body) {
    throw new Error("Upstream returned no response body.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE events are separated by a blank line ("\n\n"). The final partial
      // chunk (if any) is kept in the buffer for the next read.
      const events = buffer.split("\n\n");
      buffer = events.pop() ?? "";

      for (const event of events) {
        for (const line of event.split("\n")) {
          if (!line.startsWith("data:")) continue;
          const payload = line.slice(5).trim();
          if (payload === "[DONE]") return;
          if (!payload) continue;
          try {
            const chunk = JSON.parse(payload);
            const delta = chunk?.choices?.[0]?.delta?.content;
            if (typeof delta === "string" && delta.length > 0) {
              yield delta;
            }
          } catch {
            // Skip malformed SSE chunks rather than aborting the whole stream;
            // intermediate proxies sometimes emit keep-alive comments.
          }
        }
      }
    }
  } finally {
    try {
      reader.releaseLock();
    } catch {
      /* noop */
    }
  }
}

/**
 * Lightweight liveness probe — pings /v1/models and returns true if the
 * proxy is reachable and reports our advertised model. Used by the header
 * status indicator.
 */
export async function pingProxy(baseUrl: string): Promise<boolean> {
  try {
    const url = `${baseUrl.replace(/\/+$/, "")}/v1/models`;
    const r = await fetch(url, { method: "GET" });
    if (!r.ok) return false;
    const j = await r.json();
    return Array.isArray(j?.data) && j.data.length > 0;
  } catch {
    return false;
  }
}
