import { useEffect, useRef, useState } from "react";
import { ChatMessage } from "./components/ChatMessage";
import { Dropzone } from "./components/Dropzone";
import { EmptyState } from "./components/EmptyState";
import { Header } from "./components/Header";
import { streamChat } from "./lib/api";
import { applyTheme, getInitialTheme, type Theme } from "./lib/theme";
import type { ApiSettings, Message } from "./lib/types";

// Empty baseUrl → relative URLs → fetched from the same origin as the UI.
// In dev (vite), the Vite proxy forwards /v1/* to http://localhost:8080
// where the agent_proxy backend listens.
const DEFAULT_SETTINGS: ApiSettings = {
  baseUrl: "",
  modelName: "Phase4_Quantum_Agents",
};
const SETTINGS_KEY = "quantum-agents:settings";

function loadSettings(): ApiSettings {
  try {
    const stored = localStorage.getItem(SETTINGS_KEY);
    if (stored) return { ...DEFAULT_SETTINGS, ...JSON.parse(stored) };
  } catch {
    /* ignore */
  }
  return DEFAULT_SETTINGS;
}

function newId(): string {
  return `m_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

export default function App() {
  const [settings, setSettings] = useState<ApiSettings>(loadSettings);
  const [theme, setTheme] = useState<Theme>(getInitialTheme);
  const [messages, setMessages] = useState<Message[]>([]);
  const [streaming, setStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement | null>(null);

  // Apply theme on mount + whenever it changes.
  useEffect(() => applyTheme(theme), [theme]);

  // Auto-scroll to bottom on new content.
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  // Persist settings.
  useEffect(() => {
    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
    } catch {
      /* ignore */
    }
  }, [settings]);

  const handleFile = async (file: File) => {
    if (streaming) return;
    const qasm = await file.text();

    const userMsg: Message = {
      id: newId(),
      role: "user",
      content: qasm,
      filename: file.name,
    };
    const asstId = newId();
    const asstMsg: Message = {
      id: asstId,
      role: "assistant",
      content: "",
      streaming: true,
    };
    setMessages((prev) => [...prev, userMsg, asstMsg]);
    setStreaming(true);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      for await (const delta of streamChat(
        settings.baseUrl,
        settings.modelName,
        qasm,
        controller.signal,
      )) {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === asstId ? { ...m, content: m.content + delta } : m,
          ),
        );
      }
      setMessages((prev) =>
        prev.map((m) => (m.id === asstId ? { ...m, streaming: false } : m)),
      );
    } catch (err: unknown) {
      // User-initiated stop — close the bubble cleanly without an error banner.
      const aborted =
        (err instanceof DOMException && err.name === "AbortError") ||
        (err instanceof Error && err.name === "AbortError");
      if (aborted) {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === asstId
              ? { ...m, streaming: false, content: m.content + "\n\n_Response stopped by user._" }
              : m,
          ),
        );
      } else {
        const message =
          err instanceof Error ? err.message :
          typeof err === "string" ? err : "Unknown error";
        setMessages((prev) =>
          prev.map((m) =>
            m.id === asstId
              ? { ...m, streaming: false, error: `Connection error — ${message}` }
              : m,
          ),
        );
      }
    } finally {
      setStreaming(false);
      abortRef.current = null;
    }
  };

  const handleStop = () => {
    abortRef.current?.abort();
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* Animated cosmic background — independent of layout */}
      <div className="cosmic-mesh">
        <span className="blob" />
      </div>

      <Header
        settings={settings}
        onSettingsChange={setSettings}
        theme={theme}
        onThemeChange={setTheme}
      />

      <main className="flex-1 max-w-5xl w-full mx-auto px-4 sm:px-6 pt-6 pb-6 flex flex-col gap-6 relative">
        <div className="flex-1 min-h-0">
          {messages.length === 0 ? (
            <EmptyState />
          ) : (
            <div className="space-y-7">
              {messages.map((m) => (
                <ChatMessage key={m.id} msg={m} />
              ))}
              <div ref={endRef} />
            </div>
          )}
        </div>

        <div className="sticky bottom-4 z-10">
          <Dropzone onFile={handleFile} disabled={streaming} onStop={handleStop} />
        </div>
      </main>
    </div>
  );
}
