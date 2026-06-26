import { Hexagon, Moon, Settings, Sun, Wifi, WifiOff } from "lucide-react";
import { useEffect, useState } from "react";
import { pingProxy } from "../lib/api";
import type { Theme } from "../lib/theme";
import type { ApiSettings } from "../lib/types";

interface HeaderProps {
  settings: ApiSettings;
  onSettingsChange: (s: ApiSettings) => void;
  theme: Theme;
  onThemeChange: (t: Theme) => void;
}

export function Header({ settings, onSettingsChange, theme, onThemeChange }: HeaderProps) {
  const [open, setOpen] = useState(false);
  const [connected, setConnected] = useState<boolean | null>(null);
  const [draft, setDraft] = useState(settings);

  // Probe the proxy every 8 s for the live connection indicator.
  useEffect(() => {
    let alive = true;
    const probe = async () => {
      const ok = await pingProxy(settings.baseUrl);
      if (alive) setConnected(ok);
    };
    probe();
    const t = setInterval(probe, 8000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [settings.baseUrl]);

  useEffect(() => setDraft(settings), [settings, open]);

  const save = () => {
    onSettingsChange(draft);
    setOpen(false);
  };

  return (
    <header className="sticky top-0 z-20 backdrop-blur-xl bg-bg-base/65 border-b border-border-subtle">
      <div className="max-w-5xl mx-auto px-6 py-3.5 flex items-center gap-4">
        {/* Logo + wordmark */}
        <a className="flex items-center gap-3 group" href="/">
          <div className="relative">
            <Hexagon
              className="w-7 h-7 text-accent-cyan"
              strokeWidth={1.6}
              fill="rgb(var(--accent-cyan) / 0.10)"
            />
            <div className="absolute inset-0 blur-lg bg-accent-cyan/25 rounded-full opacity-70 group-hover:opacity-100 transition-opacity" />
          </div>
          <div className="leading-tight">
            <h1 className="text-base font-display font-semibold tracking-tight">
              <span className="bg-gradient-to-r from-accent-cyan via-accent-violet to-accent-amber bg-clip-text text-transparent">
                Quantum Agents
              </span>
            </h1>
            <p className="text-[11px] text-text-muted tracking-wide">
              Symbolic analysis of Grover circuits
            </p>
          </div>
        </a>

        <div className="flex-1" />

        {/* Connection status */}
        <div
          className="flex items-center gap-2 text-xs text-text-secondary px-2.5 py-1.5 rounded-full bg-bg-subtle/60 border border-border-subtle"
          title={connected ? "Proxy reachable" : connected === false ? "Proxy offline" : "Checking…"}
        >
          {connected === null ? (
            <span className="w-1.5 h-1.5 rounded-full bg-text-muted animate-pulse" />
          ) : connected ? (
            <>
              <span className="relative flex w-1.5 h-1.5">
                <span className="absolute inset-0 rounded-full bg-emerald-400 opacity-60 animate-ping" />
                <span className="relative rounded-full w-1.5 h-1.5 bg-emerald-400" />
              </span>
              <Wifi className="w-3.5 h-3.5 text-emerald-400" />
              <span className="hidden sm:inline">Connected</span>
            </>
          ) : (
            <>
              <span className="w-1.5 h-1.5 rounded-full bg-rose-400" />
              <WifiOff className="w-3.5 h-3.5 text-rose-400" />
              <span className="hidden sm:inline">Offline</span>
            </>
          )}
        </div>

        {/* Theme toggle */}
        <button
          onClick={() => onThemeChange(theme === "dark" ? "light" : "dark")}
          className="btn-ghost"
          title={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
          aria-label="Toggle theme"
        >
          {theme === "dark" ? (
            <Sun className="w-4 h-4 text-accent-amber" />
          ) : (
            <Moon className="w-4 h-4 text-accent-violet" />
          )}
        </button>

        {/* Settings */}
        <button
          onClick={() => setOpen(!open)}
          className="btn-ghost"
          title="API settings"
          aria-label="API settings"
        >
          <Settings className="w-4 h-4" />
        </button>
      </div>

      {/* Settings panel */}
      {open && (
        <div className="border-t border-border-subtle bg-bg-panel/85 backdrop-blur-xl animate-fade-in">
          <div className="max-w-5xl mx-auto px-6 py-5 grid sm:grid-cols-2 gap-4">
            <label className="block text-xs">
              <span className="block text-text-secondary mb-1.5 font-medium">
                Proxy base URL{" "}
                <span className="text-text-muted font-normal">
                  (leave blank to use the Vite proxy)
                </span>
              </span>
              <input
                type="text"
                value={draft.baseUrl}
                onChange={(e) => setDraft({ ...draft, baseUrl: e.target.value })}
                placeholder="(blank = same origin) e.g. http://localhost:8080"
                className="input"
              />
            </label>
            <label className="block text-xs">
              <span className="block text-text-secondary mb-1.5 font-medium">Model name</span>
              <input
                type="text"
                value={draft.modelName}
                onChange={(e) => setDraft({ ...draft, modelName: e.target.value })}
                placeholder="Phase4_Quantum_Agents"
                className="input"
              />
            </label>
            <div className="sm:col-span-2 flex justify-end gap-2 pt-1">
              <button onClick={() => setOpen(false)} className="btn-ghost text-sm">
                Cancel
              </button>
              <button onClick={save} className="btn-primary">
                Save
              </button>
            </div>
          </div>
        </div>
      )}
    </header>
  );
}
