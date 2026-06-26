import { Atom, FileCode, User } from "lucide-react";
import type { Message } from "../lib/types";
import { AgentResponse } from "./AgentResponse";

interface ChatMessageProps {
  msg: Message;
}

export function ChatMessage({ msg }: ChatMessageProps) {
  if (msg.role === "user") return <UserMessage msg={msg} />;
  return <AssistantMessage msg={msg} />;
}

function UserMessage({ msg }: { msg: Message }) {
  const kb = msg.content.length / 1024;
  const lineCount = msg.content.split("\n").length;
  return (
    <div className="flex gap-3 animate-fade-in">
      <Avatar>
        <User className="w-4 h-4 text-accent-violet" strokeWidth={2} />
      </Avatar>
      <div className="flex-1 min-w-0">
        <div className="text-xs text-text-muted mb-1.5 font-medium tracking-wide">You</div>
        <div className="card px-4 py-3">
          <div className="flex items-center gap-2 text-sm">
            <FileCode className="w-4 h-4 text-accent-violet flex-shrink-0" strokeWidth={1.8} />
            <span className="font-mono text-text-primary truncate">
              {msg.filename ?? "circuit.qasm"}
            </span>
            <span className="text-text-muted text-xs font-mono">
              {kb.toFixed(1)} KB · {lineCount} lines
            </span>
          </div>
          <details className="text-xs mt-2 group">
            <summary className="cursor-pointer text-text-muted hover:text-text-secondary transition select-none">
              <span className="group-open:hidden">Show source ↓</span>
              <span className="hidden group-open:inline">Hide source ↑</span>
            </summary>
            <pre className="mt-2.5 font-mono text-[0.78rem] text-text-secondary bg-bg-subtle rounded-lg p-3 overflow-x-auto border border-border-subtle max-h-80 leading-relaxed">
              {msg.content}
            </pre>
          </details>
        </div>
      </div>
    </div>
  );
}

function AssistantMessage({ msg }: { msg: Message }) {
  return (
    <div className="flex gap-3 animate-fade-in">
      <Avatar accent="cyan" pulsing={!!msg.streaming}>
        <Atom
          className={[
            "w-4 h-4 text-accent-cyan",
            msg.streaming ? "animate-spin-slow" : "",
          ].join(" ")}
          strokeWidth={1.8}
        />
      </Avatar>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1.5">
          <span className="text-xs text-text-muted font-medium tracking-wide">
            Quantum Agents
          </span>
          {msg.streaming && !msg.error && (
            <span className="flex items-center gap-1.5 text-[10px] uppercase tracking-[0.18em] text-accent-cyan">
              <span className="w-1.5 h-1.5 rounded-full bg-accent-cyan animate-pulse" />
              reasoning
            </span>
          )}
        </div>

        {/* Per-agent timeline */}
        {msg.content ? (
          <AgentResponse content={msg.content} streaming={!!msg.streaming} />
        ) : msg.streaming ? (
          <div className="card px-5 py-4 text-text-muted text-sm flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-accent-cyan animate-pulse" />
            Awaiting first tokens…
          </div>
        ) : null}

        {msg.error && (
          <div className="mt-3 px-3 py-2 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-500 dark:text-rose-300 text-xs">
            {msg.error}
          </div>
        )}
      </div>
    </div>
  );
}

function Avatar({
  children,
  accent = "violet",
  pulsing = false,
}: {
  children: React.ReactNode;
  accent?: "violet" | "cyan";
  pulsing?: boolean;
}) {
  const ring =
    accent === "cyan"
      ? "bg-accent-cyan/12 border-accent-cyan/30"
      : "bg-accent-violet/12 border-accent-violet/30";
  const glow =
    accent === "cyan"
      ? "shadow-[0_0_18px_-4px_rgb(var(--accent-cyan)/0.6)]"
      : "shadow-[0_0_18px_-4px_rgb(var(--accent-violet)/0.55)]";
  return (
    <div className="flex-shrink-0 mt-1">
      <div
        className={[
          "relative w-8 h-8 rounded-full border flex items-center justify-center",
          ring,
          pulsing ? glow : "",
        ].join(" ")}
      >
        {children}
      </div>
    </div>
  );
}
