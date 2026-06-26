import { Binary, Calculator, Crosshair } from "lucide-react";
import { useMemo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface AgentResponseProps {
  /** Full streamed assistant content from the proxy (markdown). */
  content: string;
  /** True while still receiving tokens; used to mark the last section as active. */
  streaming: boolean;
}

type SectionKind = "agent" | "orphan";

interface ParsedSection {
  kind: SectionKind;
  /** 1 | 2 | 3 — null for orphan, or when the header hasn't streamed in yet. */
  agentNum: 1 | 2 | 3 | null;
  /** "Oracle Extractor" etc. — null until the header has streamed in. */
  title: string | null;
  /** Markdown body without the leading header. */
  body: string;
}

const AGENT_META: Record<1 | 2 | 3, { icon: typeof Binary; label: string }> = {
  1: { icon: Crosshair,  label: "Oracle Extractor" },
  2: { icon: Binary,     label: "Marked-State Identifier" },
  3: { icon: Calculator, label: "Probability Distribution" },
};

// **Agent N reasoning (Title):**\n\n
const HEADER_RE = /^\s*\*\*\s*Agent\s+(\d+)\s+reasoning\s*\(([^)]+)\)\s*:\s*\*\*\s*\n+/i;
// Marker the trained agents emit to signal the end of their structured output.
// Anything the model continues to emit after this is post-stop drift (repeated
// examples, echoed gate definitions, etc.) and must not be shown to the user.
const END_MARKER_RE = /^[ \t]*===\s*END\s*===[ \t]*$/im;

function truncateAtEndMarker(body: string): string {
  const m = body.match(END_MARKER_RE);
  if (m && m.index !== undefined) {
    return body.slice(0, m.index).trimEnd();
  }
  return body;
}

function parseSections(content: string): ParsedSection[] {
  if (!content) return [];
  // Split on markdown horizontal rule between agents.
  const raw = content.split(/\n+---\n+/);
  const out: ParsedSection[] = [];
  for (const chunk of raw) {
    const m = chunk.match(HEADER_RE);
    if (!m) {
      out.push({ kind: "orphan", agentNum: null, title: null, body: truncateAtEndMarker(chunk) });
      continue;
    }
    const num = parseInt(m[1], 10);
    const agentNum = (num === 1 || num === 2 || num === 3) ? num as 1 | 2 | 3 : null;
    const title = m[2].trim();
    const body = truncateAtEndMarker(chunk.slice(m[0].length));
    out.push({ kind: "agent", agentNum, title, body });
  }
  return out;
}

function rewriteEqualsHeaders(md: string): string {
  // Turn the proxy's `=== Some Section ===` markers into markdown ### so they
  // render as styled small-caps section headers via our .md h3 style.
  return md.replace(/^[ \t]*===\s*(.+?)\s*===[ \t]*$/gm, "### $1");
}

export function AgentResponse({ content, streaming }: AgentResponseProps) {
  const sections = useMemo(() => parseSections(content), [content]);
  return (
    <div className="space-y-5">
      {sections.map((s, i) => {
        const isLast = i === sections.length - 1;
        const isActive = streaming && isLast;
        return <Section key={i} section={s} active={isActive} />;
      })}
    </div>
  );
}

function Section({ section, active }: { section: ParsedSection; active: boolean }) {
  // Pre-process the body once.
  const body = useMemo(() => rewriteEqualsHeaders(section.body.trim()), [section.body]);

  // No header yet → just stream raw content with a subtle placeholder.
  if (section.kind === "orphan" || section.agentNum === null) {
    return (
      <div className="px-4 py-3 rounded-xl bg-bg-subtle border border-border-subtle">
        <article className="md">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{body || "…"}</ReactMarkdown>
        </article>
      </div>
    );
  }

  const meta = AGENT_META[section.agentNum];
  const Icon = meta.icon;

  return (
    <section
      data-agent={section.agentNum}
      className={[
        "relative pl-5 sm:pl-6 animate-fade-in",
        "agent-rail",
      ].join(" ")}
    >
      {/* Number badge on the rail */}
      <div
        className={[
          "absolute -left-[15px] top-0 w-7 h-7 rounded-full",
          "flex items-center justify-center text-xs font-display font-bold",
          "border bg-bg-base agent-text agent-border",
          active ? "agent-pulse" : "",
        ].join(" ")}
      >
        {section.agentNum}
      </div>

      {/* Header */}
      <header className="flex items-center gap-3 mb-3">
        <span className="chip agent-bg agent-text agent-border border">
          <Icon className="w-3 h-3" strokeWidth={2.2} />
          Agent {section.agentNum}
        </span>
        <h2 className="font-display font-semibold text-text-primary text-[0.95rem] tracking-tight">
          {meta.label}
          <span className="ml-2 text-text-muted font-sans font-normal text-xs">
            · {section.title}
          </span>
        </h2>
        {active && (
          <span className="ml-auto flex items-center gap-1.5 text-[10px] uppercase tracking-[0.18em] text-text-muted">
            <span className="w-1.5 h-1.5 rounded-full agent-bg agent-pulse" />
            streaming
          </span>
        )}
      </header>

      {/* Body */}
      <div
        className={[
          "card px-5 py-4",
          active ? "agent-glow" : "",
        ].join(" ")}
      >
        <article className="md">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {body || (active ? "…" : "")}
          </ReactMarkdown>
        </article>
      </div>
    </section>
  );
}
