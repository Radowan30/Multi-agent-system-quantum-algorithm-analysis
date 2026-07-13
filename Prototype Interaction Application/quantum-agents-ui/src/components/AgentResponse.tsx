import { Binary, Calculator, CheckCircle2, ChevronDown, ChevronUp, Crosshair } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
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

  // Per-section expand state — persists across streaming/complete transitions.
  // • false → default view (truncated during streaming; "Analysis completed"
  //   after streaming). This is the initial state.
  // • true  → user has manually expanded — stay expanded even when streaming
  //   completes; user must click Collapse to return to the compact view.
  const [manuallyExpanded, setManuallyExpanded] = useState(false);

  // Ref + effect to check whether the rendered body actually overflows the
  // truncation cap. If it fits inside the cap there's no need for a
  // "Show more" button.
  const bodyRef = useRef<HTMLDivElement | null>(null);
  const [overflowing, setOverflowing] = useState(false);
  useEffect(() => {
    if (!bodyRef.current) return;
    const el = bodyRef.current;
    // scrollHeight vs clientHeight tells us if content is being clipped.
    setOverflowing(el.scrollHeight - el.clientHeight > 4);
  }, [body, manuallyExpanded, active]);

  // Orphan sections (before agent header parses in) use a much simpler card.
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

  // Determine which of the four rendering modes this section is in:
  //   streaming + not expanded → showing preview (first ~12 lines), Show more
  //   streaming + expanded     → showing full stream, no toggle
  //   !streaming + expanded    → showing full output, Collapse button
  //   !streaming + not expanded → showing "Analysis completed", Expand button
  const showCompactSummary = !active && !manuallyExpanded;
  const showFull = manuallyExpanded;
  const showTruncated = active && !manuallyExpanded;

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

      {showCompactSummary ? (
        // ── Compact "Analysis completed" strip with an Expand button ─────
        <button
          type="button"
          onClick={() => setManuallyExpanded(true)}
          className={[
            "w-full flex items-center gap-3 px-4 py-3",
            "card hover:bg-bg-subtle/60 transition text-left",
          ].join(" ")}
          aria-expanded={false}
          aria-label="Expand agent output"
        >
          <CheckCircle2
            className="w-4 h-4 agent-text flex-shrink-0"
            strokeWidth={2}
          />
          <span className="text-sm text-text-primary flex-1">
            Analysis completed
          </span>
          <span className="text-xs text-text-muted flex items-center gap-1">
            Expand <ChevronDown className="w-3.5 h-3.5" strokeWidth={2} />
          </span>
        </button>
      ) : (
        // ── Full or truncated body ──────────────────────────────────────
        <div
          className={[
            "card px-5 py-4 overflow-hidden",
            active ? "agent-glow" : "",
          ].join(" ")}
        >
          <div
            ref={bodyRef}
            className={[
              "transition-[max-height] duration-200",
              showTruncated
                ? "max-h-[16rem] overflow-hidden relative"
                : "",
            ].join(" ")}
            style={
              showTruncated
                ? {
                    // Soft gradient fade so the truncation is visually clear
                    // without a hard cut across a line of text.
                    WebkitMaskImage:
                      "linear-gradient(to bottom, black 78%, transparent)",
                    maskImage:
                      "linear-gradient(to bottom, black 78%, transparent)",
                  }
                : undefined
            }
          >
            <article className="md">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {body || (active ? "…" : "")}
              </ReactMarkdown>
            </article>
          </div>

          {/* Trailing control row: Show more (while streaming) / Collapse
              (once streaming has finished). */}
          {showTruncated && overflowing && (
            <div className="mt-3 pt-3 border-t border-border-subtle flex justify-center">
              <button
                type="button"
                onClick={() => setManuallyExpanded(true)}
                className="text-xs text-text-secondary hover:text-text-primary transition flex items-center gap-1.5 px-3 py-1 rounded-full hover:bg-bg-subtle"
                aria-label="Show full agent output"
              >
                Show more <ChevronDown className="w-3.5 h-3.5" strokeWidth={2} />
              </button>
            </div>
          )}
          {showFull && !active && (
            <div className="mt-3 pt-3 border-t border-border-subtle flex justify-center">
              <button
                type="button"
                onClick={() => setManuallyExpanded(false)}
                className="text-xs text-text-secondary hover:text-text-primary transition flex items-center gap-1.5 px-3 py-1 rounded-full hover:bg-bg-subtle"
                aria-label="Collapse agent output"
              >
                Collapse <ChevronUp className="w-3.5 h-3.5" strokeWidth={2} />
              </button>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
