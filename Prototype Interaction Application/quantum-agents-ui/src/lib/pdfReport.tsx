/**
 * PDF report generator for the Quantum Agents session.
 *
 * The report contains, for every uploaded QASM circuit in the current
 * session:
 *   • header with filename, qubit count and number of marked states,
 *   • the raw output of each of the three agents (Oracle Extractor,
 *     Marked-State Identifier, Probability Distribution).
 *
 * A single PDF document is generated once and re-used for both the in-page
 * preview and the download, so the two are byte-identical by construction.
 */
import {
  Document,
  Page,
  StyleSheet,
  Text,
  View,
} from "@react-pdf/renderer";
import type { Message } from "./types";

// ─── PDF styling ────────────────────────────────────────────────────────
// Palette chosen to match the on-screen per-agent semantic colours (cyan /
// violet / amber) while staying legible in print.
const COLORS = {
  ink:      "#1a1a2e",
  secondary:"#4a4a5f",
  muted:    "#8a8a9b",
  border:   "#dbdbe5",
  bgSubtle: "#f5f5fa",
  agent1:   "#087aa2",  // cyan
  agent2:   "#5d3a90",  // violet
  agent3:   "#a86800",  // amber
};

const styles = StyleSheet.create({
  page: {
    paddingHorizontal: 46,
    paddingTop: 46,
    paddingBottom: 60,
    fontFamily: "Helvetica",
    fontSize: 10.5,
    color: COLORS.ink,
    lineHeight: 1.42,
  },
  header: {
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
    paddingBottom: 14,
    marginBottom: 22,
  },
  title: {
    fontSize: 20,
    fontFamily: "Helvetica-Bold",
    color: COLORS.ink,
    letterSpacing: 0.2,
  },
  subtitle: {
    fontSize: 10,
    color: COLORS.muted,
    marginTop: 4,
  },
  sessionMeta: {
    fontSize: 9,
    color: COLORS.secondary,
    marginTop: 6,
  },

  circuit: {
    marginTop: 6,
    marginBottom: 22,
  },
  circuitHeader: {
    fontSize: 14,
    fontFamily: "Helvetica-Bold",
    color: COLORS.ink,
    marginBottom: 8,
  },
  metaRow: {
    flexDirection: "row",
    marginBottom: 14,
    borderRadius: 4,
    backgroundColor: COLORS.bgSubtle,
    padding: 10,
  },
  metaCell: {
    flex: 1,
    paddingHorizontal: 6,
  },
  metaLabel: {
    fontSize: 8.5,
    color: COLORS.muted,
    textTransform: "uppercase",
    letterSpacing: 0.6,
    marginBottom: 2,
  },
  metaValue: {
    fontSize: 10,
    fontFamily: "Helvetica-Bold",
    color: COLORS.ink,
  },
  metaValueMono: {
    fontSize: 9.5,
    fontFamily: "Courier",
    color: COLORS.ink,
  },

  agentBlock: {
    marginBottom: 16,
    borderRadius: 5,
    borderWidth: 1,
    borderColor: COLORS.border,
    padding: 12,
  },
  agentHeader: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 8,
    borderBottomWidth: 0.5,
    borderBottomColor: COLORS.border,
    paddingBottom: 6,
  },
  agentBadge: {
    fontSize: 8,
    fontFamily: "Helvetica-Bold",
    color: "#fff",
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 3,
    letterSpacing: 0.4,
  },
  agentTitle: {
    fontSize: 11,
    fontFamily: "Helvetica-Bold",
    color: COLORS.ink,
    marginLeft: 8,
  },
  agentSubtitle: {
    fontSize: 9,
    color: COLORS.muted,
    marginLeft: 6,
  },
  agentBody: {
    fontSize: 8.5,
    fontFamily: "Courier",
    color: COLORS.ink,
    lineHeight: 1.35,
  },

  separator: {
    borderTopWidth: 1,
    borderTopColor: COLORS.border,
    marginTop: 8,
    marginBottom: 18,
  },

  footer: {
    position: "absolute",
    bottom: 32,
    left: 46,
    right: 46,
    fontSize: 8,
    color: COLORS.muted,
    borderTopWidth: 0.5,
    borderTopColor: COLORS.border,
    paddingTop: 6,
    flexDirection: "row",
    justifyContent: "space-between",
  },

  emptyState: {
    textAlign: "center",
    color: COLORS.muted,
    fontSize: 11,
    marginTop: 40,
  },
});

const AGENT_COLORS = {
  1: COLORS.agent1,
  2: COLORS.agent2,
  3: COLORS.agent3,
} as const;

const AGENT_LABELS = {
  1: "Agent 1 — Oracle Extractor",
  2: "Agent 2 — Marked-State Identifier",
  3: "Agent 3 — Probability Distribution",
} as const;


// ─── Content parsers ────────────────────────────────────────────────────

interface AgentSlice {
  agentNum: 1 | 2 | 3;
  title: string;
  body: string;
}

const HEADER_RE = /^\s*\*\*\s*Agent\s+(\d+)\s+reasoning\s*\(([^)]+)\)\s*:\s*\*\*\s*\n+/i;
const END_MARKER_RE = /^[ \t]*===\s*END\s*===[ \t]*$/im;

/** Split the assistant content into agent sections. */
function parseAgentSlices(assistantContent: string): AgentSlice[] {
  if (!assistantContent) return [];
  const out: AgentSlice[] = [];
  for (const chunk of assistantContent.split(/\n+---\n+/)) {
    const m = chunk.match(HEADER_RE);
    if (!m) continue;
    const num = parseInt(m[1], 10);
    if (num !== 1 && num !== 2 && num !== 3) continue;
    let body = chunk.slice(m[0].length);
    // Strip anything after the model's END marker.
    const em = body.match(END_MARKER_RE);
    if (em && em.index !== undefined) body = body.slice(0, em.index);
    out.push({ agentNum: num, title: m[2].trim(), body: body.trim() });
  }
  return out;
}

/** Count qubits from OpenQASM source. Accepts both QASM 2.0 (`qreg q[N]`)
 *  and QASM 3.0 (`qubit[N] q`). Returns null if not detectable. */
function extractQubitCount(qasm: string): number | null {
  const q3 = qasm.match(/qubit\s*\[\s*(\d+)\s*\]/i);
  if (q3) return parseInt(q3[1], 10);
  const q2 = qasm.match(/qreg\s+\w+\s*\[\s*(\d+)\s*\]/i);
  if (q2) return parseInt(q2[1], 10);
  return null;
}

/** Extract the number of marked states from Agent 2's `=== Final Marked
 *  States ===` block. Returns null if no such block is present. */
function extractMarkedStateCount(slices: AgentSlice[]): {
  count: number | null;
  states: string[];
} {
  const a2 = slices.find((s) => s.agentNum === 2);
  if (!a2) return { count: null, states: [] };
  const m = a2.body.match(/===\s*Final Marked States\s*===\s*\n([\s\S]*?)(?=\n===|$)/i);
  if (!m) return { count: null, states: [] };
  const raw = m[1].trim();
  // States can be listed line-by-line, or JSON-like `['111', '000']`.
  const cleaned = raw.replace(/[\[\]'"]/g, "");
  const parts = cleaned
    .split(/[,\n]/)
    .map((s) => s.trim())
    .filter((s) => /^[01]+$/.test(s));
  return { count: parts.length || null, states: parts };
}


// ─── Report data model ─────────────────────────────────────────────────

export interface CircuitReport {
  filename: string;
  qasm: string;
  qubits: number | null;
  markedStateCount: number | null;
  markedStates: string[];
  agentSlices: AgentSlice[];
  complete: boolean;
}

/** Pair every user (QASM upload) message with the following assistant reply
 *  and derive the per-circuit report. Only circuits where at least one of
 *  the agent slices has streamed in are included; circuits where the
 *  assistant reply errored out with no content are also skipped. */
export function buildReportData(messages: Message[]): CircuitReport[] {
  const reports: CircuitReport[] = [];
  for (let i = 0; i < messages.length; i++) {
    const m = messages[i];
    if (m.role !== "user") continue;
    const nextAssistant = messages[i + 1];
    if (!nextAssistant || nextAssistant.role !== "assistant") continue;
    const slices = parseAgentSlices(nextAssistant.content);
    const { count, states } = extractMarkedStateCount(slices);
    reports.push({
      filename: m.filename ?? "circuit.qasm",
      qasm: m.content,
      qubits: extractQubitCount(m.content),
      markedStateCount: count,
      markedStates: states,
      agentSlices: slices,
      complete: !nextAssistant.streaming && !nextAssistant.error && slices.length === 3,
    });
  }
  return reports;
}


// ─── The PDF Document ──────────────────────────────────────────────────

function AgentBlock({ slice }: { slice: AgentSlice }) {
  const color = AGENT_COLORS[slice.agentNum];
  return (
    <View style={styles.agentBlock} wrap>
      <View style={styles.agentHeader}>
        <Text style={[styles.agentBadge, { backgroundColor: color }]}>
          AGENT {slice.agentNum}
        </Text>
        <Text style={styles.agentTitle}>{AGENT_LABELS[slice.agentNum]}</Text>
        {slice.title ? (
          <Text style={styles.agentSubtitle}>· {slice.title}</Text>
        ) : null}
      </View>
      <Text style={styles.agentBody}>{slice.body}</Text>
    </View>
  );
}

function CircuitBlock({ report, idx }: { report: CircuitReport; idx: number }) {
  return (
    <View style={styles.circuit} wrap>
      <Text style={styles.circuitHeader}>
        Circuit {idx + 1}: {report.filename}
      </Text>

      <View style={styles.metaRow}>
        <View style={styles.metaCell}>
          <Text style={styles.metaLabel}>Qubits</Text>
          <Text style={styles.metaValue}>
            {report.qubits ?? "—"}
          </Text>
        </View>
        <View style={styles.metaCell}>
          <Text style={styles.metaLabel}>Marked states</Text>
          <Text style={styles.metaValue}>
            {report.markedStateCount ?? "—"}
          </Text>
        </View>
        <View style={[styles.metaCell, { flex: 2 }]}>
          <Text style={styles.metaLabel}>Marked-state values</Text>
          <Text style={styles.metaValueMono}>
            {report.markedStates.length
              ? report.markedStates.join(", ")
              : "—"}
          </Text>
        </View>
      </View>

      {report.agentSlices.length === 0 ? (
        <Text style={{ color: COLORS.muted, fontSize: 10 }}>
          No agent output was captured for this circuit.
        </Text>
      ) : (
        report.agentSlices.map((slice, i) => (
          <AgentBlock key={i} slice={slice} />
        ))
      )}

      <View style={styles.separator} />
    </View>
  );
}

export function QuantumAgentsReport({
  reports,
  generatedAt,
}: {
  reports: CircuitReport[];
  generatedAt: Date;
}) {
  const dateStr = generatedAt.toLocaleString(undefined, {
    year: "numeric",
    month: "long",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
  const completeCount = reports.filter((r) => r.complete).length;
  return (
    <Document
      title="Quantum Agents Analysis Report"
      author="Quantum Agents"
      subject="Symbolic analysis of Grover QASM circuits"
    >
      <Page size="A4" style={styles.page}>
        <View style={styles.header} fixed>
          <Text style={styles.title}>Quantum Agents Analysis Report</Text>
          <Text style={styles.subtitle}>
            Symbolic analysis of Grover's-algorithm QASM circuits
          </Text>
          <Text style={styles.sessionMeta}>
            Generated {dateStr}   ·   {reports.length}{" "}
            {reports.length === 1 ? "circuit" : "circuits"}
            {completeCount < reports.length
              ? `   (${completeCount} with a complete three-agent analysis)`
              : ""}
          </Text>
        </View>

        {reports.length === 0 ? (
          <Text style={styles.emptyState}>
            No circuits were analysed in this session.
          </Text>
        ) : (
          reports.map((r, i) => <CircuitBlock key={i} report={r} idx={i} />)
        )}

        <View style={styles.footer} fixed>
          <Text>Quantum Agents — session report</Text>
          <Text
            render={({ pageNumber, totalPages }) =>
              `Page ${pageNumber} of ${totalPages}`
            }
          />
        </View>
      </Page>
    </Document>
  );
}
