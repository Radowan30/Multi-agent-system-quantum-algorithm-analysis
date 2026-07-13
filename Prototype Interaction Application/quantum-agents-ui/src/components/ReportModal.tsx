/**
 * Modal that previews the PDF report and offers a Download button.
 *
 * The PDF is generated once (via @react-pdf/renderer's `pdf().toBlob()`) and
 * the resulting Blob URL is used both for the in-modal <iframe> preview and
 * for the Download link, so what the user sees is exactly what they download.
 */
import { pdf } from "@react-pdf/renderer";
import { Download, Loader2, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { buildReportData, QuantumAgentsReport } from "../lib/pdfReport";
import type { Message } from "../lib/types";

interface ReportModalProps {
  open: boolean;
  onClose: () => void;
  messages: Message[];
}

function formatFilenameForDownload(): string {
  // Uses browser locale so filenames are consistent with what the user saw
  // in the "Generated" header line.
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, "0");
  return `quantum-agents-report_${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}.pdf`;
}

export function ReportModal({ open, onClose, messages }: ReportModalProps) {
  const [blobUrl, setBlobUrl] = useState<string | null>(null);
  const [building, setBuilding] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Freeze the report data at the moment the modal opens — that way, if a
  // late stream token arrives while the preview is open the preview doesn't
  // silently mutate (which would break the guarantee that download == preview).
  const reports = useMemo(
    () => (open ? buildReportData(messages) : []),
    [open, messages],
  );
  const generatedAt = useMemo(() => (open ? new Date() : new Date(0)), [open]);
  // Track the exact filename we'll use for the download link so it also
  // matches the timestamp shown in the header line.
  const downloadName = useMemo(
    () => (open ? formatFilenameForDownload() : ""),
    [open],
  );

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setBuilding(true);
    setError(null);
    setBlobUrl(null);
    const doc = <QuantumAgentsReport reports={reports} generatedAt={generatedAt} />;
    pdf(doc)
      .toBlob()
      .then((blob) => {
        if (cancelled) return;
        const url = URL.createObjectURL(blob);
        setBlobUrl(url);
        setBuilding(false);
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : "Failed to build report");
        setBuilding(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, reports, generatedAt]);

  // Revoke the Blob URL when the modal closes, so we don't leak memory across
  // repeated open/close cycles.
  useEffect(() => {
    return () => {
      if (blobUrl) URL.revokeObjectURL(blobUrl);
    };
  }, [blobUrl]);

  // Escape-to-close
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  const nothingToReport = reports.length === 0;

  return (
    <div
      className="fixed inset-0 z-40 flex items-center justify-center px-4 sm:px-8 py-4 sm:py-8 bg-bg-base/70 backdrop-blur-sm animate-fade-in"
      role="dialog"
      aria-modal="true"
      aria-label="Analysis report preview"
      onClick={onClose}
    >
      <div
        className="card-glass max-w-4xl w-full h-full flex flex-col overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="px-6 py-4 border-b border-border-subtle flex items-center gap-4">
          <div className="flex-1 min-w-0">
            <h2 className="text-base font-display font-semibold tracking-tight text-text-primary">
              Analysis report
            </h2>
            <p className="text-xs text-text-muted mt-0.5">
              {reports.length === 0
                ? "No analysed circuits available yet."
                : `${reports.length} circuit${reports.length === 1 ? "" : "s"} in this session`}
            </p>
          </div>

          {blobUrl && !nothingToReport && (
            <a
              href={blobUrl}
              download={downloadName}
              className="btn-primary text-sm inline-flex items-center gap-2"
            >
              <Download className="w-4 h-4" strokeWidth={2} />
              Download PDF
            </a>
          )}

          <button
            type="button"
            onClick={onClose}
            className="btn-ghost"
            aria-label="Close"
          >
            <X className="w-4 h-4" strokeWidth={2} />
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 min-h-0 bg-bg-subtle relative">
          {nothingToReport ? (
            <div className="absolute inset-0 flex items-center justify-center text-sm text-text-muted px-6 text-center">
              Upload a Grover QASM circuit and let the agents complete their
              analysis to enable the report.
            </div>
          ) : building ? (
            <div className="absolute inset-0 flex items-center justify-center gap-2 text-text-secondary">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span className="text-sm">Building PDF…</span>
            </div>
          ) : error ? (
            <div className="absolute inset-0 flex items-center justify-center text-sm text-rose-400 px-6 text-center">
              Failed to build the PDF: {error}
            </div>
          ) : (
            <iframe
              title="Analysis report preview"
              src={blobUrl ?? undefined}
              className="w-full h-full border-0 bg-white"
            />
          )}
        </div>
      </div>
    </div>
  );
}
