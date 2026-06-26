import { FileUp, Loader2, Square } from "lucide-react";
import { useCallback, useRef, useState } from "react";

interface DropzoneProps {
  onFile: (file: File) => void;
  disabled?: boolean;
  onStop?: () => void;
}

export function Dropzone({ onFile, disabled, onStop }: DropzoneProps) {
  const [over, setOver] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const handle = useCallback(
    (file: File | undefined) => {
      if (!file) return;
      const name = file.name.toLowerCase();
      if (!name.endsWith(".qasm") && !name.endsWith(".txt")) {
        alert("Please drop a .qasm (or .txt) file. The proxy validates the contents server-side.");
        return;
      }
      onFile(file);
    },
    [onFile],
  );

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); if (!disabled) setOver(true); }}
      onDragLeave={(e) => { e.preventDefault(); setOver(false); }}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        if (disabled) return;
        handle(e.dataTransfer.files?.[0]);
      }}
      onClick={() => !disabled && inputRef.current?.click()}
      className={[
        "group relative cursor-pointer select-none",
        "card-glass border-2 border-dashed transition-all duration-200",
        "px-6 py-7 sm:py-9 text-center",
        disabled
          ? "opacity-60 cursor-not-allowed"
          : over
          ? "border-accent-cyan scale-[1.01] shadow-[0_0_32px_-6px_rgb(var(--accent-cyan)/0.5)]"
          : "border-border-strong hover:border-accent-violet/50 hover:shadow-card-strong",
      ].join(" ")}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".qasm,.txt,text/plain"
        className="hidden"
        onChange={(e) => handle(e.target.files?.[0] ?? undefined)}
      />

      {disabled ? (
        <div className="flex flex-col items-center gap-3 text-text-secondary">
          <Loader2 className="w-7 h-7 animate-spin text-accent-violet" />
          <p className="text-sm font-medium">Agents are analysing your circuit…</p>
          {onStop && (
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); onStop(); }}
              className={[
                "mt-1 inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full",
                "text-xs font-medium font-display tracking-tight",
                "bg-bg-elev border border-border-strong text-text-primary",
                "hover:border-accent-magenta hover:text-accent-magenta",
                "transition cursor-pointer",
              ].join(" ")}
            >
              <Square className="w-3 h-3 fill-current" strokeWidth={0} />
              Stop response
            </button>
          )}
        </div>
      ) : (
        <div className="flex flex-col items-center gap-3">
          <div className="relative">
            <FileUp
              className={[
                "w-9 h-9 transition-colors",
                over ? "text-accent-cyan" : "text-accent-violet",
              ].join(" ")}
              strokeWidth={1.6}
            />
            {over && (
              <div className="absolute inset-0 blur-2xl bg-accent-cyan/40 rounded-full" />
            )}
          </div>
          <div>
            <p className="text-text-primary font-medium font-display tracking-tight">
              {over ? "Release to analyse" : "Drop a .qasm file here"}
            </p>
            <p className="text-xs text-text-muted mt-1">
              or click to browse · OpenQASM 3.0 Grover circuits
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
