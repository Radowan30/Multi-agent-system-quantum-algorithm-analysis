"""
Logger that writes per-circuit agent chain traces to `cot_agents.md`.

One block per circuit: circuit ID, the input QASM, then for each of the three
agents the input it received and the full text response it produced. Used for
qualitative analysis and for building the failure-pattern catalogue.

Public API:
    class CotLogger(output_path: str)
        .log_chain(circuit_id, qasm, ChainResult) -> None
        .close() -> None
"""

from pathlib import Path
from typing import Optional


class CotLogger:
    """Append-only logger. One file per evaluation run."""

    def __init__(self, output_path: str):
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate at construction so a fresh run starts a fresh file.
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("# Phase 3 Multi-Agent CoT Traces\n\n")

    def log_chain(self, circuit_id: Optional[str], qasm: str, result) -> None:
        """Append one circuit's full chain trace.

        `result` is a chain.ChainResult; we read its raw outputs directly so
        the log captures everything the agents emitted (including failed
        traces — useful for diagnosing parse failures).
        """
        ckt = circuit_id or "(unknown)"
        lines = [f"## Circuit: {ckt}\n"]
        lines.append(f"**success:** {result.success}    "
                     f"**failure_stage:** {result.failure_stage}    "
                     f"**total_time_s:** {result.total_time_s:.3f}\n")
        lines.append(f"### Original input QASM\n```\n{qasm.rstrip()}\n```\n")

        # Agent 1
        lines.append("### Agent 1 (Oracle Extractor)\n")
        lines.append(f"_time: {result.a1_time_s:.3f}s_\n")
        lines.append("#### Output\n```\n")
        lines.append((result.a1_output or "(no output)").rstrip())
        lines.append("\n```\n")
        if result.extracted_oracle:
            lines.append("#### Parsed oracle\n```\n"
                         f"{result.extracted_oracle.rstrip()}\n```\n")

        # Agent 2 (only if A1 succeeded)
        if result.a2_output is not None:
            lines.append("### Agent 2 (Marked-State Identifier)\n")
            lines.append(f"_time: {result.a2_time_s:.3f}s_\n")
            lines.append("#### Output\n```\n")
            lines.append(result.a2_output.rstrip())
            lines.append("\n```\n")
            if result.marked_states:
                lines.append("#### Parsed marked states\n```\n"
                             f"{chr(10).join(result.marked_states)}\n```\n")

        # Agent 3 (only if A2 succeeded)
        if result.a3_output is not None:
            lines.append("### Agent 3 (Probability Distribution)\n")
            lines.append(f"_time: {result.a3_time_s:.3f}s_\n")
            lines.append("#### Output\n```\n")
            lines.append(result.a3_output.rstrip())
            lines.append("\n```\n")
            if result.probability_dict:
                lines.append("#### Parsed probability dict\n```\n")
                for state, prob in result.probability_dict.items():
                    lines.append(f"  '{state}': {prob}\n")
                lines.append("```\n")

        lines.append("\n---\n\n")

        with open(self.path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))
