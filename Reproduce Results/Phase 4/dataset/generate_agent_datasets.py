"""
Phase 4 — generate the three agent fine-tuning datasets from the data_MMS
QASM corpus.

Each circuit produces three training entries (one per agent), each in
LLaMA-Factory's alpaca format ({"instruction", "input", "output"}). The
inputs / outputs match exactly what the Phase 3 orchestrator sends to and
parses from each agent — so the resulting fine-tuned model can be dropped
straight into the same orchestrator with NO prompt scaffolding (no system
prompt, no few-shot examples) and produce identical chain behaviour.

Datasets produced (all n=2..7 — keeps training distribution narrow and
consistent across agents, matching Phase 1's FullCircuit range)
-----------------
  Grover_Agent1_2_7_MMS.json    input = full QASM
  Grover_Agent2_2_7_MMS.json    input = "=== Extracted Oracle Gate Definition ===\\n{gate block}"
  Grover_Agent3_2_7_MMS.json    input = "=== Final Marked States ===\\n{bitstrings}"

Each dataset's `instruction` field is empty (no system prompt — Phase 4's
design carries the task definition entirely in the training data and in
the input format, matching how the orchestrator calls the merged model
with `messages=[{"role": "user", "content": <input>}]` and nothing else).

Why three separate files
------------------------
LLaMA-Factory's `dataset:` field accepts a comma-separated list and
concatenates them, mirroring Phase 1's joint training of
Grover_FullCircuit + Grover_Oracle. Keeping each agent dataset as its
own file makes per-agent counts and overrides obvious in the registration
manifest.

Output formats
--------------
The output strings are the EXACT format Phase 3's parsers expect, with
two intentional deviations from the Phase 3 prompt scaffolding:
  - no `=== END ===` terminator (Phase 4 design §6.2: stop is taught
    purely by EOS token at end-of-sequence)
  - no echoed examples / instruction prose (the model learns the format
    from the training distribution alone)

Run:
    python "Multi Agent System Phase 4/dataset/generate_agent_datasets.py" \\
        --data_dir GroverGPT-plus/data_MMS \\
        --out_dir  "Multi Agent System Phase 4/dataset"
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Phase 4 trains all three agents on the same n-range so the joint
# distribution is uniform across agents and contains no agent-specific
# n-coverage gaps. n=2..7 matches Phase 1's full-circuit range.
N_MIN = 2
N_MAX = 7


# ─── Oracle parsing (verbatim port of GroverGPT+/dataset_generate_MMS.py) ───
# Kept inline (not imported) so this script is self-contained — the
# GroverGPT-plus repo is a third-party reference, not a project module.

def _extract_oracle_structure(qasm_content: str) -> Optional[Tuple[List[str], List[str], str, str]]:
    """Return (params, ops, oracle_head, oracle_body) for the `gate Oracle` block."""
    m = re.search(
        r"gate Oracle\s*((?:[\w]+,?\s*)+)\s*\{([^}]*)\}",
        qasm_content,
        re.DOTALL,
    )
    if not m:
        return None
    oracle_head = "gate Oracle " + m.group(1)
    params = [p.strip() for p in m.group(1).split(",")]
    ops = [op.strip() for op in m.group(2).split(";") if op.strip()]
    return params, ops, oracle_head, m.group(2)


def _find_block_start(ops: List[str], mcmt_idx: int, prev_end: int) -> int:
    start = mcmt_idx
    while start > prev_end and ops[start - 1].startswith("x "):
        start -= 1
    return start


def _find_block_end(ops: List[str], mcmt_idx: int, start: int) -> int:
    return mcmt_idx + (mcmt_idx - start)


def _analyze_blocks(params: List[str], operations: List[str]) -> Tuple[
    List[str],                         # marked_states
    List[Tuple[int, int, int]],        # block ranges (start, mcmt_idx, end)
    List[List[str]],                   # per-block "state construction" trace lines
]:
    """Segment the Oracle body into MCMT-centred blocks and derive each marked state.

    Returns ALSO the per-qubit state-construction trace for each block in
    the same prose format Phase 3's Agent 2 prompt teaches the model to
    emit (`x _gate_q_K: Present → 0, then → 0...`).
    """
    mcmt_indices = [i for i, op in enumerate(operations) if op.startswith("mcmt")]
    marked_states: List[str] = []
    block_ranges: List[Tuple[int, int, int]] = []
    trace_lines: List[List[str]] = []
    prev_end = 0

    for mcmt_idx in mcmt_indices:
        start = _find_block_start(operations, mcmt_idx, prev_end)
        end = _find_block_end(operations, mcmt_idx, start)
        block_ranges.append((start, mcmt_idx, end))
        prev_end = end + 1

        block_ops = operations[start : end + 1]
        # Locate the MCMT inside the slice (canonical form: "mcmt p1, p2, ...").
        mcmt_str = f"mcmt {', '.join(params)}"
        mcmt_local_idx = block_ops.index(mcmt_str)

        current_state = ""
        block_trace: List[str] = []
        for q in params:
            pre_x = any(op == f"x {q}" for op in block_ops[:mcmt_local_idx])
            post_x = any(op == f"x {q}" for op in block_ops[mcmt_local_idx + 1 :])
            bit = "0" if (pre_x and post_x) else "1"
            current_state = bit + current_state
            presence = "Present" if (pre_x and post_x) else "Absent"
            block_trace.append(f"x {q}: {presence} → {bit}, then → {current_state}")
        marked_states.append(current_state)
        trace_lines.append(block_trace)
    return marked_states, block_ranges, trace_lines


# ─── Agent 1 output (Oracle Extraction Reasoning + extracted gate def) ───

def build_agent1_output(oracle_head: str, oracle_body: str,
                        block_ranges: List[Tuple[int, int, int]],
                        operations: List[str]) -> str:
    """Produce the Phase 3 Agent 1 output format.

    Step 3 of the reasoning describes the X-gate symmetry of each block.
    The text is generated programmatically per-block so it matches the
    actual circuit (not just a stock sentence).
    """
    block_sentences: List[str] = []
    for i, (start, mcmt_idx, end) in enumerate(block_ranges):
        # Collect qubit indices (e.g. '_gate_q_3' → 3) of X-ops before / after the MCMT.
        before_ops = operations[start:mcmt_idx]
        after_ops = operations[mcmt_idx + 1 : end + 1]
        before_qs = [int(op.split("_")[-1]) for op in before_ops if op.startswith("x ")]
        after_qs = [int(op.split("_")[-1]) for op in after_ops if op.startswith("x ")]
        matched = "matched" if sorted(before_qs) == sorted(after_qs) else "MISMATCH"
        if before_qs:
            qs_str = ",".join(f"q{q}" for q in before_qs)
            block_sentences.append(
                f"Block {i + 1}: X gates on {qs_str} before MCMT and same X gates after — {matched}."
            )
        else:
            block_sentences.append(
                f"Block {i + 1}: no X gates around MCMT (marks the all-ones state) — {matched}."
            )

    step3 = "X-gate blocks must be consistent. " + " ".join(block_sentences)

    return (
        "=== Oracle Extraction Reasoning ===\n"
        "\n"
        "1. Locating the 'gate Oracle' definition in the input.\n"
        "2. Each MCMT gate in the Oracle body, together with its surrounding X gates, encodes one marked state.\n"
        f"3. {step3}\n"
        "4. Extracted Oracle gate definition must be copied verbatim from input, with no modifications, no truncation, no extra operations.\n"
        "\n"
        "=== Extracted Oracle Gate Definition ===\n"
        f"{oracle_head.rstrip()} {{{oracle_body.rstrip()}\n}}"
    )


# ─── Agent 2 output (oracle echo + per-block state derivation + marked states) ───

def build_agent2_output(oracle_head: str, oracle_body: str,
                        operations: List[str],
                        block_ranges: List[Tuple[int, int, int]],
                        trace_lines: List[List[str]],
                        marked_states: List[str]) -> str:
    """Produce the Phase 3 Agent 2 output format."""
    lines: List[str] = []
    lines.append("The Oracle entity is extracted below:")
    lines.append("")
    lines.append(f"{oracle_head.rstrip()} {{{oracle_body.rstrip()}\n}}")
    lines.append("")
    n_blocks = len(block_ranges)
    lines.append(f"There are {n_blocks} MCMT gates, so there are {n_blocks} Blocks.")

    for i, ((start, mcmt_idx, end), trace, marked) in enumerate(
        zip(block_ranges, trace_lines, marked_states)
    ):
        block_ops = operations[start : end + 1]
        lines.append("")
        lines.append(f"=== Block {i + 1} ===")
        lines.append("Operation sequence:")
        for op in block_ops:
            lines.append(f"{op};")
        lines.append("")
        lines.append("State construction:")
        lines.extend(trace)
        lines.append(f"Final state: {marked}")

    lines.append("")
    lines.append("=== Final Marked States ===")
    lines.extend(marked_states)
    return "\n".join(lines)


# ─── Agent 3 output (probability reasoning + simulation dict) ───

def _compute_theta(t: int, n: int) -> float:
    N = 2 ** n
    return math.asin(math.sqrt(t / N))


def _compute_p_marked(t: int, n: int) -> float:
    """Optimal-k Grover probability per marked state (k_opt = floor(pi/4 * sqrt(N/t)))."""
    N = 2 ** n
    theta = _compute_theta(t, n)
    k_opt = math.floor(math.pi / 4 * math.sqrt(N / t))
    angle = (2 * k_opt + 1) * theta
    p_marked_total = math.sin(angle) ** 2
    return p_marked_total / t


def _format_prob_value(p: float) -> Tuple[str, bool]:
    """Return (string, is_scientific). The bool tells the dict formatter
    whether the value should be emitted as scientific notation."""
    rounded = round(p, 4)
    if rounded == 0.0 and p != 0.0:
        return ("{:.4e}".format(p), True)
    return (f"{rounded:.4f}", False)


def _format_prob(p: float) -> str:
    s, _ = _format_prob_value(p)
    return s


def build_agent3_output(marked_states: List[str], n: int) -> str:
    """Produce the Phase 3 Agent 3 output format."""
    t = len(marked_states)
    N = 2 ** n
    p_marked = _compute_p_marked(t, n)
    p_marked_str = _format_prob(p_marked)

    if N - t > 0:
        p_unmarked = (1.0 - t * p_marked) / (N - t)
    else:
        p_unmarked = 0.0
    p_unmarked_str = _format_prob(p_unmarked)

    total = t * p_marked + (N - t) * p_unmarked
    total_str = _format_prob(total)

    emit = min(N, 30)
    n_unmarked_emit = emit - t

    lines = ["=== Probability Reasoning ===", ""]
    lines.append(f"  1. Marked-state count: t = {t}")
    lines.append(f"  2. Qubit count: n = {n}, so state-space size N = 2^n = {N}")
    lines.append(
        f"  3. Each marked state receives equal amplitude under Grover's algorithm. "
        f"So, per marked state probability: p_marked = {p_marked_str}"
    )
    lines.append(
        f"  4. Remaining (N − t) = {N}−{t} = {N - t} non-marked states share the residual probability. "
        f"Per non-marked state probability: p_unmarked = {p_unmarked_str} (very small positive number)"
    )
    lines.append(
        f"  5. Total probability is approximately 1: "
        f"t · p_marked + (N − t) · p_unmarked = {t} · {p_marked_str} + ({N - t}) · {p_unmarked_str} = {total_str}"
    )
    lines.append(f"  6. Verbatim marked-state strings (each length n = {n}):")
    for m in marked_states:
        lines.append(f"     - {m}")
    lines.append(
        f"  7. Output schema: {{ '<{n}-bit state string>': <4-decimal prob>, ... }} "
        f"sorted in descending order by probability. "
        f"I will emit min(N, 30) = min({N},30) = {emit} entries total where:"
    )
    lines.append(f" - first entries are {t} marked-state strings at {p_marked_str} probability")
    lines.append(
        f" - remaining entries are (min(N,30) - t) = (min({N},30) - {t}) = {n_unmarked_emit} "
        f"un-marked-state strings at {p_unmarked_str} probability in ascending bitstring order"
    )

    # Simulation dict — one entry per line, marked states first then ascending unmarked.
    lines.append("")
    lines.append("=== Simulation Results of the Grover's Algorithm ===")
    lines.append("{")

    unmarked: List[str] = []
    marked_set = set(marked_states)
    if n_unmarked_emit > 0:
        for i in range(N):
            bits = format(i, f"0{n}b")
            if bits in marked_set:
                continue
            unmarked.append(bits)
            if len(unmarked) >= n_unmarked_emit:
                break

    entries: List[Tuple[str, str]] = []
    for m in marked_states:
        entries.append((m, p_marked_str))
    for u in unmarked:
        entries.append((u, p_unmarked_str))

    for i, (key, val_str) in enumerate(entries):
        suffix = "," if i < len(entries) - 1 else ""
        lines.append(f"  '{key}': {val_str}{suffix}")
    lines.append("}")
    return "\n".join(lines)


# ─── Driver ────────────────────────────────────────────────────────────────

def _iter_qasm_in_range(data_dir: Path, n_min: int, n_max: int):
    """Yield (n, qasm_path) for grover_n{n}/*.qasm files with n in [n_min, n_max]."""
    subdirs = []
    for entry in sorted(data_dir.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("grover_n"):
            continue
        suffix = entry.name[len("grover_n"):]
        if not suffix.isdigit():
            continue
        n = int(suffix)
        if n_min <= n <= n_max:
            subdirs.append((n, entry))
    subdirs.sort()
    for n, sub in subdirs:
        for qasm in sorted(sub.glob("*.qasm")):
            yield n, qasm


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="GroverGPT-plus/data_MMS",
                    help="Root QASM corpus (contains grover_n2/, grover_n3/, ...)")
    ap.add_argument("--out_dir", default="Multi Agent System Phase 4/dataset",
                    help="Where to write Grover_AgentN_*.json")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    agent1: List[Dict] = []
    agent2: List[Dict] = []
    agent3: List[Dict] = []
    skipped = 0

    for n, qasm_path in _iter_qasm_in_range(data_dir, N_MIN, N_MAX):
        content = qasm_path.read_text()
        parsed = _extract_oracle_structure(content)
        if parsed is None:
            skipped += 1
            continue
        params, ops, oracle_head, oracle_body = parsed
        marked_states, block_ranges, trace_lines = _analyze_blocks(params, ops)
        if not marked_states:
            skipped += 1
            continue

        a1_input = content.rstrip()
        a1_output = build_agent1_output(oracle_head, oracle_body, block_ranges, ops)

        # A2's input mirrors what Phase 3's A1Result.text passes downstream:
        # "=== Extracted Oracle Gate Definition ===\n<gate Oracle ...>{...}".
        a1_gate_text = f"{oracle_head.rstrip()} {{{oracle_body.rstrip()}\n}}"
        a2_input = f"=== Extracted Oracle Gate Definition ===\n{a1_gate_text}"
        a2_output = build_agent2_output(
            oracle_head, oracle_body, ops, block_ranges, trace_lines, marked_states,
        )

        # A3's input mirrors A2Result.text: "=== Final Marked States ===\n<bitstrings>".
        a3_input = "=== Final Marked States ===\n" + "\n".join(marked_states)
        a3_output = build_agent3_output(marked_states, n)

        agent1.append({"instruction": "", "input": a1_input, "output": a1_output})
        agent2.append({"instruction": "", "input": a2_input, "output": a2_output})
        agent3.append({"instruction": "", "input": a3_input, "output": a3_output})

    def _dump(records: List[Dict], path: Path, label: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"  {label}:  {len(records):>5}  →  {path.name}")

    print(f"Skipped (no oracle / no marked states): {skipped}")
    print("\nWrote:")
    _dump(agent1, out_dir / f"Grover_Agent1_{N_MIN}_{N_MAX}_MMS.json", "Agent 1")
    _dump(agent2, out_dir / f"Grover_Agent2_{N_MIN}_{N_MAX}_MMS.json", "Agent 2")
    _dump(agent3, out_dir / f"Grover_Agent3_{N_MIN}_{N_MAX}_MMS.json", "Agent 3")
    print(f"\nTotal training entries (joint): "
          f"{len(agent1) + len(agent2) + len(agent3)}")


if __name__ == "__main__":
    main()
