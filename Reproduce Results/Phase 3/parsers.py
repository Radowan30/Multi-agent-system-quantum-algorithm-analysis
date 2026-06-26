"""
Output parsers for the three agents in the Phase 3 pipeline.

Each parser is permissive: it locates the relevant `=== ... ===` marker
inside the model's full response and extracts the content that follows. If
the marker is missing or the content after it doesn't have the expected
shape, the parser returns None — the chain treats this as a parse failure
at that agent.

Each parser returns a small dataclass with TWO fields:
  - `.text`  — the marker-prefixed text fed verbatim to the next agent
               (matches the format the next agent's few-shot examples use)
  - `.data`  — the structured payload (str / list / dict) used for metrics

Public API:
    parse_agent1_output(text: str) -> Optional[A1Result]
    parse_agent2_output(text: str) -> Optional[A2Result]
    parse_agent3_output(text: str) -> Optional[A3Result]
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

# Markers — kept here as constants. Parsers locate them; build functions
# in prompts.py reference them only indirectly (parser output already has
# them prefixed).
AGENT1_MARKER = "=== Extracted Oracle Gate Definition ==="
AGENT2_MARKER = "=== Final Marked States ==="
AGENT3_MARKER = "=== Simulation Results of the Grover's Algorithm ==="


@dataclass
class A1Result:
    text: str          # `=== Extracted Oracle Gate Definition ===\ngate Oracle ... { ... }`
    oracle_block: str  # just `gate Oracle ... { ... }` (for oracle-accuracy comparison)


@dataclass
class A2Result:
    text: str                 # `=== Final Marked States ===\n<bitstrings>`
    marked_states: List[str]  # ['01', '10', ...]


@dataclass
class A3Result:
    text: str                          # `=== Simulation Results ... ===\n{...}`
    probability_dict: Dict[str, float] # {'01': 1.0, '00': 0.0, ...}


def parse_agent1_output(text: str) -> Optional[A1Result]:
    """Locate the Oracle gate block emitted after Agent 1's marker.

    STRICT by design: requires a properly-closed `gate Oracle ... { ... }`
    block with a balanced `}`. If the closing brace is missing — e.g.
    because a model trained on the full monolithic CoT continued generating
    past the oracle into what it thinks is the next phase — this returns
    None, the chain records a parse failure at A1, and the circuit is
    treated as a failure for downstream metrics.

    This strict behavior is intentional for the Phase 3 framing: the
    fine-tuned-for-monolithic models are *expected* to fight the agent
    decomposition, and that failure signal is itself meaningful data
    (Research Plan §5.7 — untrained-base comparison is the primary Phase 3
    measurement; fine-tuned-model agent performance is reported as a
    secondary comparison).

    Returns A1Result with the marker-prefixed text and the gate block alone,
    or None if the marker is missing / no balanced gate block is found.
    """
    idx = text.rfind(AGENT1_MARKER)
    if idx < 0:
        return None
    after = text[idx + len(AGENT1_MARKER):]
    # `gate Oracle ... { ... }` — no nested braces in our domain.
    match = re.search(
        r"(gate\s+Oracle\b[^{]*\{[^{}]*\})",
        after,
        re.DOTALL,
    )
    if not match:
        return None
    oracle = match.group(1).strip()
    return A1Result(
        text=f"{AGENT1_MARKER}\n{oracle}",
        oracle_block=oracle,
    )


def parse_agent2_output(text: str) -> Optional[A2Result]:
    """Locate the marked-state block emitted after Agent 2's final marker.

    We use the LAST occurrence of the marker because Agent 2's own few-shot
    examples contain the marker and the model may echo it; the actual final
    answer is always the last one.

    Bitstrings are validated to contain only '0' and '1' with consistent
    length; the first non-bitstring line stops the scan.
    """
    idx = text.rfind(AGENT2_MARKER)
    if idx < 0:
        return None
    after = text[idx + len(AGENT2_MARKER):].strip()
    states: List[str] = []
    for line in after.split("\n"):
        line = line.strip()
        if not line:
            if states:
                break
            continue
        if line.startswith("==="):
            break
        if not all(c in "01" for c in line):
            break
        if states and len(line) != len(states[0]):
            break
        states.append(line)
    if not states:
        return None
    return A2Result(
        text=f"{AGENT2_MARKER}\n" + "\n".join(states),
        marked_states=states,
    )


def parse_agent3_output(text: str) -> Optional[A3Result]:
    """Locate the probability dict emitted after Agent 3's marker.

    Accepts single or double quotes around the bitstring key, and
    probabilities in decimal or scientific notation. First '{' after the
    marker opens the block; first '}' after that closes it.
    """
    idx = text.rfind(AGENT3_MARKER)
    if idx < 0:
        return None
    after = text[idx + len(AGENT3_MARKER):]
    brace_start = after.find("{")
    if brace_start < 0:
        return None
    brace_end = after.find("}", brace_start)
    if brace_end < 0:
        return None
    body = after[brace_start + 1 : brace_end]

    prob_dict: Dict[str, float] = {}
    for entry in body.split(","):
        entry = entry.strip()
        if not entry:
            continue
        m = re.match(
            r"['\"]([01]+)['\"]\s*:\s*([\d.eE+\-]+)",
            entry,
        )
        if not m:
            continue
        bits = m.group(1)
        try:
            val = float(m.group(2))
        except ValueError:
            continue
        prob_dict[bits] = val  # last value wins on collision (rare)
    if not prob_dict:
        return None
    # Preserve the original dict text the model emitted, including the braces
    # and trailing whitespace — useful for the cot_agents.md log.
    return A3Result(
        text=f"{AGENT3_MARKER}\n{after[brace_start : brace_end + 1].strip()}",
        probability_dict=prob_dict,
    )
