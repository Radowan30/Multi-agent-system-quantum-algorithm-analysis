"""
Parser for GroverGPT+ model output — shared across all three evaluation phases.

Model output format (from dataset_generate_MMS.py):
  === Analysis ===
  ...
  === Block N ===
  ...
  === Final Marked States ===
  <state_1>
  <state_2>
  ...
  === Simulation Results of the Grover's Algorithm ===
  {'<state>': <prob>, ...}
"""

import ast
import re
from typing import Dict, List, Optional, Tuple


_MARKED_HEADER   = "=== Final Marked States ==="
_RESULTS_HEADER  = "=== Simulation Results of the Grover's Algorithm ==="


def parse_model_output(
    text: str,
) -> Tuple[Optional[Dict[str, float]], Optional[List[str]]]:
    """
    Parse a single model output string.

    Returns
    -------
    (predicted_probs, predicted_marked)
      predicted_probs   : dict mapping bit-string → float, or None if parsing fails
      predicted_marked  : list of bit-strings from the Final Marked States section,
                          or None if that section is missing

    The predicted_probs dict is the primary input to SA and CF metrics.
    The predicted_marked list is logged for diagnostics but is not used for
    metric computation (metrics use the actual marked states from the circuit).
    """
    predicted_probs  = _parse_probs(text)
    predicted_marked = _parse_marked_states(text)
    return predicted_probs, predicted_marked


def _frac_to_decimal(match: "re.Match") -> str:
    """Rewrite an integer fraction 'a/b' as its decimal value.

    The model sometimes emits probabilities as fractions (e.g. 1/8) instead of
    floats; ast.literal_eval cannot evaluate a division expression, so fractions
    are converted to decimals before parsing. A zero denominator is left as-is
    so the surrounding ast.literal_eval still fails it as unparsable.
    """
    num, den = int(match.group(1)), int(match.group(2))
    return repr(num / den) if den != 0 else match.group(0)


def _parse_probs(text: str) -> Optional[Dict[str, float]]:
    """Extract the probability dict from the Simulation Results section."""
    idx = text.find(_RESULTS_HEADER)
    if idx == -1:
        return None

    after = text[idx + len(_RESULTS_HEADER):].strip()

    # The dict may span multiple lines; find the matching closing brace.
    brace_start = after.find("{")
    if brace_start == -1:
        return None

    depth = 0
    brace_end = -1
    for i, ch in enumerate(after[brace_start:], start=brace_start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                brace_end = i + 1
                break

    if brace_end == -1:
        return None

    raw = after[brace_start:brace_end]
    # Convert any integer fractions (e.g. 1/8) to decimals first — the model
    # emits fractions for some circuits and ast.literal_eval cannot evaluate
    # them. Genuinely unparsable values still fall through to None below.
    raw = re.sub(r"(\d+)\s*/\s*(\d+)", _frac_to_decimal, raw)
    try:
        result = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None

    if not isinstance(result, dict):
        return None

    # Normalise: keys to str, values to float.
    # Skip entries whose key is not a valid binary string — the model sometimes
    # outputs non-binary state labels (e.g. '6543210'), which would crash the
    # int(s, 2) sort in metrics.py and are meaningless as quantum states.
    probs: Dict[str, float] = {}
    for k, v in result.items():
        key = str(k)
        if not re.fullmatch(r"[01]+", key):
            continue
        try:
            probs[key] = float(v)
        except (TypeError, ValueError):
            continue

    return probs if probs else None


def _parse_marked_states(text: str) -> Optional[List[str]]:
    """Extract the list of marked states from the Final Marked States section."""
    idx = text.find(_MARKED_HEADER)
    if idx == -1:
        return None

    after = text[idx + len(_MARKED_HEADER):]

    # Everything up to the next === header is the marked states block.
    next_header = after.find("===")
    block = after[:next_header] if next_header != -1 else after

    states = [line.strip() for line in block.splitlines() if line.strip()]
    # Keep only bit-strings (all chars are '0' or '1').
    states = [s for s in states if re.fullmatch(r"[01]+", s)]
    return states if states else None


def parse_batch(
    texts: List[str],
) -> List[Tuple[Optional[Dict[str, float]], Optional[List[str]]]]:
    """Parse a list of model output strings. Returns one result tuple per input."""
    return [parse_model_output(t) for t in texts]
