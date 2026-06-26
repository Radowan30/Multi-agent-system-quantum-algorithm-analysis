"""
Evaluation metrics for GroverGPT+ — shared across all three evaluation phases.

Metrics (Research_Plan.md Section 3.3):
  SA  — Search Accuracy          (equations 1-3)
  CF  — Classical Fidelity       (Bhattacharyya coefficient)
  CR  — Compression Ratio        (equation 8: mean of per-circuit L_base/L_quantum)
  SRR — Sequence Reduction Ratio (equation 9: mean of per-circuit (L_base-L_quantum)/L_base)
  RET — Relative Execution Time  (S(n) = T(n)/T(2), full-circuit inputs)
          Phase 1: n=2-9   Phase 2: n=2-19

Ground truth probabilities are computed via the analytically exact Grover formula,
which is equivalent to statevector simulation to <1e-10 and is stated explicitly
in the research paper.
"""

import math
import statistics
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Notation: marked-state count (the paper uses TWO symbols for it)
# ---------------------------------------------------------------------------
# The number of marked states (`len(marked_states)`) is central to every metric
# here. The paper uses two different symbols for this single quantity:
#   * `t` — npj Supplementary §8 (Grover formula); GroverGPT-2 preprint App D.
#   * `k` — npj main text Eq. 1–2 (SA definition); also the `data_MMS` filename
#           convention (e.g. `grover_n5_k2_m00000_00100` means t=2).
# We follow each function's matching paper context:
#   * grover_ground_truth uses `t` (implements the Grover formula).
#   * search_accuracy     uses `k` (implements the SA definition).
# Both denote the same quantity (`len(marked_states)`). Do NOT confuse either
# with `k_opt` — the optimal Grover ITERATION count, k_opt = ⌊(π/4)·√(N/t)⌋,
# derived from (n, t) and used inside grover_ground_truth.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Ground truth — analytically exact Grover probabilities
# ---------------------------------------------------------------------------

def grover_ground_truth(n: int, marked_states: List[str]) -> Dict[str, float]:
    """
    Return the exact post-measurement probability distribution for a Grover
    circuit on n qubits with the given marked states.

    Formula (equivalent to statevector simulation to <1e-10):
      N     = 2^n
      t     = |marked_states|
      θ     = arcsin(√(t/N))
      k_opt = ⌊(π/4)·√(N/t)⌋
      angle = (2·k_opt + 1)·θ

      p_marked_each   = sin²(angle) / t       ← dividing by t is critical:
                                                 sin²(angle) is the total
                                                 probability for all t marked
                                                 states combined; omitting /t
                                                 is only correct when t=1
      p_unmarked_each = cos²(angle) / (N - t)
    """
    N = 2 ** n
    t = len(marked_states)
    theta = math.asin(math.sqrt(t / N))
    k_opt = int(math.floor(math.pi / 4 * math.sqrt(N / t)))
    angle = (2 * k_opt + 1) * theta

    p_marked_each = math.sin(angle) ** 2 / t
    p_unmarked_each = math.cos(angle) ** 2 / (N - t) if N > t else 0.0

    marked_set = set(marked_states)
    all_states = [format(i, f"0{n}b") for i in range(N)]
    return {
        s: (p_marked_each if s in marked_set else p_unmarked_each)
        for s in all_states
    }


# ---------------------------------------------------------------------------
# Search Accuracy (SA) — equations 1-3 of the paper
# ---------------------------------------------------------------------------

def search_accuracy(
    predicted_probs: Dict[str, float],
    marked_states: List[str],
    tau: float = 0.3,
) -> float:
    """
    Compute Search Accuracy (SA) for a single circuit prediction.

    Algorithm (equations 1-3):
      1. Sort all states by predicted probability descending;
         ties broken by integer interpretation of bit-string ascending.
      2. Take the top-k candidates, where k = |marked_states|.
      3. Filter candidates: keep only those with predicted prob ≥ τ (default 0.3).
      4. SA = |filtered ∩ marked| / k

    Returns a value in [0, 1].
    """
    k = len(marked_states)
    if k == 0:
        return 0.0

    sorted_states = sorted(
        predicted_probs.keys(),
        key=lambda s: (-predicted_probs[s], int(s, 2)),
    )
    top_k = sorted_states[:k]
    filtered = [s for s in top_k if predicted_probs[s] >= tau]
    marked_set = set(marked_states)
    hits = sum(1 for s in filtered if s in marked_set)
    return hits / k


# ---------------------------------------------------------------------------
# Classical Fidelity (CF) — Bhattacharyya coefficient
# ---------------------------------------------------------------------------

# An output whose probabilities sum to more than this is over-summed — not a
# valid distribution as emitted. A real distribution sums to 1; the 0.02 margin
# absorbs the 4-decimal rounding of the model's outputs.
_RENORM_SUM_THRESHOLD = 1.02


def classical_fidelity(
    predicted_probs: Dict[str, float],
    ground_truth_probs: Dict[str, float],
    mode: str = "raw",
) -> float:
    """
    Compute Classical Fidelity (CF) = (Σ_i √(p_i · q_i))²  (paper Eq. 3).

    p_i = predicted probability for state i
    q_i = ground-truth probability for state i (from grover_ground_truth)

    States absent from the model output are assigned p_i = 0.

    `mode` controls how the predicted distribution is treated:

      "raw" (default) — paper-faithful. The model output is used exactly as
        emitted, with no rescaling. This reproduces the authors' metric: an
        over-summed output (Σp > 1) yields CF > 1, and a truncated output
        (Σp < 1) is penalised for its missing mass. The GroverGPT-2 preprint's
        Fig. 4/5 fidelity axes (running up to 1.3) and its text ("close to or
        above 1.0") confirm the authors plot raw, uncapped CF.

      "renormalize" — an over-summed output (Σp above a small rounding margin)
        is rescaled so it totals 1 before CF is computed, keeping CF within
        [0, 1]. Truncated outputs (Σp < 1) are deliberately left untouched:
        renormalising them would scale the listed states up to absorb the
        missing mass, which erases the truncation penalty the paper
        intentionally keeps ("truncation does not artificially inflate
        fidelity scores"). So this mode corrects over-summing only — never
        truncation.

    Returns CF: in [0, 1] for "renormalize"; in [0, ∞) for "raw".
    """
    if mode not in ("raw", "renormalize"):
        raise ValueError(f"mode must be 'raw' or 'renormalize', got {mode!r}")

    # A negative "probability" is invalid; clamp to 0 (also guards the sqrt).
    pred = {s: max(0.0, p) for s, p in predicted_probs.items()}

    if mode == "renormalize":
        total = sum(pred.values())
        # Rescale over-summed outputs only. Truncated outputs (total < 1) are
        # NOT rescaled on purpose: dividing them up to sum 1 would redistribute
        # the missing-state mass onto the listed states and inflate CF, undoing
        # the paper's deliberate truncation penalty.
        if total > _RENORM_SUM_THRESHOLD:
            pred = {s: p / total for s, p in pred.items()}

    all_states = set(pred) | set(ground_truth_probs)
    bc = sum(
        math.sqrt(pred.get(s, 0.0) * ground_truth_probs.get(s, 0.0))
        for s in all_states
    )
    return bc ** 2


# ---------------------------------------------------------------------------
# Compression Ratio (CR) — equation 8
#
# Per circuit:    CR_i  = L_base^(n,i) / L_quantum^(n,i)
# Aggregated:     CR_n  = (1/M_n) Σ_{i=1}^{M_n} CR_i
# ---------------------------------------------------------------------------

def compression_ratio_per_circuit(base_tokens: int, quantum_tokens: int) -> float:
    """CR for a single QASM circuit: L_base / L_quantum."""
    if quantum_tokens == 0:
        raise ValueError("quantum_tokens must be > 0")
    return base_tokens / quantum_tokens


def aggregate_cr(cr_list: List[float]) -> Tuple[float, float]:
    """
    CR_n = (1/M_n) Σ CR_i  (equation 8).

    Returns (mean, std) over the M_n per-circuit CR values.
    """
    if not cr_list:
        raise ValueError("cr_list is empty")
    mean = statistics.mean(cr_list)
    std = statistics.stdev(cr_list) if len(cr_list) > 1 else 0.0
    return mean, std


# ---------------------------------------------------------------------------
# Sequence Reduction Ratio (SRR) — equation 9
#
# Per circuit:    SRR_i = (L_base^(n,i) - L_quantum^(n,i)) / L_base^(n,i)
# Aggregated:     SRR_n = (1/M_n) Σ_{i=1}^{M_n} SRR_i
# ---------------------------------------------------------------------------

def sequence_reduction_ratio_per_circuit(base_tokens: int, quantum_tokens: int) -> float:
    """SRR for a single QASM circuit: (L_base - L_quantum) / L_base."""
    if base_tokens == 0:
        raise ValueError("base_tokens must be > 0")
    return (base_tokens - quantum_tokens) / base_tokens


def aggregate_srr(srr_list: List[float]) -> Tuple[float, float]:
    """
    SRR_n = (1/M_n) Σ SRR_i  (equation 9).

    Returns (mean, std) over the M_n per-circuit SRR values.
    """
    if not srr_list:
        raise ValueError("srr_list is empty")
    mean = statistics.mean(srr_list)
    std = statistics.stdev(srr_list) if len(srr_list) > 1 else 0.0
    return mean, std


# ---------------------------------------------------------------------------
# General aggregation — SA, CF (mean ± std for error bars in plots)
# ---------------------------------------------------------------------------

def aggregate_stats(values: List[float]) -> Tuple[float, float]:
    """Return (mean, std) for a list of per-circuit metric values (SA or CF)."""
    if not values:
        raise ValueError("values list is empty")
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


# ---------------------------------------------------------------------------
# Relative Execution Time — full-circuit inputs
#
# Phase 1: n=2-9    Phase 2: n=2-19
# ---------------------------------------------------------------------------

def relative_execution_time(times_by_n: Dict[int, float]) -> Dict[int, float]:
    """
    S(n) = T(n) / T(2) — relative execution time normalised to n=2.

    Works for any n range (Phase 1: n=2-9, Phase 2, 3: n=2-19).

    Parameters
    ----------
    times_by_n : dict mapping n → mean inference time (seconds) at that qubit count

    Returns
    -------
    dict mapping n → S(n), with T(2) normalised to 1.0
    """
    if 2 not in times_by_n:
        raise ValueError("times_by_n must contain an entry for n=2 (normalisation base)")
    t2 = times_by_n[2]
    if t2 <= 0:
        raise ValueError("T(2) must be positive")
    return {n: t / t2 for n, t in times_by_n.items()}