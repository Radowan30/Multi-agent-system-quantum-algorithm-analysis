"""
Circuit utilities shared across all three evaluation phases.

Two evaluation runs (Research_Plan.md Section 3.3):
  ┌──────────────┬─────────┬──────────────────────────────────────────────────┐
  │ Run          │ n range │ Metrics                                          │
  ├──────────────┼─────────┼──────────────────────────────────────────────────┤
  │ Full circuit │  2 – 9  │ SA, CF, CR, SRR, relative execution time         │
  │ Oracle-only  │  2 – 20 │ SA, CF  (one graph; training boundary at n=10)   │
  └──────────────┴─────────┴──────────────────────────────────────────────────┘

Circuit sources (fresh vs data_MMS):
  Full circuit  n=2-7  (in training) → generate fresh, save to data_MMS_eval/
  Full circuit  n=8-9  (OOD)         → load from GroverGPT-plus/data_MMS/
  Oracle-only   n=2-10 (in training) → generate fresh, save to data_MMS_eval/
  Oracle-only   n=11-20 (OOD)        → load from GroverGPT-plus/data_MMS/

The same training-range boundaries apply in Phases 2 and 3. As for evaluation during phases 2 and 3, the Full circuit n will range from 2-19 qubits (because we are using the LLaMA 3.1 8B Instruct model which has enough context window to handle up to 19 qubits), and we will evaluate the Oracle-only circuits for n=2-20 qubits only in Phases 1 and 2.

Fresh circuits are saved to disk so evaluation runs are fully reproducible.
"""

import math
import os
import re
import glob
import random
from itertools import combinations
from typing import List, Dict

from qiskit import QuantumCircuit
from qiskit.circuit.library import MCMT, ZGate
import qiskit.qasm3 as qasm3

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_MMS_EVAL_DIR = os.path.join(_EVAL_DIR, "data_MMS_eval")
DATA_MMS_DIR = os.path.join(_EVAL_DIR, "..", "GroverGPT-plus", "data_MMS")

# Evaluation n ranges (Research_Plan Section 3.3)
FULL_CIRCUIT_N_MIN, FULL_CIRCUIT_N_MAX = 2, 9
ORACLE_N_MIN,       ORACLE_N_MAX       = 2, 20

# Training-range boundaries — circuits with n below these thresholds were
# included in the GroverGPT+ training set, so we generate fresh circuits for
# them instead of loading from data_MMS (to avoid testing on training data).
#   Full circuit training: Grover_FullCircuit_2_7_MMS.json  → n=2..7
#   Oracle training:       Grover_Oracle_2_10_MMS.json      → n=2..10
FULL_CIRCUIT_FRESH_MAX_N = 8    # n < 8  (n=2..7) → fresh; n ≥ 8  → data_MMS
ORACLE_FRESH_MAX_N       = 11   # n < 11 (n=2..10) → fresh; n ≥ 11 → data_MMS


# ---------------------------------------------------------------------------
# Grover circuit construction — Qiskit 1.4.5
#
# Confirmed to reproduce all data_MMS circuits exactly (4955/4956 match;
# the single failure is n=24 k=2, which is outside the evaluation range n≤20).
# ---------------------------------------------------------------------------

def _oracle_gate(n: int, marked_states: List[str]):
    """
    Build the Oracle gate for the given marked states.

    For each marked state we place X gates on every qubit that holds '0' in
    that state, then apply MCMT(Z, n-1 controls, 1 target), then undo the X
    gates.  The X-MCMT-X sandwich makes the phase kick happen only when the
    register matches the target state.

    Qiskit uses little-endian qubit ordering internally, so we reverse the
    state string before iterating: position 0 in the reversed string maps to
    physical qubit 0.

    MCMT(ZGate(), n-1, 1) emits a named 'mcmt' gate in QASM 3.0 output,
    which is exactly what appears in the data_MMS oracle bodies.
    """
    oracle = QuantumCircuit(n)
    for ms in marked_states:
        rev = ms[::-1]
        zero_qubits = [i for i, c in enumerate(rev) if c == "0"]
        if zero_qubits:
            oracle.x(zero_qubits)
        oracle.compose(MCMT(ZGate(), n - 1, 1), inplace=True)
        if zero_qubits:
            oracle.x(zero_qubits)
    gate = oracle.to_gate()
    gate.name = "Oracle"
    return gate


def _diffuser_gate(n: int):
    """
    Build the Grover diffuser gate manually.

    The diffuser reflects amplitudes about the uniform superposition:
        H^n  X^n  (n-controlled Z)  X^n  H^n

    The n-controlled Z is decomposed as H on the target qubit, an
    n-1-controlled X on that same qubit, then H again.  This is the standard
    identity CZ = (I⊗H) CX (I⊗H).

    We build the diffuser by hand (not with MCMT) because using MCMT here
    would emit a 'mcmt' gate in the QASM output.  The data_MMS diffuser uses
    the explicit h/ccx/mcx decomposition, so we must match it manually.
    """
    d = QuantumCircuit(n)
    d.h(range(n))
    d.x(range(n))
    d.h(n - 1)                         # H on target before controlled-X
    if n == 2:
        d.cx(0, 1)                      # 1-controlled X  (CX / CNOT)
    elif n == 3:
        d.ccx(0, 1, 2)                  # 2-controlled X  (Toffoli)
    else:
        d.mcx(list(range(n - 1)), n - 1)  # (n-1)-controlled X
    d.h(n - 1)                         # H on target after controlled-X
    d.x(range(n))
    d.h(range(n))
    gate = d.to_gate()
    gate.name = "Diffuser"
    return gate


def _k_opt(n: int, t: int) -> int:
    """Optimal Grover iteration count: ⌊(π/4)·√(N/t)⌋, N = 2^n."""
    return int(math.floor(math.pi / 4 * math.sqrt(2 ** n / t)))


def generate_full_circuit_qasm(n: int, marked_states: List[str]) -> str:
    """Return QASM 3.0 string for a complete Grover circuit."""
    t = len(marked_states)
    k = _k_opt(n, t)
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    for _ in range(k):
        qc.append(_oracle_gate(n, marked_states), range(n))
        qc.append(_diffuser_gate(n), range(n))
    qc.measure(range(n), range(n))
    return qasm3.dumps(qc)


def generate_oracle_only_qasm(n: int, marked_states: List[str]) -> str:
    """
    Return the Oracle gate definition extracted from a complete Grover circuit.

    The format matches the oracle input used in training:
    just the 'gate Oracle { ... }' block — no QASM header, no standalone circuit.
    """
    full_qasm = generate_full_circuit_qasm(n, marked_states)
    return extract_oracle_qasm(full_qasm)


# ---------------------------------------------------------------------------
# k (marked-state count) distribution — matches data_MMS
# ---------------------------------------------------------------------------

def _max_k_for_n(n: int) -> int:
    """
    Maximum k (number of marked states) used at qubit count n, matching data_MMS.

    data_MMS caps at k=3 for all n≥4.  Smaller n are capped lower because
    marking too large a fraction of states makes the algorithm trivial:
      n=2 → k=1 only  (k=2 would mark half of the 4 states)
      n=3 → k=1,2     (k=3 would give only 56 unseen circuits, too few for
                        meaningful evaluation — matching data_MMS design)
      n≥4 → k=1,2,3

    Formula: min(3, n-1).  Gives 1, 2, 3, 3, 3, … for n=2, 3, 4, 5, 6, …
    """
    return min(3, n - 1)


def _per_k_cap() -> int:
    """
    Cap on circuits per k value, matching data_MMS.

    data_MMS includes every k=1 combination when the total fits within 100,
    and caps k=2 and k=3 at 100 each where the combinatorial space is larger.
    Once n≥7, k=1 itself exceeds 100 (2^7=128 states), so it is also capped.
    """
    return 100


# Maximum unseen circuits generated per (n, k) level during fresh generation.
# 500 per k gives a pool of up to ~1 500 per n — well above the evaluation
# target of 100–300 — while keeping generation times practical for k=3 at
# large n (where tens of thousands of unseen combos exist).
_MAX_UNSEEN_PER_K = 500


def _total_available_circuits(n: int) -> int:
    """
    Total circuits obtainable for n qubits under the data_MMS sampling rule
    (all k=1 up to 100, plus up to 100 each for k=2 and k=3 when applicable).
    """
    N = 2 ** n
    cap = _per_k_cap()
    total = 0
    for k in range(1, _max_k_for_n(n) + 1):
        total += min(math.comb(N, k), cap)
    return total


def target_circuit_count(n: int) -> int:
    """
    Number of evaluation circuits for n qubits.

    Formula from Research_Plan Section 3.3: max(100, 2^n), bounded above by
    the circuits actually available (from data_MMS or combinatorial space).

    Results: n=2→4, n=3→36, n=4–6→100, n=7→128, n=8→256, n≥9→300.
    (n≥9 caps at 300 because that is how many circuits data_MMS holds per n.)
    """
    available = _total_available_circuits(n)
    desired = max(100, 2 ** n)
    return min(desired, available)


# ---------------------------------------------------------------------------
# Filename helpers — same convention as data_MMS
# ---------------------------------------------------------------------------

def _states_to_filename(n: int, k: int, marked_states: List[str]) -> str:
    """grover_n{n}_k{k}_m{s1}_{s2}...qasm  (marked states joined by underscore)."""
    return f"grover_n{n}_k{k}_m{'_'.join(marked_states)}.qasm"


def _filename_to_marked_states(fname: str) -> List[str]:
    """Inverse of _states_to_filename: parse marked states from filename."""
    base = os.path.splitext(os.path.basename(fname))[0]
    m = re.match(r"grover_n\d+_k\d+_m(.+)", base)
    return m.group(1).split("_") if m else []


# ---------------------------------------------------------------------------
# Fresh circuit generation — for in-training-range n values
# ---------------------------------------------------------------------------

def _data_mms_combos(n: int) -> set:
    """Return the frozensets of marked states present in data_MMS for n."""
    folder = os.path.join(DATA_MMS_DIR, f"grover_n{n}")
    if not os.path.isdir(folder):
        return set()
    seen = set()
    for fname in os.listdir(folder):
        if fname.endswith(".qasm"):
            states = _filename_to_marked_states(fname)
            if states:
                seen.add(frozenset(states))
    return seen


def generate_and_save_fresh_circuits(n: int, mode: str, seed: int = 42) -> List[Dict]:
    """
    Generate ALL evaluation circuits for in-training-range n, save to
    data_MMS_eval/{mode}/grover_n{n}/, and return the full list.

    For each k from 1 to _max_k_for_n(n):
      - Enumerate every combination of k marked states NOT present in
        data_MMS (the complete training-set complement for that k level).
      - Generate and save QASM for all of them, up to _MAX_UNSEEN_PER_K
        per k (keeps generation practical when the unseen pool is large,
        e.g. k=3 at n ≥ 6 where tens of thousands of unseen combos exist).
      - Skip k levels that are fully exhausted in data_MMS.

    Fallback — n=2 and n=3 only:
      All combinations for these n values are exhausted in data_MMS (k=1
      only at n=2; k=1,2 at n=3).  These circuits are therefore used as-is,
      accepting the training-data overlap.  This is scientifically valid:
      the n=2 and n=3 points anchor the left side of the SA/CF curve and
      confirm that training succeeded; generalisation is assessed at n≥8
      for full-circuit and n≥11 for oracle-only evaluation.
      (See Research_Plan Section 3.3 and expert review 2026-05-15.)

    Saved to data_MMS_eval/{mode}/ so full and oracle modes never collide
    on the same filename.  Idempotent: existing files are not rewritten.

    The caller (build_manifest) samples target_circuit_count(n) circuits
    from the returned pool for the evaluation manifest.

    Parameters
    ----------
    n    : number of qubits
    mode : 'full'   → complete Grover circuit QASM
           'oracle' → Oracle gate definition block (matching training format)
    seed : random seed for reproducible sampling when unseen > _MAX_UNSEEN_PER_K
    """
    out_dir = os.path.join(DATA_MMS_EVAL_DIR, mode, f"grover_n{n}")
    os.makedirs(out_dir, exist_ok=True)

    N = 2 ** n
    rng = random.Random(seed)
    in_training = _data_mms_combos(n)
    all_states = [format(i, f"0{n}b") for i in range(N)]

    circuits = []
    any_unseen_found = False

    for k in range(1, _max_k_for_n(n) + 1):
        all_combos = list(combinations(all_states, k))
        unseen = [c for c in all_combos if frozenset(c) not in in_training]

        if not unseen:
            continue  # k level fully exhausted in data_MMS — skip entirely

        any_unseen_found = True
        rng.shuffle(unseen)

        for combo in unseen[:_MAX_UNSEEN_PER_K]:
            marked = list(combo)
            fname = _states_to_filename(n, k, marked)
            fpath = os.path.join(out_dir, fname)

            if os.path.exists(fpath):
                with open(fpath) as f:
                    qasm_str = f.read()
            else:
                qasm_str = (
                    generate_full_circuit_qasm(n, marked)
                    if mode == "full"
                    else generate_oracle_only_qasm(n, marked)
                )
                with open(fpath, "w") as f:
                    f.write(qasm_str)

            circuits.append({
                "n": n,
                "k": k,
                "marked_states": marked,
                "qasm": qasm_str,
                "source_file": fname,
            })

    # Fallback: n=2 (k=1 only, 4 circuits) and n=3 (k=1,2, 36 circuits).
    # All combinations within the paper's k-distribution are in training data.
    # Use them anyway — training-data overlap is unavoidable and accepted here.
    if not any_unseen_found:
        all_combos = []
        for k in range(1, _max_k_for_n(n) + 1):
            all_combos.extend(combinations(all_states, k))
        rng.shuffle(all_combos)
        for combo in all_combos:
            k = len(combo)
            marked = list(combo)
            fname = _states_to_filename(n, k, marked)
            fpath = os.path.join(out_dir, fname)
            if os.path.exists(fpath):
                with open(fpath) as f:
                    qasm_str = f.read()
            else:
                qasm_str = (
                    generate_full_circuit_qasm(n, marked)
                    if mode == "full"
                    else generate_oracle_only_qasm(n, marked)
                )
                with open(fpath, "w") as f:
                    f.write(qasm_str)
            circuits.append({
                "n": n,
                "k": k,
                "marked_states": marked,
                "qasm": qasm_str,
                "source_file": fname,
            })

    return circuits


# ---------------------------------------------------------------------------
# data_MMS loader — for out-of-training-range n values
# ---------------------------------------------------------------------------

def load_data_mms_circuits(n: int, mode: str, seed: int = 42) -> List[Dict]:
    """
    Load evaluation circuits from data_MMS for out-of-training-range n.

    Selects target_circuit_count(n) files at random (reproducible via seed).
    For mode='oracle', the oracle subcircuit is extracted from the full QASM.
    """
    folder = os.path.join(DATA_MMS_DIR, f"grover_n{n}")
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"data_MMS folder not found: {folder}")

    all_files = sorted(glob.glob(os.path.join(folder, "*.qasm")))
    rng = random.Random(seed)
    rng.shuffle(all_files)
    selected = all_files[:target_circuit_count(n)]

    circuits = []
    for fpath in selected:
        marked = _filename_to_marked_states(fpath)
        with open(fpath) as f:
            full_qasm = f.read()
        qasm_str = full_qasm if mode == "full" else extract_oracle_qasm(full_qasm)
        circuits.append({
            "n": n,
            "k": len(marked),
            "marked_states": marked,
            "qasm": qasm_str,
            "source_file": os.path.basename(fpath),
        })
    return circuits


# ---------------------------------------------------------------------------
# Oracle extraction from a full-circuit QASM string
# ---------------------------------------------------------------------------

def extract_oracle_qasm(full_qasm: str) -> str:
    """
    Extract the Oracle gate definition from a full Grover circuit QASM string.

    Returns just the 'gate Oracle ... { ... }' block — no QASM header, no
    standalone circuit wrapper.  This matches the oracle input format used
    during training (see GroverGPT-plus/dataset_generate_MMS.py).

    Used both by generate_oracle_only_qasm (fresh circuits) and
    load_data_mms_circuits (OOD circuits loaded from data_MMS).
    """
    lines = full_qasm.split("\n")
    oracle_block = []
    in_oracle = False
    brace_count = 0
    for line in lines:
        if line.strip().startswith("gate Oracle"):
            in_oracle = True
            brace_count = 0
        if in_oracle:
            oracle_block.append(line)
            brace_count += line.count("{")
            brace_count -= line.count("}")
            if brace_count == 0 and line.strip().endswith("}"):
                return "\n".join(oracle_block)
    raise ValueError("No Oracle gate definition found in QASM")


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def get_circuits(n: int, mode: str, seed: int = 42,
                 circuit_source: str = "strict") -> List[Dict]:
    """
    Return evaluation circuits for the given n and mode.

    Parameters
    ----------
    n              : number of qubits
    mode           : 'full'   → complete Grover circuit QASM
                     'oracle' → oracle-only QASM
    seed           : random seed for reproducible circuit selection
    circuit_source : 'strict' — fresh unseen circuits for in-training-range n,
                                data_MMS circuits for OOD n (more rigorous;
                                guarantees no test/train overlap for n≥4).
                     'paper'  — data_MMS circuits for ALL n, matching the
                                original GroverGPT+ paper evaluation approach.
                                Use to reproduce published figures directly.
    """
    if mode not in ("full", "oracle"):
        raise ValueError(f"mode must be 'full' or 'oracle', got {mode!r}")
    if circuit_source not in ("strict", "paper"):
        raise ValueError(f"circuit_source must be 'strict' or 'paper', got {circuit_source!r}")

    in_fresh_range = (
        (mode == "full"   and n < FULL_CIRCUIT_FRESH_MAX_N) or
        (mode == "oracle" and n < ORACLE_FRESH_MAX_N)
    )
    if circuit_source == "strict" and in_fresh_range:
        return generate_and_save_fresh_circuits(n, mode, seed)
    else:
        return load_data_mms_circuits(n, mode, seed)
