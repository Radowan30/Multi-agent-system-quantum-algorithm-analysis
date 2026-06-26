"""
Oracle extraction accuracy — post-hoc metric derived from saved CoT outputs.

Measures whether the model correctly identified and emitted the Oracle entity
in its CoT output. Defined as a binary per-circuit score:
  - 1 if the model's extracted oracle block matches the ground-truth oracle body
    after QASM-aware whitespace normalization
  - 0 if unparseable (missing header) or differs

This is the primary metric for the Phase 1/Phase 2 claim that oracle-only inputs
make symbolic analysis trivially easy: oracle-only mode should score ≈1.0 across
n (the model is just copying its input), while full-circuit mode should degrade
with n. The gap quantifies the cognitive load of the extraction step.

Reads two files:
  - outputs_<mode>_<n_min>_<n_max>_<source>.jsonl (saved by eval_phase_1_2.py)
  - circuits_<mode>_<n_min>_<n_max>_<source>.json (the manifest; ground truth)

Writes:
  - oracle_extraction_<mode>_<n_min>_<n_max>_<source>.json — per-n accuracy

Phase-agnostic: works for any phase, model, mode, or n range — auto-detects all
of them from the input files.

Usage:
  python evaluation/oracle_extraction_accuracy.py \\
      --outputs_jsonl evaluation/phase-1/results_alpha16_cl4000/outputs_full_2_9_paper.jsonl \\
      --manifest evaluation/circuit_manifests/circuits_full_2_9_paper.json \\
      --results_dir evaluation/phase-1/results_alpha16_cl4000
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional

_EVAL_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_EVAL_DIR)
sys.path.insert(0, os.path.join(_PROJECT_DIR, "GroverGPT-plus"))

from dataset_generate_MMS import extract_oracle_structure  # noqa: E402

_HEADER_MODEL = "The Oracle entity is extracted below:"


def normalize_oracle_block(text: str) -> List[str]:
    """Canonicalize a QASM oracle body for semantic comparison.

    - Splits on ';' (statement terminator).
    - Within each op: collapses whitespace runs (spaces/tabs/newlines) to a
      single space, then normalizes whitespace around commas to ', '.
    - Drops empty pieces.

    Returns a list of canonical-form operation strings (no trailing semicolon).

    Equivalence semantics (QASM 3.0):
      - leading/trailing whitespace on each line is ignored
      - indentation is ignored
      - blank lines between operations are ignored
      - multiple spaces between tokens are equivalent to one
      - whitespace around commas is normalized
    Differences preserved (treated as not-equal):
      - missing/extra/reordered/replaced operations
      - wrong qubit indices
      - case differences (QASM is case-sensitive: 'x' ≠ 'X')
      - zero whitespace where a token boundary is required
        (e.g. 'mcmt_gate_q_0' is one identifier, not 'mcmt' applied to '_gate_q_0')
      - whitespace inside an identifier
        (e.g. '_gate_q_ 0' tokenizes as two tokens, not one)
    """
    ops = []
    for op_raw in text.split(';'):
        op = ' '.join(op_raw.split())          # collapse all whitespace runs
        op = re.sub(r'\s*,\s*', ', ', op)      # normalize comma spacing
        if op:
            ops.append(op)
    return ops


def parse_model_oracle_block(output_text: str) -> Optional[str]:
    """Extract the oracle-entity block from a model's CoT output.

    Returns the substring between 'The Oracle entity is extracted below:' and
    the next '===' header, or None if the header is missing.
    """
    idx = output_text.find(_HEADER_MODEL)
    if idx == -1:
        return None
    after = output_text[idx + len(_HEADER_MODEL):]
    end = after.find("===")
    return after[:end] if end != -1 else after


def extract_oracle_body(qasm_or_oracle_block: str) -> Optional[str]:
    """Return the body inside the 'gate Oracle ... { ... }' block.

    Works on both forms of QASM that appear in the manifest:
      - Full-circuit QASM (the body is one block among many)
      - Bare oracle block (the manifest's 'qasm' field for mode='oracle')
    Returns None if no 'gate Oracle' definition is found.
    """
    result = extract_oracle_structure(qasm_or_oracle_block)
    # extract_oracle_structure returns (params, ops, oracle_head, oracle_body)
    # if the gate is found, else ([], []).
    if not result or len(result) < 4:
        return None
    return result[3]


def score_extraction(model_block: Optional[str], gt_body: Optional[str]) -> int:
    """Binary score: 1 if normalized forms match, else 0."""
    if not model_block or gt_body is None:
        return 0
    return 1 if normalize_oracle_block(model_block) == normalize_oracle_block(gt_body) else 0


def run_analysis(outputs_jsonl_path: str, manifest_path: str) -> Dict:
    """Compute per-n oracle-extraction accuracy from saved outputs.

    Returns
    -------
    {
      "manifest_path":      <path>,
      "outputs_jsonl_path": <path>,
      "mode":               <str>,
      "circuit_source":     <str>,
      "n_min":              <int>,
      "n_max":              <int>,
      "per_n": {
        "<n>": {
          "accuracy": <float in [0,1]>,
          "correct":  <int>,
          "total":    <int>
        },
        ...
      }
    }
    """
    with open(manifest_path) as f:
        manifest = json.load(f)
    by_file = {
        c["source_file"]: c
        for n_circuits in manifest["by_n"].values()
        for c in n_circuits
    }

    per_n: Dict[int, List[int]] = defaultdict(list)
    n_missing_gt    = 0
    n_missing_model = 0

    with open(outputs_jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            circuit = by_file.get(rec["source_file"])
            if circuit is None:
                # Outputs file references a circuit not in the manifest — skip.
                continue
            n = int(circuit["n"])

            gt_body = extract_oracle_body(circuit["qasm"])
            if gt_body is None:
                n_missing_gt += 1
                continue

            model_block = parse_model_oracle_block(rec["output_text"])
            if model_block is None:
                n_missing_model += 1

            per_n[n].append(score_extraction(model_block, gt_body))

    out = {
        "manifest_path":      manifest_path,
        "outputs_jsonl_path": outputs_jsonl_path,
        "mode":               manifest["mode"],
        "circuit_source":     manifest.get("circuit_source", "strict"),
        "n_min":              manifest["n_min"],
        "n_max":              manifest["n_max"],
        "n_missing_gt":       n_missing_gt,
        "n_missing_model":    n_missing_model,
        "per_n":              {},
    }
    for n in sorted(per_n.keys()):
        scores = per_n[n]
        out["per_n"][str(n)] = {
            "accuracy": sum(scores) / len(scores) if scores else None,
            "correct":  sum(scores),
            "total":    len(scores),
        }
    return out


def _default_output_path(results_dir: str, manifest: Dict) -> str:
    source = manifest.get("circuit_source", "strict")
    return os.path.join(
        results_dir,
        f"oracle_extraction_{manifest['mode']}_{manifest['n_min']}_{manifest['n_max']}_{source}.json",
    )


def main():
    p = argparse.ArgumentParser(description="Compute oracle-extraction accuracy from saved CoT outputs")
    p.add_argument("--outputs_jsonl", required=True,
                   help="Path to the outputs JSONL produced by eval_phase_1_2.py")
    p.add_argument("--manifest", required=True,
                   help="Path to the circuit manifest JSON that drove the evaluation")
    p.add_argument("--results_dir", required=True,
                   help="Directory to write the analysis JSON into")
    p.add_argument("--output_json", default=None,
                   help="Override the output JSON path (default: auto-named)")
    args = p.parse_args()

    print(f"[OracleAcc] outputs : {args.outputs_jsonl}")
    print(f"[OracleAcc] manifest: {args.manifest}")
    result = run_analysis(args.outputs_jsonl, args.manifest)

    if args.output_json:
        out_path = args.output_json
    else:
        with open(args.manifest) as f:
            manifest = json.load(f)
        os.makedirs(args.results_dir, exist_ok=True)
        out_path = _default_output_path(args.results_dir, manifest)

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[OracleAcc] saved   : {out_path}")

    # Console summary
    print(f"\n  mode={result['mode']}  source={result['circuit_source']}  "
          f"n={result['n_min']}..{result['n_max']}")
    if result["n_missing_gt"]:
        print(f"  WARNING: {result['n_missing_gt']} circuits had no extractable Oracle in manifest qasm")
    if result["n_missing_model"]:
        print(f"  note: {result['n_missing_model']} model outputs missing the extraction header")
    print(f"\n  {'n':>4}  {'accuracy':>10}  {'correct/total':>15}")
    for n_str, stats in result["per_n"].items():
        acc = stats["accuracy"]
        acc_s = f"{acc:.3f}" if acc is not None else "  N/A"
        print(f"  {n_str:>4}  {acc_s:>10}  {stats['correct']:>6}/{stats['total']:<6}")


if __name__ == "__main__":
    main()
