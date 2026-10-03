"""
Build the monolithic-equivalent-data dataset for the Phase 4 ablation.

Each example pairs the same bare-QASM input that Agent 1 receives with a
target output that is the exact character-for-character concatenation of
the three Phase 4 agent targets for that circuit:

    output = Agent1_output + Agent2_output + Agent3_output

No separator is inserted, so the monolithic target contains exactly the
same characters as the three agent targets combined, and (verified below)
tokenizes to exactly the same number of tokens. A single model trained on
this data sees the same target content as the Phase 4 multi-agent system;
the only difference is that it must produce all three stages in one
generation instead of three separate calls.

The three agent datasets are index-aligned (entry i of each file is the
same circuit). This is asserted below by checking that the oracle block at
the end of Agent 1's output equals Agent 2's input and that the final
marked-state block of Agent 2's output equals Agent 3's input.

Usage (defaults shown; run with venv-train or venv-eval):
    python generate_monolithic_dataset.py \\
        --agent_dir "../dataset" \\
        --tokenizer $WORK/models/Llama-3-8B-Instruct-262k-quantum \\
        --out $WORK/LLaMA-Factory/data/Grover_Monolithic_2_7_MMS.json
"""

import argparse
import json
import os
from pathlib import Path

A1_ORACLE_MARKER = "=== Extracted Oracle Gate Definition ==="
A2_FINAL_MARKER = "=== Final Marked States ==="


def _tail_from(text: str, marker: str) -> str:
    idx = text.index(marker)
    return text[idx:].strip()


def main():
    ap = argparse.ArgumentParser()
    work = os.environ.get("WORK")
    here = Path(__file__).resolve().parent
    ap.add_argument("--agent_dir", default=str(here.parent / "dataset"),
                    help="Folder holding Grover_Agent{1,2,3}_2_7_MMS.json (default: ../dataset)")
    ap.add_argument("--tokenizer",
                    default=os.path.join(work, "models", "Llama-3-8B-Instruct-262k-quantum") if work else None,
                    help="Extended quantum-native tokenizer used for the token-count check "
                         "(default: $WORK/models/Llama-3-8B-Instruct-262k-quantum)")
    ap.add_argument("--out",
                    default=os.path.join(work, "LLaMA-Factory", "data", "Grover_Monolithic_2_7_MMS.json") if work else None,
                    help="Output JSON (default: $WORK/LLaMA-Factory/data/Grover_Monolithic_2_7_MMS.json)")
    args = ap.parse_args()
    if not args.tokenizer or not args.out:
        ap.error("set WORK, or pass --tokenizer and --out")

    agent_dir = Path(args.agent_dir)
    a1, a2, a3 = (json.loads((agent_dir / f"Grover_Agent{i}_2_7_MMS.json").read_text())
                  for i in (1, 2, 3))
    assert len(a1) == len(a2) == len(a3), "agent datasets have different lengths"

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    def n_tokens(s: str) -> int:
        return len(tok(s, add_special_tokens=False)["input_ids"])

    records = []
    for i, (e1, e2, e3) in enumerate(zip(a1, a2, a3)):
        # Alignment: each agent's input is the previous agent's final block.
        assert _tail_from(e1["output"], A1_ORACLE_MARKER) == e2["input"].strip(), \
            f"entry {i}: Agent 1 oracle != Agent 2 input"
        assert _tail_from(e2["output"], A2_FINAL_MARKER) == e3["input"].strip(), \
            f"entry {i}: Agent 2 marked states != Agent 3 input"

        combined = e1["output"] + e2["output"] + e3["output"]

        # Exactness: same characters and same token count as the three parts.
        assert combined == "".join((e1["output"], e2["output"], e3["output"]))
        parts_tokens = n_tokens(e1["output"]) + n_tokens(e2["output"]) + n_tokens(e3["output"])
        assert n_tokens(combined) == parts_tokens, \
            f"entry {i}: combined output tokenizes differently from its parts"

        records.append({
            "instruction": e1["instruction"],   # empty, as in Phase 4
            "input": e1["input"],               # bare QASM, as for Agent 1
            "output": combined,
        })

    Path(args.out).write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"Wrote {len(records)} examples -> {args.out}")
    print("Verified: agent alignment, identical characters, identical token counts.")


if __name__ == "__main__":
    main()
