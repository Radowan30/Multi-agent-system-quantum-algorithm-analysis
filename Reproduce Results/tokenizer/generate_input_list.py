"""
Collect every .qasm file in data_MMS into one JSON list of QASM strings.

This is the corpus the quantum-native vocabulary is derived from
(prepare_quantum_model.py --inputs_list). It replaces the 14-sample
inputs_list_all_0.json of the original GroverGPT+ release, which is why our
extended tokenizer has 139 new entries instead of 96.

Usage:
    python generate_input_list.py                      # reads $WORK/data_MMS
    python generate_input_list.py --data_dir <data_MMS> --output <file.json>

Default output: $WORK/tokenizer/inputs_list_all_data_MMS.json
"""

import argparse
import json
import os


def main():
    work = os.environ.get("WORK")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=os.path.join(work, "data_MMS") if work else None,
                    help="data_MMS folder (default: $WORK/data_MMS)")
    ap.add_argument("--output",
                    default=os.path.join(work, "tokenizer", "inputs_list_all_data_MMS.json") if work else None,
                    help="Output JSON (default: $WORK/tokenizer/inputs_list_all_data_MMS.json)")
    args = ap.parse_args()
    if not args.data_dir or not args.output:
        ap.error("set WORK, or pass --data_dir and --output")

    qasm_texts = []
    for root, dirs, files in os.walk(args.data_dir):
        dirs.sort()                       # deterministic traversal order
        for fname in sorted(files):
            if fname.endswith(".qasm"):
                with open(os.path.join(root, fname), "r") as f:
                    qasm_texts.append(f.read())

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(qasm_texts, f)

    print(f"Collected {len(qasm_texts)} QASM files from {args.data_dir}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
