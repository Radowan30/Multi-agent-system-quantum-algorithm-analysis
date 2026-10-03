"""
Register the Grover datasets with LLaMA-Factory by merging
llamafactory_dataset_info_grover_only.json into LLaMA-Factory's
data/dataset_info.json (existing entries are kept; Grover entries are
added or updated).

Usage:
    python register_datasets.py [--dataset_info $WORK/LLaMA-Factory/data/dataset_info.json]
"""

import argparse
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    work = os.environ.get("WORK")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_info",
                    default=os.path.join(work, "LLaMA-Factory", "data", "dataset_info.json") if work else None,
                    help="LLaMA-Factory's data/dataset_info.json (default: under $WORK)")
    args = ap.parse_args()
    if not args.dataset_info:
        ap.error("set WORK, or pass --dataset_info")

    with open(os.path.join(_HERE, "llamafactory_dataset_info_grover_only.json")) as f:
        grover = json.load(f)
    with open(args.dataset_info) as f:
        info = json.load(f)
    info.update(grover)
    with open(args.dataset_info, "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"Registered {len(grover)} Grover datasets in {args.dataset_info}:")
    for name in grover:
        print(f"  {name}")


if __name__ == "__main__":
    main()
