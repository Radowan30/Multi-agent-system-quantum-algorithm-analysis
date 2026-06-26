"""
Reads all .qasm files from data_MMS/ and writes their contents to
inputs_list_all_data_MMS.json — a drop-in replacement for inputs_list_all_0.json
that covers all available circuits instead of just 14 samples.
"""

import os
import json

DATA_DIR = os.path.join(os.path.dirname(__file__), "data_MMS")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "inputs_list_all_data_MMS.json")

qasm_texts = []
file_count = 0

for root, dirs, files in os.walk(DATA_DIR):
    for fname in sorted(files):
        if fname.endswith(".qasm"):
            with open(os.path.join(root, fname), "r") as f:
                qasm_texts.append(f.read())
            file_count += 1

with open(OUTPUT_FILE, "w") as f:
    json.dump(qasm_texts, f)

print(f"Collected {file_count} QASM files from {DATA_DIR}")
print(f"Saved to {OUTPUT_FILE}")
