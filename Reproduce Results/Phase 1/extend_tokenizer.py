"""
Extends the LLaMA tokenizer with quantum-specific tokens derived from all
QASM files in data_MMS/ via inputs_list_all_data_MMS.json.

Equivalent to running extend_tokenizer.ipynb but using the full data_MMS
corpus instead of the 14-sample inputs_list_all_0.json.

Output: ./Grover_Extend_Tokenizer_data_MMS/
"""

import json
import regex as re
from transformers import AutoTokenizer

# --- Paths ---
# Load from the full model folder (not just original_tokenizer_backup).
# The model folder has config.json which tells AutoTokenizer to use LlamaTokenizerFast,
# producing a correct full tokenizer_config.json with added_tokens_decoder etc.
# Before running this script, restore the original tokenizer files from backup:
#   cp .../original_tokenizer_backup/* .../models/Meta-Llama-3-8B-Instruct/
ORIGINAL_TOKENIZER_PATH = "/home/quantum-user/radowan/final_year_project/models/Meta-Llama-3-8B-Instruct"
INPUT_JSON = "/home/quantum-user/radowan/final_year_project/GroverGPT-plus/inputs_list_all_data_MMS.json"
OUTPUT_DIR = "/home/quantum-user/radowan/final_year_project/GroverGPT-plus/Grover_Extend_Tokenizer_data_MMS"


# --- Exact same tokenization rule as extend_tokenizer.ipynb Cell 7 ---
def _tokenize_line(command):
    command = command.strip()
    if not command:
        return []

    if command.startswith("gate"):
        gate_match = re.match(r"gate\s+(\w+)(?:\s*\((.*?)\))?\s+([^{]+)\s*{", command)
        if not gate_match:
            raise SyntaxError(f"Invalid gate definition: {command}")
        gate_name = gate_match.group(1)
        params_part = gate_match.group(2) or ""
        qubits_part = gate_match.group(3)
        params = [p.strip() for p in params_part.split(",") if p.strip()]
        qubits = [q.strip() for q in qubits_part.split(",") if q.strip()]
        tokens = ["gate", gate_name] + params + qubits + ["{"]
        tokens = [re.sub(r'^(_gate_q_|unitary_|mcx_vchain_)\d+$', r'\1', t) for t in tokens]
        return tokens

    groups = re.match(r"^(\w+)(?:\((.*?)\))?\s+([^;]+);", command)
    if groups:
        op_name = groups.group(1)
        params = groups.group(2)
        targets = groups.group(3)
        tokens = [op_name]
        if params:
            tokens += ["("] + [p.strip() for p in params.split(",")] + [")"]
        tokens += [t.strip() for t in targets.split(",")]
        tokens = [token for token in tokens if token]
        tokens = [re.sub(r'^(_gate_q_|unitary_|mcx_vchain_)\d+$', r'\1', t) for t in tokens]
        return tokens

    if command == "}":
        return ["}"]

    raise SyntaxError(f"Unrecognized command: {command}")


# --- Step 1: Load original base tokenizer ---
print(f"Loading original tokenizer from:\n  {ORIGINAL_TOKENIZER_PATH}\n")
tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_TOKENIZER_PATH, trust_remote_code=True)
base_vocab = tokenizer.get_vocab()
print(f"Base vocab size: {len(base_vocab)}")

# --- Step 2: Load all QASM texts from JSON ---
print(f"\nLoading QASM circuits from:\n  {INPUT_JSON}")
with open(INPUT_JSON, "r") as f:
    data_list = json.load(f)
print(f"Loaded {len(data_list)} QASM circuits")

# --- Step 3: Collect new tokens (same logic as Cell 8) ---
new_vocab = []
errors = 0

for data_item in data_list:
    lines = [line.strip() for line in data_item.split("\n") if line.strip()]
    total_tokens = []
    for line in lines:
        try:
            tokens = _tokenize_line(line)
            total_tokens.extend(tokens)
        except SyntaxError:
            errors += 1
            continue
    for token in total_tokens:
        if token and token not in base_vocab and token not in new_vocab:
            new_vocab.append(token)

print(f"\nNew tokens not in base vocab: {len(new_vocab)}")
print(f"Tokenization errors skipped:  {errors}")
print(f"\nFull list of new tokens:")
for i, t in enumerate(new_vocab):
    print(f"  {i+1:3d}: {t!r}")

# --- Step 4: Add tokens and save ---
tokenizer.add_tokens(new_vocab)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nExtended tokenizer saved to:\n  {OUTPUT_DIR}")
print(f"New vocab size: {len(tokenizer)}")
