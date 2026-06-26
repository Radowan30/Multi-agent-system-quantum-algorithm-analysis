"""
Phase 2 — extend LLaMA 3.1 8B Instruct tokenizer with the same 139 quantum-
native tokens used in Phase 1.

Same tokenization rule as `GroverGPT-plus/extend_tokenizer_data_MMS.py`; only
the source tokenizer path changes (LLaMA 3 → LLaMA 3.1). LLaMA 3.1 ships the
same 128,256-token tiktoken vocabulary as LLaMA 3, so the resulting extended
tokenizer reaches the same 128,395 tokens.

Output: ./Grover_Extend_Tokenizer_data_MMS_llama31/
(then copy tokenizer.json, tokenizer_config.json, special_tokens_map.json
back into ../models/Llama-3.1-8B-Instruct/ before training)
"""

import json
import regex as re
from transformers import AutoTokenizer

ORIGINAL_TOKENIZER_PATH = "/home/quantum-user/radowan/final_year_project/models/Llama-3.1-8B-Instruct"
INPUT_JSON = "/home/quantum-user/radowan/final_year_project/GroverGPT-plus/inputs_list_all_data_MMS.json"
OUTPUT_DIR = "/home/quantum-user/radowan/final_year_project/Fine-tuned Llama 3.1 8B/Grover_Extend_Tokenizer_data_MMS_llama31"


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


print(f"Loading original tokenizer from:\n  {ORIGINAL_TOKENIZER_PATH}\n")
tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_TOKENIZER_PATH, trust_remote_code=True)
base_vocab = tokenizer.get_vocab()
print(f"Base vocab size: {len(base_vocab)}")

print(f"\nLoading QASM circuits from:\n  {INPUT_JSON}")
with open(INPUT_JSON, "r") as f:
    data_list = json.load(f)
print(f"Loaded {len(data_list)} QASM circuits")

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

tokenizer.add_tokens(new_vocab)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nExtended tokenizer saved to:\n  {OUTPUT_DIR}")
print(f"Final vocab size: {len(tokenizer)}")
