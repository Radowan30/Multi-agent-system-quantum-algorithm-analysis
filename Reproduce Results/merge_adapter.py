"""
Merges the GroverGPT+ LoRA adapter into the base model, producing a
standalone model directory that vLLM can load directly.

Run this once in venv-train before running any inference scripts.

Output: saves/Meta-Llama-3-8B-Instruct/merged/GroverGPT+/
"""

import json
import os
import struct
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_PATH = "/home/quantum-user/radowan/final_year_project/models/Meta-Llama-3-8B-Instruct"
ADAPTER_PATH    = "/home/quantum-user/radowan/final_year_project/saves/Meta-Llama-3-8B-Instruct/lora/GroverGPT+"
OUTPUT_PATH     = "/home/quantum-user/radowan/final_year_project/saves/Meta-Llama-3-8B-Instruct/merged/GroverGPT+"


def get_adapter_vocab_size(adapter_path):
    """
    Read the embedding vocabulary size directly from the adapter safetensors
    header — without loading any tensor data into memory.

    This is necessary because len(tokenizer) does not match the adapter's
    actual embedding size. During training LLaMA-Factory:
      1. Added a pad token to the tokenizer:  128395 → 128396
      2. Padded the embedding matrix to the next multiple of 64: 128396 → 128448
    The tokenizer reports 128396; the adapter weights contain 128448.
    We must resize to 128448 or PEFT will raise a size mismatch error.
    """
    sf_path = os.path.join(adapter_path, "adapter_model.safetensors")
    with open(sf_path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    # Find the embedding key — scan for it rather than hardcode,
    # since the exact name varies across PEFT/LLaMA-Factory versions.
    for key, meta in header.items():
        if isinstance(meta, dict) and "embed_tokens" in key and "shape" in meta:
            return meta["shape"][0]
    raise KeyError("Could not find an embed_tokens tensor in the adapter safetensors.")


print("Loading base model on CPU (bfloat16) ...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="cpu",   # keep on CPU to avoid holding two copies in VRAM
)

print("Loading extended tokenizer from adapter ...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

# Resize to the actual embedding size stored in the adapter weights.
# len(tokenizer) = 128396, but the adapter embedding matrix is 128448
# (see get_adapter_vocab_size docstring for why they differ).
target_vocab_size = get_adapter_vocab_size(ADAPTER_PATH)
print(f"Resizing base model vocabulary: {base_model.config.vocab_size} → {target_vocab_size}")
print(f"  tokenizer reports {len(tokenizer)} tokens; adapter weights require {target_vocab_size}")
base_model.resize_token_embeddings(target_vocab_size)

print("Loading PEFT adapter ...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

print("Merging LoRA weights into base model ...")
merged_model = model.merge_and_unload()

print(f"\nSaving merged model to:\n  {OUTPUT_PATH}")
merged_model.save_pretrained(OUTPUT_PATH, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_PATH)

print("\nDone.")
print(f"  Model vocab size: {merged_model.config.vocab_size}")
print(f"  Tokenizer vocab size: {len(tokenizer)}")
