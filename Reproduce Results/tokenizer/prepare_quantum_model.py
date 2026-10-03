"""
Prepare a base model for fine-tuning: add the 139 quantum-native tokens and set
the chat template, writing the result to a NEW directory. The downloaded base
model is never modified, because Phase 3 also evaluates the untouched
("untrained") base models and the evaluation scripts use their original
tokenizer as the CR/SRR baseline.

What the output directory contains:
  * the base tokenizer extended with the tokens in quantum_tokens.json, added in
    that exact order so the new token IDs (128256-128394) match ours;
  * the simple LLaMA 3 chat template (llama3_chat_template.jinja), which is the
    template LLaMA-Factory's `llama3` template trains with. LLaMA 3 8B Instruct
    and Llama-3-8B-Instruct-262k already ship it; LLaMA 3.1 8B Instruct ships a
    longer template with a default system prompt, which would not match training;
  * config.json and generation_config.json. For LLaMA 3.1 8B Instruct the end-of-
    sequence ids are set to 128009 (config) and [128001, 128009] (generation),
    the same as LLaMA 3 8B Instruct, so <|eom_id|> (128008) is not a stop token;
  * symbolic links to the base model's weight files (use --copy_weights to copy).

The token list was derived with the GroverGPT+ tokenisation rule from every
circuit in data_MMS. Pass --inputs_list (built by generate_input_list.py) to
recompute that set and check it matches quantum_tokens.json.

Usage:
    python prepare_quantum_model.py \\
        --base_model  $WORK/models/Llama-3.1-8B-Instruct \\
        --output_dir  $WORK/models/Llama-3.1-8B-Instruct-quantum \\
        [--inputs_list $WORK/tokenizer/inputs_list_all_data_MMS.json]
"""

import argparse
import glob
import json
import os
import shutil

import regex as re
from transformers import AutoTokenizer

_HERE = os.path.dirname(os.path.abspath(__file__))
TOKENS_FILE = os.path.join(_HERE, "quantum_tokens.json")
TEMPLATE_FILE = os.path.join(_HERE, "llama3_chat_template.jinja")
EXPECTED_PROMPT = ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\ntest<|eot_id|>"
                   "<|start_header_id|>assistant<|end_header_id|>\n\n")


# --- GroverGPT+ tokenisation rule (extend_tokenizer.ipynb, Cell 7) ---
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


def recompute_tokens(inputs_list, base_vocab):
    """New tokens (not in the base vocabulary) found in the QASM corpus."""
    with open(inputs_list) as f:
        data_list = json.load(f)
    new_tokens, errors = set(), 0
    for qasm in data_list:
        for line in (l.strip() for l in qasm.split("\n") if l.strip()):
            try:
                tokens = _tokenize_line(line)
            except SyntaxError:
                errors += 1
                continue
            new_tokens.update(t for t in tokens if t and t not in base_vocab)
    return new_tokens, len(data_list), errors


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_model", required=True, help="Downloaded base model directory (not modified)")
    ap.add_argument("--output_dir", required=True, help="New directory for the prepared model")
    ap.add_argument("--inputs_list", default=None,
                    help="QASM corpus JSON from generate_input_list.py; if given, the token "
                         "set is recomputed and checked against quantum_tokens.json")
    ap.add_argument("--copy_weights", action="store_true",
                    help="Copy the weight files instead of linking to them")
    args = ap.parse_args()

    base, out = os.path.abspath(args.base_model), os.path.abspath(args.output_dir)
    if base == out:
        ap.error("--output_dir must differ from --base_model (the base model must stay unmodified)")

    with open(TOKENS_FILE) as f:
        tokens = json.load(f)

    print(f"Loading base tokenizer from {base}")
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    base_vocab = tokenizer.get_vocab()
    base_size = len(tokenizer)
    print(f"  base vocabulary: {base_size} tokens")

    if args.inputs_list:
        found, n_circuits, errors = recompute_tokens(args.inputs_list, base_vocab)
        print(f"  recomputed from {n_circuits} circuits ({errors} unparsable lines skipped): "
              f"{len(found)} new tokens")
        if found != set(tokens):
            raise SystemExit(f"Token set differs from quantum_tokens.json: "
                             f"missing {sorted(set(tokens) - found)[:10]}, "
                             f"extra {sorted(found - set(tokens))[:10]}")
        print("  matches quantum_tokens.json")

    already = [t for t in tokens if t in base_vocab]
    if already:
        raise SystemExit(f"Base tokenizer already contains quantum tokens (is it the unmodified "
                         f"base model?): {already[:5]}")
    tokenizer.add_tokens(tokens)
    added = [tokenizer.convert_ids_to_tokens(i) for i in range(base_size, len(tokenizer))]
    assert added == tokens, "added tokens did not receive the expected ids"
    with open(TEMPLATE_FILE) as f:
        tokenizer.chat_template = f.read()

    os.makedirs(out, exist_ok=True)
    tokenizer.save_pretrained(out)   # bos/eos/pad live in tokenizer_config.json

    # Configs (with the LLaMA 3.1 end-of-sequence fix).
    for name in ("config.json", "generation_config.json"):
        with open(os.path.join(base, name)) as f:
            cfg = json.load(f)
        eos = cfg.get("eos_token_id")
        if isinstance(eos, list) and 128008 in eos:
            cfg["eos_token_id"] = 128009 if name == "config.json" else [128001, 128009]
            print(f"  {name}: eos_token_id {eos} -> {cfg['eos_token_id']}")
        with open(os.path.join(out, name), "w") as f:
            json.dump(cfg, f, indent=2)

    # Weights.
    weights = sorted(glob.glob(os.path.join(base, "*.safetensors")))
    weights += [p for p in glob.glob(os.path.join(base, "model.safetensors.index.json"))]
    if not weights:
        raise SystemExit(f"No .safetensors weights found in {base}")
    for src in weights:
        dst = os.path.join(out, os.path.basename(src))
        if os.path.lexists(dst):
            os.remove(dst)
        if args.copy_weights:
            shutil.copy2(src, dst)
        else:
            os.symlink(src, dst)
    print(f"  {'copied' if args.copy_weights else 'linked'} {len(weights)} weight files")

    # Verify.
    check = AutoTokenizer.from_pretrained(out)
    prompt = check.apply_chat_template([{"role": "user", "content": "test"}],
                                       tokenize=False, add_generation_prompt=True)
    assert len(check) == base_size + len(tokens), len(check)
    assert prompt == EXPECTED_PROMPT, f"unexpected chat template rendering: {prompt!r}"
    print(f"Prepared model written to {out}")
    print(f"  vocabulary: {len(check)} tokens (+{len(tokens)}), chat template: simple llama3")


if __name__ == "__main__":
    main()
