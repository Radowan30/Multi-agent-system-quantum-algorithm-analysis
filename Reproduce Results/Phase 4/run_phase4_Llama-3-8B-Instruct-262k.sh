#!/bin/bash
# Phase 4 — fine-tune Llama-3-8B-Instruct-262k on the three agent-specific datasets (jointly, one
# model per base) and merge the adapter.
#
# Prerequisites: $WORK/models/Llama-3-8B-Instruct-262k-quantum (tokenizer/prepare_quantum_model.py)
# and Grover_Agent{1,2,3}_2_7_MMS.json copied into $WORK/LLaMA-Factory/data/ and
# registered (README section 5).
#
# Usage:  bash "Reproduce Results/Phase 4/run_phase4_Llama-3-8B-Instruct-262k.sh"
# Output: $WORK/saves/Llama-3-8B-Instruct-262k/{lora,merged}/Phase4_alpha32
# Then evaluate with run_paper_eval_Llama-3-8B-Instruct-262k.sh.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../paths.sh"
HERE="$REPRO_DIR/Phase 4"

require "$MODELS/Llama-3-8B-Instruct-262k-quantum/tokenizer.json" \
    "run tokenizer/prepare_quantum_model.py for Llama-3-8B-Instruct-262k"
for i in 1 2 3; do
    require "$LLAMA_FACTORY/data/Grover_Agent${i}_2_7_MMS.json" "copy Phase 4/dataset/*.json into LLaMA-Factory/data (README section 5)"
done

train_and_merge "$HERE/train_Llama-3-8B-Instruct-262k.yaml" "$HERE/merge_Llama-3-8B-Instruct-262k.yaml"
check_chat_template "$SAVES/Llama-3-8B-Instruct-262k/merged/Phase4_alpha32"
section "Phase 4 training (Llama-3-8B-Instruct-262k) done"
