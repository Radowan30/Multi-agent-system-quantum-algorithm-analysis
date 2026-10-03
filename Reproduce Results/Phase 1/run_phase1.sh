#!/bin/bash
# Phase 1 — recreate GroverGPT+ (LLaMA 3 8B Instruct + quantum-native tokenizer).
#
# Steps: train + merge -> chat-template check -> evaluation manifests ->
#        full-circuit eval (n=2..9) -> oracle-only eval (n=2..20) ->
#        CoT trace inspection -> oracle extraction accuracy -> plots.
#
# Prerequisites (README sections 2-7): venvs, LLaMA-Factory with the Grover
# datasets registered, $WORK/data_MMS, and $WORK/models/Meta-Llama-3-8B-Instruct-quantum
# made by tokenizer/prepare_quantum_model.py.
#
# Usage:  bash "Reproduce Results/Phase 1/run_phase1.sh"
# Output: $WORK/saves/Meta-Llama-3-8B-Instruct/{lora,merged}/GroverGPT+_alpha32
#         $WORK/results/phase-1/

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../paths.sh"
PHASE1="$REPRO_DIR/Phase 1"
MERGED="$SAVES/Meta-Llama-3-8B-Instruct/merged/GroverGPT+_alpha32"
OUT="$RESULTS/phase-1"

require "$MODELS/Meta-Llama-3-8B-Instruct-quantum/tokenizer.json" \
    "run tokenizer/prepare_quantum_model.py for Meta-Llama-3-8B-Instruct (README section 6)"
require "$LLAMA_FACTORY/data/Grover_FullCircuit_2_7_MMS.json" "copy the GroverGPT+ datasets (README section 5)"

train_and_merge "$PHASE1/train.yaml" "$PHASE1/merge.yaml"
check_chat_template "$MERGED"

section "Evaluation manifests"
cd "$EVAL_DIR"
for spec in "full 2 9" "oracle 2 20"; do
    set -- $spec
    [ -f "$MANIFESTS/circuits_$1_$2_$3_paper.json" ] || \
        "$PY_EVAL" generate_eval_circuits.py --mode "$1" --n_min "$2" --n_max "$3" --circuit_source paper
done

section "Full-circuit evaluation (n=2..9)"
"$PY_EVAL" run_eval_phase1.py --mode full --circuit_source paper --model_path "$MERGED" --results_dir "$OUT"
section "Oracle-only evaluation (n=2..20)"
"$PY_EVAL" run_eval_phase1.py --mode oracle --circuit_source paper --model_path "$MERGED" --results_dir "$OUT"

section "CoT trace inspection"
"$PY_EVAL" inspect_cot_traces.py --model_path "$MERGED" --results_dir "$OUT"

section "Oracle extraction accuracy"
"$PY_EVAL" oracle_extraction_accuracy.py --outputs_jsonl "$OUT/outputs_full_2_9_paper.jsonl" \
    --manifest "$MANIFESTS/circuits_full_2_9_paper.json" --results_dir "$OUT"
"$PY_EVAL" oracle_extraction_accuracy.py --outputs_jsonl "$OUT/outputs_oracle_2_20_paper.jsonl" \
    --manifest "$MANIFESTS/circuits_oracle_2_20_paper.json" --results_dir "$OUT"

section "Plots"
for mode in full oracle; do
    "$PY_EVAL" plot_results.py --phase 1 --mode "$mode" --circuit_source paper --cf_key both --results_dir "$OUT"
done
"$PY_EVAL" plot_oracle_accuracy.py --dual \
    --full   "$OUT/oracle_extraction_full_2_9_paper.json"   "Fine-tuned Llama 3 8B" \
    --oracle "$OUT/oracle_extraction_oracle_2_20_paper.json" "Fine-tuned Llama 3 8B" \
    --train_full_range 2,7 --train_oracle_range 2,10 \
    --output "$OUT/oracle_accuracy_dual.png" --title "Oracle extraction accuracy"

section "Phase 1 done. Results in $OUT"
