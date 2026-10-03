#!/bin/bash
# Phase 2 — fine-tune Llama-3-8B-Instruct-262k on the GroverGPT+ dataset with the Phase 1
# recipe, then evaluate full-circuit inputs over n=2..19 and oracle-only
# inputs over n=2..20.
#
# Steps: train + merge -> chat-template check -> evaluation manifests ->
#        full-circuit eval -> oracle-only eval -> CoT trace inspection ->
#        oracle extraction accuracy -> plots.
#
# Prerequisites: as for Phase 1, plus $WORK/models/Llama-3-8B-Instruct-262k-quantum made by
# tokenizer/prepare_quantum_model.py (README section 6).
#
# Usage:  bash "Reproduce Results/Phase 2/Llama-3-8B-Instruct-262k/run_phase2.sh"
# Output: $WORK/saves/Llama-3-8B-Instruct-262k/{lora,merged}/Gradient262k_alpha32
#         $WORK/results/phase-2/Llama-3-8B-Instruct-262k/

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../paths.sh"
HERE="$REPRO_DIR/Phase 2/Llama-3-8B-Instruct-262k"
MERGED="$SAVES/Llama-3-8B-Instruct-262k/merged/Gradient262k_alpha32"
OUT="$RESULTS/phase-2/Llama-3-8B-Instruct-262k"

require "$MODELS/Llama-3-8B-Instruct-262k-quantum/tokenizer.json" \
    "run tokenizer/prepare_quantum_model.py for Llama-3-8B-Instruct-262k (README section 6)"

train_and_merge "$HERE/train.yaml" "$HERE/merge.yaml"
check_chat_template "$MERGED"

section "Evaluation manifests"
cd "$EVAL_DIR"
for spec in "full 2 19" "oracle 2 20"; do
    set -- $spec
    [ -f "$MANIFESTS/circuits_$1_$2_$3_paper.json" ] || \
        "$PY_EVAL" generate_eval_circuits.py --mode "$1" --n_min "$2" --n_max "$3" --circuit_source paper
done

section "Full-circuit evaluation (n=2..19)"
"$PY_EVAL" run_eval_phase2.py --mode full --circuit_source paper \
    --model_path "$MERGED" --results_dir "$OUT" --gpu_memory_utilization 0.90
section "Oracle-only evaluation (n=2..20)"
"$PY_EVAL" run_eval_phase2.py --mode oracle --circuit_source paper \
    --model_path "$MERGED" --results_dir "$OUT" --gpu_memory_utilization 0.90 --max_model_len 8192

section "CoT trace inspection"
"$PY_EVAL" inspect_cot_traces.py --model_path "$MERGED" --results_dir "$OUT" \
    --full_n_range 2-19 --max_model_len 120000 --gpu_memory_utilization 0.90

section "Oracle extraction accuracy"
"$PY_EVAL" oracle_extraction_accuracy.py --outputs_jsonl "$OUT/outputs_full_2_19_paper.jsonl" \
    --manifest "$MANIFESTS/circuits_full_2_19_paper.json" --results_dir "$OUT"
"$PY_EVAL" oracle_extraction_accuracy.py --outputs_jsonl "$OUT/outputs_oracle_2_20_paper.jsonl" \
    --manifest "$MANIFESTS/circuits_oracle_2_20_paper.json" --results_dir "$OUT"

section "Plots"
for mode in full oracle; do
    "$PY_EVAL" plot_results.py --phase 2 --mode "$mode" --circuit_source paper --cf_key both --results_dir "$OUT"
done
# RET normalised to T(3) as well as T(2)
"$PY_EVAL" plot_ret_normalized.py --results_dir "$OUT" --circuit_source paper \
    --normalize_n 3 --label "Llama-3-8B-Instruct-262k (Phase 2, alpha=32)"
"$PY_EVAL" plot_oracle_accuracy.py --dual \
    --full   "$OUT/oracle_extraction_full_2_19_paper.json"   "Fine-tuned Gradient-262k (alpha=32)" \
    --oracle "$OUT/oracle_extraction_oracle_2_20_paper.json" "Fine-tuned Gradient-262k (alpha=32)" \
    --train_full_range 2,7 --train_oracle_range 2,10 \
    --output "$OUT/oracle_accuracy_dual.png" --title "Oracle extraction accuracy"

section "Phase 2 (Llama-3-8B-Instruct-262k) done. Results in $OUT"
