#!/bin/bash
# Phase 4 — full paper evaluation of the Llama-3-8B-Instruct-262k multi-agent system.
#
# Steps:
#   1. Build the paper circuit set (4,024 circuits, n=2..19) and the RET set
#      (3 mixed-k circuits per n).
#   2. RET pass: concurrency 1, so each chain is timed on its own.
#   3. Bulk pass: concurrency 8, all 4,024 circuits.
#   4. Post-process into Phase 1/2-compatible JSONs (SA, CF, RET, per-agent
#      accuracy) and draw the plots.
# run_eval.py starts and stops its own vLLM server, so do NOT start one
# yourself. The bulk pass takes about a day on an RTX PRO 6000.
#
# Prerequisites: run_phase4_Llama-3-8B-Instruct-262k.sh finished; the 2-19 paper manifest
# exists (it is generated here if missing).
#
# Usage:  bash "Reproduce Results/Phase 4/run_paper_eval_Llama-3-8B-Instruct-262k.sh"
# Output: $WORK/results/phase-4/Llama-3-8B-Instruct-262k/

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../paths.sh"
HERE="$REPRO_DIR/Phase 4"
MERGED="$SAVES/Llama-3-8B-Instruct-262k/merged/Phase4_alpha32"
OUT="$RESULTS/phase-4/Llama-3-8B-Instruct-262k"
BULK="$OUT/_bulk"
RET="$OUT/_ret"
MAX_MODEL_LEN=150000      # Gradient's window is 262k; 150k fits n=19 plus output

require "$MERGED/config.json" "run run_phase4_Llama-3-8B-Instruct-262k.sh first"
mkdir -p "$BULK" "$RET"

if [ ! -f "$MANIFESTS/circuits_full_2_19_paper.json" ]; then
    section "Evaluation manifest (full, n=2..19)"
    (cd "$EVAL_DIR" && "$PY_EVAL" generate_eval_circuits.py --mode full --n_min 2 --n_max 19 --circuit_source paper)
fi

section "Building the paper circuit set and the RET set"
"$PY_EVAL" "$HERE/build_paper_circuits.py" --out_dir "$BULK"
"$PY_EVAL" "$HERE/build_ret_circuits.py" --out_dir "$RET"

section "RET pass (concurrency 1)"
"$PY_EVAL" "$HERE/run_eval.py" --model_path "$MERGED" \
    --circuits_jsonl "$RET/circuits.jsonl" --output_dir "$RET" --orchestrator langchain \
    --max_model_len $MAX_MODEL_LEN --gpu_memory_utilization 0.90 --max_tokens 8192 \
    --max_concurrency 1 --label phase4_gradient_paper_ret

section "Bulk pass (concurrency 8, 4,024 circuits)"
"$PY_EVAL" "$HERE/run_eval.py" --model_path "$MERGED" \
    --circuits_jsonl "$BULK/circuits.jsonl" --output_dir "$BULK" --orchestrator langchain \
    --max_model_len $MAX_MODEL_LEN --gpu_memory_utilization 0.90 --max_tokens 8192 \
    --max_concurrency 8 --label phase4_gradient_paper_bulk

section "Post-processing"
"$PY_EVAL" "$HERE/paper_eval_postprocess.py" --bulk_dir "$BULK" --ret_dir "$RET" \
    --out_dir "$OUT" --n_min 2 --n_max 19

section "Plots"
cd "$EVAL_DIR"
"$PY_EVAL" plot_results.py --phase 2 --mode full --circuit_source paper --cf_key both --results_dir "$OUT"
"$PY_EVAL" plot_oracle_accuracy.py \
    --input "$OUT/oracle_extraction_full_2_19_paper.json" "Agent 1 (Oracle Extraction)" \
    --output "$OUT/oracle_accuracy.png" \
    --title "Phase 4 Gradient — Agent 1 (Oracle Extraction) Accuracy"
"$PY_EVAL" plot_oracle_accuracy.py \
    --input "$OUT/marked_state_accuracy_full_2_19_paper.json" "Agent 2 (Marked State Identification)" \
    --output "$OUT/marked_state_accuracy.png" \
    --title "Phase 4 Gradient — Agent 2 Marked-State Identification Accuracy (strict conditional on A1)"
"$PY_EVAL" plot_oracle_accuracy.py \
    --input "$OUT/agent3_accuracy_full_2_19_paper.json" "Agent 3 (Probability Distribution; mean SA)" \
    --output "$OUT/agent3_accuracy.png" \
    --title "Phase 4 Gradient — Agent 3 Probability Distribution Accuracy (strict conditional on A2)"

section "Phase 4 evaluation (Llama-3-8B-Instruct-262k) done. Results in $OUT"
