#!/bin/bash
# Phase 4 LLaMA 3.1 8B full paper eval — mirrors run_paper_eval_gradient.sh,
# only the merged-model path and the output directory name differ.
#
# Output goes into evaluation/phase-4/results_metaAI_llama3_1_8b/
#
# Pipeline:
#   1. Build paper circuits JSONL (4024 circuits, shared format)
#   2. Build RET circuits JSONL (54 circuits = 3 mixed-k per n=2..19)
#   3. RET pass  — concurrency=1, single-instance timing (Phase 1/2 methodology)
#   4. Bulk pass — concurrency=8, full ~4024 circuits, langchain orchestrator
#   5. Postprocess — Phase 1/2-compatible files + cot_traces.md
#   6. Plots — SA/CF (raw + renorm), RET, oracle/marked-state/A3 accuracy
#
# max_model_len 131072 matches LLaMA 3.1's 128K native context (the Gradient
# variant uses 150000 because Gradient is 262K-native). Both share the same
# extended quantum-native tokenizer.
#
# Run:
#   bash "Multi Agent System Phase 4/run_paper_eval_llama31.sh" \
#       > /tmp/phase4_paper_eval_llama31.log 2>&1 &

set -euo pipefail

PROJ=/home/quantum-user/radowan/final_year_project
PHASE4_DIR="$PROJ/Multi Agent System Phase 4"
PY_EVAL="$PROJ/venv-eval/bin/python"

MERGED="$PROJ/saves/Llama-3.1-8B-Instruct/merged/Phase4_alpha32"
RESULTS_DIR="$PROJ/evaluation/phase-4/results_metaAI_llama3_1_8b"

BULK_DIR="$RESULTS_DIR/_bulk"
RET_DIR="$RESULTS_DIR/_ret"

mkdir -p "$RESULTS_DIR" "$BULK_DIR" "$RET_DIR"

section() {
    echo
    echo "================================================================"
    echo "  $*"
    echo "================================================================"
}

cd "$PROJ"

# ═══════════════════════════════════════════════════════════════════════════
# 1. Build circuits JSONLs
# ═══════════════════════════════════════════════════════════════════════════

section "Phase 4 LLaMA 3.1 paper eval — building paper circuits (~4024)"
"$PY_EVAL" "$PHASE4_DIR/build_paper_circuits.py" --out_dir "$BULK_DIR"

section "Phase 4 LLaMA 3.1 paper eval — building RET circuits (3 mixed-k per n)"
"$PY_EVAL" "$PHASE4_DIR/build_ret_circuits.py" --out_dir "$RET_DIR"

# ═══════════════════════════════════════════════════════════════════════════
# 2. RET pass — concurrency=1, single-instance timing
# ═══════════════════════════════════════════════════════════════════════════

section "Phase 4 LLaMA 3.1 paper eval — RET pass (concurrency=1, 54 circuits)"
"$PY_EVAL" "$PHASE4_DIR/run_eval.py" \
    --model_path "$MERGED" \
    --circuits_jsonl "$RET_DIR/circuits.jsonl" \
    --output_dir "$RET_DIR" \
    --orchestrator langchain \
    --max_model_len 131072 \
    --gpu_memory_utilization 0.90 \
    --max_tokens 8192 \
    --max_concurrency 1 \
    --label phase4_llama31_paper_ret

# ═══════════════════════════════════════════════════════════════════════════
# 3. Bulk pass — concurrency=8, full ~4024 circuits
# ═══════════════════════════════════════════════════════════════════════════

section "Phase 4 LLaMA 3.1 paper eval — bulk pass (concurrency=8, ~4024 circuits)"
"$PY_EVAL" "$PHASE4_DIR/run_eval.py" \
    --model_path "$MERGED" \
    --circuits_jsonl "$BULK_DIR/circuits.jsonl" \
    --output_dir "$BULK_DIR" \
    --orchestrator langchain \
    --max_model_len 131072 \
    --gpu_memory_utilization 0.90 \
    --max_tokens 8192 \
    --max_concurrency 8 \
    --label phase4_llama31_paper_bulk

# ═══════════════════════════════════════════════════════════════════════════
# 4. Postprocess
# ═══════════════════════════════════════════════════════════════════════════

section "Phase 4 LLaMA 3.1 paper eval — postprocessing"
"$PY_EVAL" "$PHASE4_DIR/paper_eval_postprocess.py" \
    --bulk_dir "$BULK_DIR" \
    --ret_dir "$RET_DIR" \
    --out_dir "$RESULTS_DIR" \
    --n_min 2 --n_max 19

# ═══════════════════════════════════════════════════════════════════════════
# 5. Plots
# ═══════════════════════════════════════════════════════════════════════════

section "Phase 4 LLaMA 3.1 paper eval — SA/CF + RET plots (raw + renorm CF)"
"$PY_EVAL" "$PROJ/evaluation/plot_results.py" \
    --phase 2 --mode full --circuit_source paper \
    --cf_key both --results_dir "$RESULTS_DIR"

section "Phase 4 LLaMA 3.1 paper eval — oracle accuracy plot (A1)"
"$PY_EVAL" "$PROJ/evaluation/plot_oracle_accuracy.py" \
    --input "$RESULTS_DIR/oracle_extraction_full_2_19_paper.json" "Agent 1 (Oracle Extraction)" \
    --output "$RESULTS_DIR/oracle_accuracy.png" \
    --title "Phase 4 LLaMA 3.1 — Agent 1 (Oracle Extraction) Accuracy"

section "Phase 4 LLaMA 3.1 paper eval — marked-state accuracy plot (A2)"
"$PY_EVAL" "$PROJ/evaluation/plot_oracle_accuracy.py" \
    --input "$RESULTS_DIR/marked_state_accuracy_full_2_19_paper.json" "Agent 2 (Marked State Identification)" \
    --output "$RESULTS_DIR/marked_state_accuracy.png" \
    --title "Phase 4 LLaMA 3.1 — Agent 2 Marked-State Identification Accuracy (strict conditional on A1)"

section "Phase 4 LLaMA 3.1 paper eval — Agent 3 accuracy plot"
"$PY_EVAL" "$PROJ/evaluation/plot_oracle_accuracy.py" \
    --input "$RESULTS_DIR/agent3_accuracy_full_2_19_paper.json" "Agent 3 (Probability Distribution; mean SA)" \
    --output "$RESULTS_DIR/agent3_accuracy.png" \
    --title "Phase 4 LLaMA 3.1 — Agent 3 Probability Distribution Accuracy (strict conditional on A2)"

section "Phase 4 LLaMA 3.1 paper eval — DONE"
echo "Results in: $RESULTS_DIR"
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
