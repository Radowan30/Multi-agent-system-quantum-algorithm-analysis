#!/bin/bash
# Phase 4 (LLaMA 3.1 variant) pipeline — train, merge, stratified eval.
# Mirrors run_phase4.sh but on the LLaMA 3.1 base. No gate (the Gradient
# variant has already proven the recipe works; this is a robustness check).
#
# Output:
#   saves/Llama-3.1-8B-Instruct/lora/Phase4_alpha32/    LoRA adapter
#   saves/Llama-3.1-8B-Instruct/merged/Phase4_alpha32/  merged model
#   evaluation/phase-4/stratified_llama31/              stratified eval (153 circuits)
#
# Run:
#   bash "Multi Agent System Phase 4/run_phase4_llama31.sh" \
#       > /tmp/phase4_llama31.log 2>&1 &

set -euo pipefail

PROJ=/home/quantum-user/radowan/final_year_project
PHASE4_DIR="$PROJ/Multi Agent System Phase 4"
LF="$PROJ/LLaMA-Factory"

PY_EVAL="$PROJ/venv-eval/bin/python"
LF_CLI="$PROJ/venv-train/bin/llamafactory-cli"

TRAIN_YAML="$PHASE4_DIR/train_phase4_llama31_alpha32.yaml"
MERGE_YAML="$PHASE4_DIR/merge_phase4_llama31_alpha32.yaml"
MERGED="$PROJ/saves/Llama-3.1-8B-Instruct/merged/Phase4_alpha32"

STRAT_DIR="$PROJ/evaluation/phase-4/stratified_llama31"

# Reuse the same seed-42 stratified 153-circuit set as the Gradient run
# for direct comparability. The circuits.jsonl and qasm/ are shared from
# the Gradient stratified dir.
SOURCE_STRAT="$PROJ/evaluation/phase-4/stratified"

section() {
    echo
    echo "================================================================"
    echo "  $*"
    echo "================================================================"
}

cd "$PROJ"

# ═══════════════════════════════════════════════════════════════════════════
# 1. Train
# ═══════════════════════════════════════════════════════════════════════════

section "Phase 4 LLaMA 3.1 alpha32 — training"
cd "$LF" && "$LF_CLI" train "$TRAIN_YAML"
cd "$PROJ"

# ═══════════════════════════════════════════════════════════════════════════
# 2. Merge
# ═══════════════════════════════════════════════════════════════════════════

section "Phase 4 LLaMA 3.1 alpha32 — merging adapter"
cd "$LF" && "$LF_CLI" export "$MERGE_YAML"
cd "$PROJ"

# ═══════════════════════════════════════════════════════════════════════════
# 3. Stratified eval (153 circuits, identical seed-42 set as Gradient run)
# ═══════════════════════════════════════════════════════════════════════════

section "Phase 4 LLaMA 3.1 alpha32 — preparing stratified circuits dir"
mkdir -p "$STRAT_DIR"
# Reuse the existing JSONL + qasm dir; copy so the output dir is self-contained
# but doesn't re-roll the seed.
cp "$SOURCE_STRAT/circuits.jsonl" "$STRAT_DIR/circuits.jsonl"
cp -r "$SOURCE_STRAT/qasm" "$STRAT_DIR/qasm"

section "Phase 4 LLaMA 3.1 alpha32 — stratified eval (153 circuits, langchain orchestrator)"
"$PY_EVAL" "$PHASE4_DIR/run_eval.py" \
    --model_path "$MERGED" \
    --circuits_jsonl "$STRAT_DIR/circuits.jsonl" \
    --output_dir "$STRAT_DIR" \
    --orchestrator langchain \
    --max_model_len 131072 \
    --gpu_memory_utilization 0.90 \
    --max_tokens 8192 \
    --max_concurrency 8 \
    --label phase4_llama31_alpha32_stratified

section "Phase 4 LLaMA 3.1 alpha32 — DONE"
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
