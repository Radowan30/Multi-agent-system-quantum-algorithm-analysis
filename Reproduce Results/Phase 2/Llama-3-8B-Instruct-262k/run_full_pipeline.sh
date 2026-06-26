#!/bin/bash
# Phase 2 (Gradient variant) pipeline — alpha=32, with one approval gate after
# both eval modes complete, before the post-eval analysis (CoT inspection,
# oracle accuracy, plots).
#
# Pre-approved flow (no gates):
#   1. Train
#   2. Merge LoRA → standalone model
#   3. Full-circuit eval (n=2..19, max_model_len=120K, 0.90 GPU util)
#   4. Oracle-only eval (n=2..20, max_model_len=8K, 0.90 GPU util)
#
# GATE — waits for /tmp/phase_gradient_alpha32_gate1.go before proceeding
#
# Post-gate (after approval):
#   5. CoT trace inspection
#   6. Oracle extraction accuracy
#   7. All plots (including n=3-normalized RET)
#
# Output:
#   saves/Llama-3-8B-Instruct-262k/lora/Gradient262k_alpha32/
#   saves/Llama-3-8B-Instruct-262k/merged/Gradient262k_alpha32/
#   evaluation/phase-2/results_gradient262k_alpha32/...
#
# Gate mechanism — to approve the post-eval analysis:
#   touch /tmp/phase_gradient_alpha32_gate1.go
#
# Run:
#   bash "Fine-tuned Gradient 262k 8B/run_phase_gradient_alpha32.sh" \
#       > /tmp/phase_gradient_alpha32.log 2>&1 &

set -euo pipefail

PROJ=/home/quantum-user/radowan/final_year_project
GRAD_DIR="$PROJ/Fine-tuned Gradient 262k 8B"
LF="$PROJ/LLaMA-Factory"

PY_EVAL="$PROJ/venv-eval/bin/python"
LF_CLI="$PROJ/venv-train/bin/llamafactory-cli"

TRAIN_YAML="train_gradient_alpha32.yaml"
MERGE_YAML="merge_gradient_alpha32.yaml"
MERGED="$PROJ/saves/Llama-3-8B-Instruct-262k/merged/Gradient262k_alpha32"
RESULTS_DIR="$PROJ/evaluation/phase-2/results_gradient262k_alpha32"

GATE1_DONE=/tmp/phase_gradient_alpha32_gate1.done
GATE1_GO=/tmp/phase_gradient_alpha32_gate1.go

rm -f "$GATE1_DONE" "$GATE1_GO"

section() {
    echo
    echo "================================================================"
    echo "  $*"
    echo "================================================================"
}

cd "$PROJ"

# ═══════════════════════════════════════════════════════════════════════════
# Pre-approved stages — train, merge, both eval modes
# ═══════════════════════════════════════════════════════════════════════════

section "Gradient alpha32 — training"
cd "$LF" && "$LF_CLI" train "$GRAD_DIR/$TRAIN_YAML"
cd "$PROJ"

section "Gradient alpha32 — merging adapter"
cd "$LF" && "$LF_CLI" export "$GRAD_DIR/$MERGE_YAML"
cd "$PROJ"

# Full-circuit eval — needs 120K window for n=19 (~110K tokens).
section "Gradient alpha32 — full-circuit eval (n=2..19)"
"$PY_EVAL" "$PROJ/evaluation/phase-2/run_eval.py" \
    --mode full --circuit_source paper \
    --model_path "$MERGED" --results_dir "$RESULTS_DIR" \
    --gpu_memory_utilization 0.90

# Oracle eval — 8K window is plenty.
section "Gradient alpha32 — oracle-only eval (n=2..20)"
"$PY_EVAL" "$PROJ/evaluation/phase-2/run_eval.py" \
    --mode oracle --circuit_source paper \
    --model_path "$MERGED" --results_dir "$RESULTS_DIR" \
    --gpu_memory_utilization 0.90 \
    --max_model_len 8192

# ═══════════════════════════════════════════════════════════════════════════
# Gate — wait for approval before running post-eval analysis
# ═══════════════════════════════════════════════════════════════════════════

section "GATE — train + merge + both evals complete; awaiting approval for post-eval analysis"
echo "Train + Merge + Full eval (n=2..19) + Oracle eval (n=2..20) complete."
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo
echo "Remaining steps if approved: CoT trace inspection, oracle extraction accuracy, all plots."
echo
echo "To approve and continue, run:"
echo "    touch $GATE1_GO"
echo
echo "Polling every 30 seconds for the approval file..."
touch "$GATE1_DONE"
while [ ! -f "$GATE1_GO" ]; do
    sleep 30
done
rm -f "$GATE1_GO"
echo "Approved at: $(date '+%Y-%m-%d %H:%M:%S'). Proceeding with post-eval analysis."

# ═══════════════════════════════════════════════════════════════════════════
# Post-approval — CoT trace inspection, oracle extraction accuracy, plots
# ═══════════════════════════════════════════════════════════════════════════

section "Gradient alpha32 — CoT trace inspection"
"$PY_EVAL" "$PROJ/evaluation/inspect_cot_traces.py" \
    --model_path "$MERGED" --results_dir "$RESULTS_DIR" \
    --full_n_range 2-19 \
    --max_model_len 120000 \
    --gpu_memory_utilization 0.90

section "Gradient alpha32 — oracle extraction accuracy"
"$PY_EVAL" "$PROJ/evaluation/oracle_extraction_accuracy.py" \
    --outputs_jsonl "$RESULTS_DIR/outputs_full_2_19_paper.jsonl" \
    --manifest      "$PROJ/evaluation/circuit_manifests/circuits_full_2_19_paper.json" \
    --results_dir   "$RESULTS_DIR"
"$PY_EVAL" "$PROJ/evaluation/oracle_extraction_accuracy.py" \
    --outputs_jsonl "$RESULTS_DIR/outputs_oracle_2_20_paper.jsonl" \
    --manifest      "$PROJ/evaluation/circuit_manifests/circuits_oracle_2_20_paper.json" \
    --results_dir   "$RESULTS_DIR"

section "Gradient alpha32 — SA/CF/CR/SRR/RET plots"
for MODE in full oracle; do
    "$PY_EVAL" "$PROJ/evaluation/plot_results.py" \
        --phase 2 --mode "$MODE" --circuit_source paper \
        --cf_key both --results_dir "$RESULTS_DIR" || true
done

section "Gradient alpha32 — n=3-normalized RET plot (in case n=2 fails)"
"$PY_EVAL" "$PROJ/evaluation/phase-2/plot_ret_normalized.py" \
    --results_dir "$RESULTS_DIR" \
    --circuit_source paper \
    --normalize_n 3 \
    --label "Llama-3-8B-Instruct-262k (Phase 2, alpha=32)" || true

section "Gradient alpha32 — oracle accuracy dual plot"
"$PY_EVAL" "$PROJ/evaluation/plot_oracle_accuracy.py" --dual \
    --full   "$RESULTS_DIR/oracle_extraction_full_2_19_paper.json"   "Fine-tuned Gradient-262k (alpha=32)" \
    --oracle "$RESULTS_DIR/oracle_extraction_oracle_2_20_paper.json" "Fine-tuned Gradient-262k (alpha=32)" \
    --train_full_range   2,7 \
    --train_oracle_range 2,10 \
    --output "$RESULTS_DIR/oracle_accuracy_dual.png" \
    --title  "Oracle extraction accuracy"

section "Gradient alpha32 — DONE"
