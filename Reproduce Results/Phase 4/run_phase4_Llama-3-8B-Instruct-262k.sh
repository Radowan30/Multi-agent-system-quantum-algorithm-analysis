#!/bin/bash
# Phase 4 pipeline — alpha=32, single-session joint training of all three
# agent datasets, single merged model, with one approval gate after the
# smoke test before the full eval.
#
# Pre-approved flow (no gates):
#   1. Train (3 agent datasets concatenated, n=2..7, 10 epochs)
#   2. Merge LoRA → standalone model
#   3. Smoke-test: 3 circuits (EASY n=3, MID n=8, HARD n=15) via vanilla orchestrator
#
# GATE — waits for /tmp/phase4_alpha32_gate1.go before proceeding to full eval
#
# Post-gate (after approval):
#   4. Full eval — 51 circuits (1 per (n,k), n=2..19) via vanilla orchestrator
#
# Output:
#   saves/Llama-3-8B-Instruct-262k/lora/Phase4_alpha32/    LoRA adapter
#   saves/Llama-3-8B-Instruct-262k/merged/Phase4_alpha32/  merged model
#   evaluation/phase-4/smoke/                              smoke results
#   evaluation/phase-4/full/                               full eval results
#
# Gate mechanism — to approve the full eval after smoke completes:
#   touch /tmp/phase4_alpha32_gate1.go
#
# Run:
#   bash "Multi Agent System Phase 4/run_phase4.sh" \
#       > /tmp/phase4_alpha32.log 2>&1 &

set -euo pipefail

PROJ=/home/quantum-user/radowan/final_year_project
PHASE4_DIR="$PROJ/Multi Agent System Phase 4"
LF="$PROJ/LLaMA-Factory"

PY_EVAL="$PROJ/venv-eval/bin/python"
LF_CLI="$PROJ/venv-train/bin/llamafactory-cli"

TRAIN_YAML="$PHASE4_DIR/train_phase4_alpha32.yaml"
MERGE_YAML="$PHASE4_DIR/merge_phase4_alpha32.yaml"
MERGED="$PROJ/saves/Llama-3-8B-Instruct-262k/merged/Phase4_alpha32"

SMOKE_DIR="$PROJ/evaluation/phase-4/smoke"
FULL_DIR="$PROJ/evaluation/phase-4/full"

# Reuse Phase 3's 51-circuit JSONL so Phase 4 full-eval results are
# directly comparable to the Phase 3 four-model preliminary results.
FULL_CIRCUITS_JSONL="$PROJ/evaluation/phase-3/preliminary/circuits.jsonl"

GATE1_DONE=/tmp/phase4_alpha32_gate1.done
GATE1_GO=/tmp/phase4_alpha32_gate1.go

rm -f "$GATE1_DONE" "$GATE1_GO"

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

section "Phase 4 alpha32 — training"
cd "$LF" && "$LF_CLI" train "$TRAIN_YAML"
cd "$PROJ"

# ═══════════════════════════════════════════════════════════════════════════
# 2. Merge
# ═══════════════════════════════════════════════════════════════════════════

section "Phase 4 alpha32 — merging adapter"
cd "$LF" && "$LF_CLI" export "$MERGE_YAML"
cd "$PROJ"

# ═══════════════════════════════════════════════════════════════════════════
# 3. Smoke test — 3 circuits (EASY n=3, MID n=8, HARD n=15)
# ═══════════════════════════════════════════════════════════════════════════

section "Phase 4 alpha32 — building smoke circuits"
mkdir -p "$SMOKE_DIR"
"$PY_EVAL" "$PHASE4_DIR/build_smoke_circuits.py" --out_dir "$SMOKE_DIR"

section "Phase 4 alpha32 — smoke eval (3 circuits, vanilla orchestrator)"
"$PY_EVAL" "$PHASE4_DIR/run_eval.py" \
    --model_path "$MERGED" \
    --circuits_jsonl "$SMOKE_DIR/circuits.jsonl" \
    --output_dir "$SMOKE_DIR" \
    --orchestrator vanilla \
    --max_model_len 150000 \
    --gpu_memory_utilization 0.90 \
    --max_tokens 8192 \
    --max_concurrency 3 \
    --label "phase4_alpha32_smoke"

# ═══════════════════════════════════════════════════════════════════════════
# Gate — wait for approval before running full eval
# ═══════════════════════════════════════════════════════════════════════════

section "GATE — train + merge + smoke complete; awaiting approval for full eval"
echo "Smoke results at: $SMOKE_DIR"
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo
echo "Remaining steps if approved: full eval on 51 circuits (Phase 3 preliminary set)."
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
echo "Approved at: $(date '+%Y-%m-%d %H:%M:%S'). Proceeding with full eval."

# ═══════════════════════════════════════════════════════════════════════════
# 4. Full eval — 51 circuits via vanilla orchestrator
# ═══════════════════════════════════════════════════════════════════════════

section "Phase 4 alpha32 — full eval (51 circuits, vanilla orchestrator)"
mkdir -p "$FULL_DIR"
"$PY_EVAL" "$PHASE4_DIR/run_eval.py" \
    --model_path "$MERGED" \
    --circuits_jsonl "$FULL_CIRCUITS_JSONL" \
    --output_dir "$FULL_DIR" \
    --orchestrator vanilla \
    --max_model_len 150000 \
    --gpu_memory_utilization 0.90 \
    --max_tokens 8192 \
    --max_concurrency 8 \
    --label "phase4_alpha32_full"

section "Phase 4 alpha32 — DONE"
echo "Train + Merge + Smoke + Full eval complete."
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
