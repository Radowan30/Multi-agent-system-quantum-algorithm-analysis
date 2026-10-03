#!/bin/bash
# Phase 3 — prompt-engineered three-agent pipeline on four LLM variations.
#
# Steps:
#   1. Build the 51-circuit Phase 3 set (one circuit per (n, k) stratum).
#   2. Run the three-agent chain (LangChain orchestrator, concurrency 32) for:
#        untrained Llama-3-8B-Instruct-262k   ($WORK/models/Llama-3-8B-Instruct-262k)
#        untrained LLaMA 3.1 8B               ($WORK/models/Llama-3.1-8B-Instruct)
#        Phase 2 fine-tuned 262k              ($WORK/saves/.../merged/Gradient262k_alpha32)
#        Phase 2 fine-tuned LLaMA 3.1 8B      ($WORK/saves/.../merged/Llama31_alpha32)
#      run_eval.py starts and stops its own vLLM server for each variation,
#      so do NOT start a vLLM server yourself.
#   3. Aggregate the per-agent metrics and draw the plots.
#
# Prerequisites: Phase 2 finished (both merged models), the full 2-19 paper
# manifest generated, and the downloaded base models left unmodified.
#
# Usage:  bash "Reproduce Results/Phase 3/run_phase3.sh"
# Output: $WORK/results/phase-3/<variation>/

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../paths.sh"
PHASE3="$REPRO_DIR/Phase 3"
OUT="$RESULTS/phase-3"

require "$MANIFESTS/circuits_full_2_19_paper.json" \
    "generate it with evaluation_pipeline/generate_eval_circuits.py --mode full --n_min 2 --n_max 19 --circuit_source paper"

section "Building the Phase 3 circuit set"
"$PY_EVAL" "$PHASE3/build_phase3_circuits.py" --out_dir "$OUT"

# name | model directory | max_model_len (native context of the base)
VARIATIONS=(
    "untrained_gradient262k|$MODELS/Llama-3-8B-Instruct-262k|150000"
    "untrained_llama31|$MODELS/Llama-3.1-8B-Instruct|131072"
    "finetuned_gradient262k_alpha32|$SAVES/Llama-3-8B-Instruct-262k/merged/Gradient262k_alpha32|150000"
    "finetuned_llama31_alpha32|$SAVES/Llama-3.1-8B-Instruct/merged/Llama31_alpha32|131072"
)

for entry in "${VARIATIONS[@]}"; do
    IFS='|' read -r name model max_len <<< "$entry"
    require "$model/config.json" "download the base model (README section 6) or run Phase 2 first"
    section "Phase 3 chain: $name"
    "$PY_EVAL" "$PHASE3/run_eval.py" \
        --model_path "$model" \
        --circuits_jsonl "$OUT/circuits.jsonl" \
        --output_dir "$OUT/$name" \
        --orchestrator langchain \
        --max_model_len "$max_len" \
        --gpu_memory_utilization 0.90 \
        --max_tokens 8192 \
        --max_concurrency 32 \
        --label "$name"
done

section "Aggregating metrics and drawing plots"
"$PY_EVAL" "$PHASE3/make_phase3_plots.py" --results_root "$OUT"

section "Phase 3 done. Results in $OUT"
