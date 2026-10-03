#!/bin/bash
# Phase 4 ablation — a single monolithic model trained on the same data as the
# multi-agent system (LLaMA 3.1 8B base).
#
# The training target of each example is the three agent outputs joined end to
# end (identical text and token count), with the bare QASM circuit as input.
# Everything else matches the Phase 4 multi-agent pipeline: Table II settings
# (cutoff_len raised to 5000 so no target is truncated), the same 4,024 paper
# circuits and RET circuits, and the same metric and post-processing code.
#
# Stages (run one or both):
#   train  build the dataset (if missing) -> LoRA fine-tune -> merge -> template check
#   eval   paper + RET circuit sets -> RET pass (concurrency 1) -> sanity check ->
#          bulk pass (concurrency 8) -> post-processing -> plots
# run_monolithic_eval.py starts and stops its own vLLM server.
#
# Usage:  bash "Reproduce Results/Phase 4/Monolithic Model/run_LLaMA-3.1-8B.sh" train|eval|all
# Output: $WORK/saves/Llama-3.1-8B-Instruct/{lora,merged}/Phase4_monolithic_alpha32
#         $WORK/results/phase-4/monolithic_LLaMA-3.1-8B/

set -euo pipefail
STAGE="${1:-all}"
source "$(dirname "${BASH_SOURCE[0]}")/../../paths.sh"
HERE="$REPRO_DIR/Phase 4/Monolithic Model"
PHASE4="$REPRO_DIR/Phase 4"
MODEL=Llama-3.1-8B-Instruct
MERGED="$SAVES/$MODEL/merged/Phase4_monolithic_alpha32"
OUT="$RESULTS/phase-4/monolithic_LLaMA-3.1-8B"
MAX_MODEL_LEN=131072
LABEL=monolithic_llama31
TAG="Monolithic LLaMA 3.1"
SUFFIX=LLaMA-3.1-8B

if [ "$STAGE" = "train" ] || [ "$STAGE" = "all" ]; then
    require "$MODELS/$MODEL-quantum/tokenizer.json" "run tokenizer/prepare_quantum_model.py for $MODEL"
    if [ ! -f "$LLAMA_FACTORY/data/Grover_Monolithic_2_7_MMS.json" ]; then
        section "Building the monolithic dataset"
        "$PY_EVAL" "$HERE/generate_monolithic_dataset.py" \
            --tokenizer "$MODELS/$MODEL-quantum" \
            --out "$LLAMA_FACTORY/data/Grover_Monolithic_2_7_MMS.json"
    fi
    train_and_merge "$HERE/train_$SUFFIX.yaml" "$HERE/merge_$SUFFIX.yaml"
    check_chat_template "$MERGED"
fi

if [ "$STAGE" = "eval" ] || [ "$STAGE" = "all" ]; then
    require "$MERGED/config.json" "run this script with the train stage first"
    BULK="$OUT/_bulk"; RET="$OUT/_ret"
    mkdir -p "$BULK" "$RET"
    if [ ! -f "$MANIFESTS/circuits_full_2_19_paper.json" ]; then
        section "Evaluation manifest (full, n=2..19)"
        (cd "$EVAL_DIR" && "$PY_EVAL" generate_eval_circuits.py --mode full --n_min 2 --n_max 19 --circuit_source paper)
    fi

    section "$TAG — paper circuits (4,024) and RET circuits (54)"
    "$PY_EVAL" "$PHASE4/build_paper_circuits.py" --out_dir "$BULK"
    "$PY_EVAL" "$PHASE4/build_ret_circuits.py" --out_dir "$RET"

    section "$TAG — RET pass (concurrency 1)"
    "$PY_EVAL" "$HERE/run_monolithic_eval.py" --model_path "$MERGED" \
        --circuits_jsonl "$RET/circuits.jsonl" --output_dir "$RET" \
        --max_model_len $MAX_MODEL_LEN --gpu_memory_utilization 0.90 --max_tokens 8192 \
        --max_concurrency 1 --label "${LABEL}_ret"

    # Stop before the day-long bulk pass if nothing in distribution parsed
    # (that would mean a broken setup, e.g. a wrong chat template).
    section "$TAG — sanity check (RET pass, n <= 7)"
    "$PY_EVAL" - "$RET/per_circuit_metrics.jsonl" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
ind = [r for r in rows if r["n"] <= 7]
ok = sum(r["success"] for r in ind)
print(f"in-distribution RET circuits: {len(ind)}, fully parsed: {ok}, mean SA: {sum(r['sa'] for r in ind)/len(ind):.3f}")
if ok == 0:
    sys.exit("No in-distribution output parsed; inspect _ret/chain_results.jsonl before the bulk pass")
EOF

    section "$TAG — bulk pass (concurrency 8, 4,024 circuits)"
    "$PY_EVAL" "$HERE/run_monolithic_eval.py" --model_path "$MERGED" \
        --circuits_jsonl "$BULK/circuits.jsonl" --output_dir "$BULK" \
        --max_model_len $MAX_MODEL_LEN --gpu_memory_utilization 0.90 --max_tokens 8192 \
        --max_concurrency 8 --label "${LABEL}_bulk"

    section "$TAG — post-processing (Phase 4 post-processor, unchanged)"
    "$PY_EVAL" "$PHASE4/paper_eval_postprocess.py" --bulk_dir "$BULK" --ret_dir "$RET" \
        --out_dir "$OUT" --n_min 2 --n_max 19

    section "$TAG — plots"
    cd "$EVAL_DIR"
    "$PY_EVAL" plot_results.py --phase 2 --mode full --circuit_source paper --cf_key both --results_dir "$OUT"
    "$PY_EVAL" plot_oracle_accuracy.py \
        --input "$OUT/oracle_extraction_full_2_19_paper.json" "Section 1 (Oracle Extraction)" \
        --output "$OUT/oracle_accuracy.png" --title "$TAG — Section 1 (Oracle Extraction) Accuracy"
    "$PY_EVAL" plot_oracle_accuracy.py \
        --input "$OUT/marked_state_accuracy_full_2_19_paper.json" "Section 2 (Marked-State Identification)" \
        --output "$OUT/marked_state_accuracy.png" \
        --title "$TAG — Section 2 Marked-State Accuracy (strict conditional on Section 1)"
    "$PY_EVAL" plot_oracle_accuracy.py \
        --input "$OUT/agent3_accuracy_full_2_19_paper.json" "Section 3 (Probability Distribution; mean SA)" \
        --output "$OUT/agent3_accuracy.png" \
        --title "$TAG — Section 3 Probability-Distribution Accuracy (strict conditional on Section 2)"

    section "$TAG done. Results in $OUT"
fi
