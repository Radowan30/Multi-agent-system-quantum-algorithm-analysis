# Shared locations for the reproduction scripts (see README.md, section 2).
#
# Every run_*.sh script sources this file, so the only thing you need to set is
# WORK, your working directory:
#
#     export WORK=$HOME/grover-multiagent-reproduction
#
# Expected layout under WORK (created step by step by the README):
#     $WORK/venv-train  $WORK/venv-eval  $WORK/venv-inference
#     $WORK/LLaMA-Factory/           training toolkit (datasets go in LLaMA-Factory/data/)
#     $WORK/data_MMS/                raw GroverGPT+ circuits (grover_n2/ ... grover_n24/)
#     $WORK/models/                  downloaded base models + their "-quantum" training copies
#     $WORK/saves/                   LoRA adapters and merged models
#     $WORK/eval_circuits/manifests/ evaluation circuit manifests
#     $WORK/results/                 evaluation outputs
#
# The LLaMA-Factory YAML files use paths relative to $WORK/LLaMA-Factory
# (../models, ../saves), so they work unchanged with this layout.

if [ -z "${WORK:-}" ]; then
    echo "WORK is not set. Run: export WORK=<your working directory> (README.md, section 2)" >&2
    return 1 2>/dev/null || exit 1
fi
export WORK

REPRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$REPRO_DIR/evaluation_pipeline"

LLAMA_FACTORY="$WORK/LLaMA-Factory"
LF_CLI="$WORK/venv-train/bin/llamafactory-cli"
PY_TRAIN="$WORK/venv-train/bin/python"
PY_EVAL="$WORK/venv-eval/bin/python"
PY_INFER="$WORK/venv-inference/bin/python"

DATA_MMS="$WORK/data_MMS"
MODELS="$WORK/models"
SAVES="$WORK/saves"
RESULTS="$WORK/results"
MANIFESTS="$WORK/eval_circuits/manifests"

section() {
    echo
    echo "================================================================"
    echo "  $*"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================"
}

# Fail early with a clear message when an earlier step has not been run.
require() {
    if [ ! -e "$1" ]; then
        echo "Missing: $1" >&2
        echo "  -> $2" >&2
        exit 1
    fi
}

# Train a LoRA adapter and merge it, from inside LLaMA-Factory (the YAMLs use
# paths relative to it).
train_and_merge() {
    local train_yaml="$1" merge_yaml="$2"
    section "Training ($(basename "$train_yaml"))"
    (cd "$LLAMA_FACTORY" && "$LF_CLI" train "$train_yaml")
    section "Merging the adapter ($(basename "$merge_yaml"))"
    (cd "$LLAMA_FACTORY" && "$LF_CLI" export "$merge_yaml")
}

# The merged model must render the simple llama3 chat template it was trained
# with, otherwise vLLM prompts differ from training and every output fails.
check_chat_template() {
    "$PY_EVAL" - "$1" <<'PY'
import sys
from transformers import AutoTokenizer
want = ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\ntest<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n")
tok = AutoTokenizer.from_pretrained(sys.argv[1])
got = tok.apply_chat_template([{"role": "user", "content": "test"}], tokenize=False,
                              add_generation_prompt=True)
if got != want:
    sys.exit(f"[template] {sys.argv[1]} renders an unexpected chat template: {got!r}")
print(f"[template] OK, simple llama3 template ({sys.argv[1]})")
PY
}
