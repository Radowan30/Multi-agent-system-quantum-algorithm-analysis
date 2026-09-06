# Reproduce Results

Step-by-step guide for re-running every phase of the thesis *"LLM Based
Multi-Agent System for Improving Quantum Algorithm Symbolic Analysis"* on
your own machine.

Following this guide end-to-end produces the same fine-tuned models, the same
evaluation metric values, and the same figures that appear in the **`Our
Results/`** folder of this repository.

## What you will end up with

- **Phase 1:** a recreated GroverGPT+ model (LoRA fine-tune of LLaMA 3 8B).
- **Phase 2:** two fine-tuned LLMs with larger context windows — `Llama-3-8B-Instruct-262k` and `LLaMA 3.1 8B`.
- **Phase 3:** a prompt-engineered three-agent pipeline, evaluated on four LLM variations.
- **Phase 4:** a fully fine-tuned three-agent system on agent-specific datasets, with the option to serve it via vLLM behind an OpenAI-compatible chat endpoint.

The serving step at the end produces an HTTP endpoint that the **`Prototype
Interaction Application/`** folder of this repository can immediately connect to.

---

## 1. Prerequisites

### Hardware

- **GPU:** an NVIDIA GPU with at least **80 GB of VRAM** and Compute Capability 8.0+. The thesis was run on an **NVIDIA RTX PRO 6000 Blackwell Workstation Edition (96 GB)**.
- **CPU:** any modern x86-64 CPU.
- **RAM:** **32 GB minimum** for the serving step alone; **64 GB recommended** to comfortably run vLLM + the agent proxy + a desktop session without swap. The Phase 4 merged model occupies the GPU (≈30 GB VRAM); host RAM usage per vLLM instance is roughly **25 – 35 GB** depending on `max-model-len` and the page cache.
- **Disk:** at least **500 GB** of free space (model weights, datasets, evaluation outputs).

### Software

- **OS:** Linux (the thesis was run on Ubuntu 24.04 LTS inside WSL2 on Windows 11; native Ubuntu works the same).
- **NVIDIA driver:** a driver that supports **CUDA 12.8 runtime** (driver version ≥ 555). Check with `nvidia-smi`.
- **Python:** **3.11.x**. We used 3.11.14.
- **Git, wget, build-essential.**

### Acknowledgement

This work builds directly upon the **GroverGPT+** research. The original source
code and the QASM training dataset are released by the original authors at:

> **GroverGPT+ official GitHub repository:**
> [https://github.com/JimXiong16/GroverGPT-2](https://github.com/JimXiong16/GroverGPT-2)

We thank the authors for releasing both their code and their dataset openly.

---

## 2. Clone this repository

```bash
git clone <this repository URL>
cd <repository folder>
```

The folder layout you will be working in:

```
Reproduce Results/
├── envs/                              — pip requirements for the three venvs
├── Phase 1/                           — tokenizer extension + training/merge configs (LLaMA 3 8B)
├── Phase 2/
│   ├── Llama-3-8B-Instruct-262k/
│   └── LLaMA 3.1 8B/
├── Phase 3/                           — prompt-engineered multi-agent system
│   └── prompt_templates/
├── Phase 4/                           — agent-specific fine-tuned multi-agent system
│   └── dataset/
├── evaluation_pipeline/               — shared metrics + plotting code used by all phases
└── merge_adapter.py                   — LoRA adapter → merged model utility
```

---

## 3. Set up the three Python virtual environments

The project uses three separate Python 3.11 virtual environments to keep
training, evaluation/orchestration, and inference dependencies cleanly
separated. All three are pinned for **CUDA 12.8** with PyTorch 2.11.

Pick a working directory (referred to as `$WORK` below) and create the venvs
inside it:

```bash
export WORK=$HOME/grover-multiagent-reproduction
mkdir -p "$WORK" && cd "$WORK"

python3.11 -m venv venv-train
python3.11 -m venv venv-eval
python3.11 -m venv venv-inference
```

Install pip + setuptools fresh in each, then install the pinned requirements:

```bash
# --- venv-train (LLaMA-Factory + PEFT training stack) ---
source venv-train/bin/activate
python -m ensurepip --upgrade
pip install --upgrade pip wheel setuptools
pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
pip install -r <path to>/envs/venv-train_requirements.txt
deactivate

# --- venv-eval (evaluation + agent orchestration + Qiskit + vLLM) ---
source venv-eval/bin/activate
python -m ensurepip --upgrade
pip install --upgrade pip wheel setuptools
pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
pip install -r <path to>/envs/venv-eval_requirements.txt
deactivate

# --- venv-inference (lean vLLM serving environment) ---
source venv-inference/bin/activate
python -m ensurepip --upgrade
pip install --upgrade pip wheel setuptools
pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
pip install -r <path to>/envs/venv-inference_requirements.txt
deactivate
```

> **Note:** Some packages (e.g. `vllm`, `bitsandbytes`) are large and may take
> 10–20 minutes each to install. Keep network access available throughout.

Verify each venv:

```bash
$WORK/venv-train/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
$WORK/venv-eval/bin/python  -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
$WORK/venv-inference/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

All three should print `2.11.0+cu128 12.8 True`.

---

## 4. Install LLaMA-Factory

LLaMA-Factory is the training pipeline used in every fine-tuning step. The
pinned version is `0.9.5.dev0`.

```bash
cd $WORK
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
git checkout v0.9.5            # exact release we used; remove this line for the latest
$WORK/venv-train/bin/pip install -e ".[torch,metrics]"
```

Register the Grover datasets so LLaMA-Factory can find them. Copy the
provided dataset-registry snippet into LLaMA-Factory's `data/dataset_info.json`:

```bash
# The snippet sits at envs/llamafactory_dataset_info_grover_only.json — merge
# its entries into LLaMA-Factory/data/dataset_info.json (a normal JSON merge).
```

---

## 5. Download the dataset

The training dataset is the **same one released by the GroverGPT+ authors**.
Download it from their official repository:

> **Dataset source:**
> [https://github.com/JimXiong16/GroverGPT-2](https://github.com/JimXiong16/GroverGPT-2)
> (look inside the `data` folder for `data_MMS.rar` and the prepared JSON files
> `Grover_FullCircuit_2_7_MMS.json` and `Grover_Oracle_2_10_MMS.json`)

After downloading, place the two JSON dataset files into your LLaMA-Factory
`data/` folder:

```bash
cp Grover_FullCircuit_2_7_MMS.json  $WORK/LLaMA-Factory/data/
cp Grover_Oracle_2_10_MMS.json      $WORK/LLaMA-Factory/data/
```

You will also need the raw `data_MMS` folder of QASM circuits for the
evaluation step. Unrar `data_MMS.rar` into a known location, e.g.
`$WORK/data_MMS/`.

---

## 6. Download the base models

Download each base model from Hugging Face into `$WORK/models/`:

```bash
mkdir -p $WORK/models && cd $WORK/models

# Base model for Phase 1
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct \
    --local-dir Meta-Llama-3-8B-Instruct

# Base models for Phases 2 and 4
huggingface-cli download gradientai/Llama-3-8B-Instruct-262k \
    --local-dir Llama-3-8B-Instruct-262k

huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
    --local-dir Llama-3.1-8B-Instruct
```

> **Note:** Both Meta models require accepting their licence on the Hugging
> Face website and authenticating with `huggingface-cli login`.

---

## 7. Phase 1 — Recreate GroverGPT+

**Goal:** reproduce the GroverGPT+ baseline by fine-tuning LLaMA 3 8B.

### 7.1 Extend the tokenizer with quantum-native vocabulary

The released GroverGPT+ tokenizer-extension code adds only 96 entries (against
the 266 claimed in the paper). Our extended version adds **139** new entries
by re-running the same logic on the full QASM corpus.

```bash
cd "Phase 1/"
$WORK/venv-train/bin/python generate_input_list.py   # builds inputs_list_all_data_MMS.json
$WORK/venv-train/bin/python extend_tokenizer.py      # writes the extended tokenizer files
```

Copy the extended tokenizer files into the base model directory so LLaMA-Factory
picks them up during training:

```bash
cp -r ./Grover_Extend_Tokenizer_data_MMS/* $WORK/models/Meta-Llama-3-8B-Instruct/
```

### 7.2 Fine-tune

Edit `train.yaml` and replace the absolute path `model_name_or_path` with your
local `$WORK/models/Meta-Llama-3-8B-Instruct` path. Then:

```bash
cd $WORK/LLaMA-Factory
$WORK/venv-train/bin/llamafactory-cli train  $WORK/<repo>/Phase\ 1/train.yaml
```

This trains a LoRA adapter for 10 epochs (≈4–6 hours on the RTX PRO 6000) and
saves it under `$WORK/saves/Meta-Llama-3-8B-Instruct/lora/GroverGPT+_alpha32/`.

### 7.3 Merge the adapter into a single model

```bash
$WORK/venv-train/bin/llamafactory-cli export $WORK/<repo>/Phase\ 1/merge.yaml
```

The merged model is written to `$WORK/saves/Meta-Llama-3-8B-Instruct/merged/GroverGPT+_alpha32/`.

### 7.4 Evaluate

The evaluation produces every figure under `Our Results/Phase 1/`.

```bash
cd $WORK/<repo>/evaluation_pipeline
$WORK/venv-eval/bin/python run_eval_phase1.py \
    --model_path  $WORK/saves/Meta-Llama-3-8B-Instruct/merged/GroverGPT+_alpha32 \
    --data_dir    $WORK/data_MMS \
    --output_dir  $WORK/results/phase-1
```

Then generate the plots:

```bash
$WORK/venv-eval/bin/python plot_results.py \
    --phase 1 --mode full --circuit_source paper --results_dir $WORK/results/phase-1
$WORK/venv-eval/bin/python plot_results.py \
    --phase 1 --mode oracle --circuit_source paper --results_dir $WORK/results/phase-1
$WORK/venv-eval/bin/python plot_oracle_accuracy.py \
    --input $WORK/results/phase-1/oracle_extraction_full_2_9_paper.json "Full-circuit input" \
            $WORK/results/phase-1/oracle_extraction_oracle_2_20_paper.json "Oracle-only input" \
    --output $WORK/results/phase-1/oracle_accuracy_dual.png
```

---

## 8. Phase 2 — Fine-tune LLMs with a larger context window

Phase 2 trains the **same dataset** and the **same hyperparameters** as Phase 1
on two larger-context base models. Repeat the same five-step sequence per
model.

### 8.1 Llama-3-8B-Instruct-262k

```bash
cd $WORK/<repo>/Phase\ 2/Llama-3-8B-Instruct-262k

# 1. Extend the tokenizer (re-uses the Phase 1 logic on this base model)
$WORK/venv-train/bin/python extend_tokenizer.py
cp -r ./Grover_Extend_Tokenizer_data_MMS_gradient/* $WORK/models/Llama-3-8B-Instruct-262k/

# 2. Edit train.yaml to point model_name_or_path at $WORK/models/Llama-3-8B-Instruct-262k
# 3. Train
cd $WORK/LLaMA-Factory
$WORK/venv-train/bin/llamafactory-cli train  $WORK/<repo>/Phase\ 2/Llama-3-8B-Instruct-262k/train.yaml

# 4. Merge
$WORK/venv-train/bin/llamafactory-cli export $WORK/<repo>/Phase\ 2/Llama-3-8B-Instruct-262k/merge.yaml

# 5. Evaluate (the run_eval_phase2.py script handles n ∈ {2..19} for full-circuit and n ∈ {2..20} for oracle)
cd $WORK/<repo>/evaluation_pipeline
$WORK/venv-eval/bin/python run_eval_phase2.py \
    --model_path  $WORK/saves/Llama-3-8B-Instruct-262k/merged/Gradient262k_alpha32 \
    --data_dir    $WORK/data_MMS \
    --output_dir  $WORK/results/phase-2/Llama-3-8B-Instruct-262k

# 6. Generate plots (same script set as Phase 1)
$WORK/venv-eval/bin/python plot_results.py \
    --phase 2 --mode full --circuit_source paper --results_dir $WORK/results/phase-2/Llama-3-8B-Instruct-262k
$WORK/venv-eval/bin/python plot_results.py \
    --phase 2 --mode oracle --circuit_source paper --results_dir $WORK/results/phase-2/Llama-3-8B-Instruct-262k
$WORK/venv-eval/bin/python plot_oracle_accuracy.py \
    --input $WORK/results/phase-2/Llama-3-8B-Instruct-262k/oracle_extraction_full_2_19_paper.json "Full-circuit input" \
            $WORK/results/phase-2/Llama-3-8B-Instruct-262k/oracle_extraction_oracle_2_20_paper.json "Oracle-only input" \
    --output $WORK/results/phase-2/Llama-3-8B-Instruct-262k/oracle_accuracy_dual.png
```

There is also a shell script `run_full_pipeline.sh` that runs steps 1–6 in
order if you prefer; edit the paths inside it first.

### 8.2 LLaMA 3.1 8B

Repeat the same six steps in `Phase 2/LLaMA 3.1 8B/`, pointing at
`$WORK/models/Llama-3.1-8B-Instruct` instead. After the standard plot scripts,
also generate the alternative RET normalisation (the thesis Figure 21 and 22):

```bash
$WORK/venv-eval/bin/python plot_ret_normalized.py \
    --results $WORK/results/phase-2/LLaMA-3.1-8B/results_full_2_19_paper.json \
    --normalise_to 3 \
    --output $WORK/results/phase-2/LLaMA-3.1-8B/ret_normalized_n3_paper.png
```

---

## 9. Phase 3 — Multi-agent system through prompt engineering

Phase 3 does not fine-tune anything new; it wraps the Phase 2 fine-tuned
models (plus their untrained counterparts) in a sequential three-agent
pipeline orchestrated through LangChain.

### 9.1 Serve the model with vLLM

In a dedicated terminal, start a vLLM OpenAI-compatible server. For each of
the four LLM variations being evaluated, point `--model` at the appropriate
local model directory:

```bash
# Example — Phase 2 fine-tuned Llama-3-8B-Instruct-262k
$WORK/venv-inference/bin/python -m vllm.entrypoints.openai.api_server \
    --model $WORK/saves/Llama-3-8B-Instruct-262k/merged/Gradient262k_alpha32 \
    --served-model-name finetuned_gradient262k \
    --host 127.0.0.1 --port 8000 \
    --max-model-len 150000 \
    --gpu-memory-utilization 0.90 \
    --dtype bfloat16 --enable-prefix-caching
```

For the untrained variations, point `--model` at the unmodified base model
directory (e.g. `$WORK/models/Llama-3-8B-Instruct-262k`).

### 9.2 Run the three-agent pipeline evaluation

In a second terminal:

```bash
cd $WORK/<repo>/Phase\ 3
$WORK/venv-eval/bin/python run_eval.py \
    --vllm_url http://127.0.0.1:8000/v1 \
    --vllm_model finetuned_gradient262k \
    --circuits_jsonl $WORK/data_MMS/stratified/circuits.jsonl \
    --qasm_dir       $WORK/data_MMS/stratified/qasm \
    --output_dir     $WORK/results/phase-3/finetuned_gradient262k
```

Repeat for the other three variations, changing `--vllm_model` and `--output_dir`
each time. The orchestration code lives in `chain.py` / `parsers.py` /
`prompts.py`, and the agent prompts (with worked examples) live in
`prompt_templates/`.

### 9.3 Post-process and plot

Use the same `paper_eval_postprocess.py` from the Phase 4 folder to aggregate
the per-circuit chain results into per-n metrics, then the standard plotting
scripts to produce Figures 23 – 38.

---

## 10. Phase 4 — Multi-agent system fine-tuned on agent-specific data

Phase 4 replaces Phase 3's prompt-engineered agents with fine-tuned ones, each
trained on a dataset that contains only its own sub-task's input-output pairs.

### 10.1 Generate the three agent-specific datasets

Augment the GroverGPT+ training data into three agent-specific subsets:

```bash
cd $WORK/<repo>/Phase\ 4/dataset
$WORK/venv-eval/bin/python generate_agent_datasets.py \
    --input_file $WORK/LLaMA-Factory/data/Grover_FullCircuit_2_7_MMS.json \
    --output_dir .
```

This produces `Grover_Agent1_2_7_MMS.json`, `Grover_Agent2_2_7_MMS.json`,
`Grover_Agent3_2_7_MMS.json`. Copy them into LLaMA-Factory's `data/` folder
and register them in `dataset_info.json` (the snippet at
`envs/llamafactory_dataset_info_grover_only.json` already contains the
correct entries).

### 10.2 Fine-tune each base model on the joint of the three agent datasets

```bash
cd $WORK/LLaMA-Factory

# Llama-3-8B-Instruct-262k base
$WORK/venv-train/bin/llamafactory-cli train  $WORK/<repo>/Phase\ 4/train_Llama-3-8B-Instruct-262k.yaml
$WORK/venv-train/bin/llamafactory-cli export $WORK/<repo>/Phase\ 4/merge_Llama-3-8B-Instruct-262k.yaml

# LLaMA 3.1 8B base
$WORK/venv-train/bin/llamafactory-cli train  $WORK/<repo>/Phase\ 4/train_LLaMA-3.1-8B.yaml
$WORK/venv-train/bin/llamafactory-cli export $WORK/<repo>/Phase\ 4/merge_LLaMA-3.1-8B.yaml
```

Each training run takes roughly 6–8 hours on the RTX PRO 6000.

### 10.3 Evaluate the Phase 4 multi-agent system end-to-end

```bash
# Terminal 1 — serve the Phase 4 merged model with vLLM
$WORK/venv-inference/bin/python -m vllm.entrypoints.openai.api_server \
    --model $WORK/saves/Llama-3-8B-Instruct-262k/merged/Phase4_alpha32 \
    --served-model-name Phase4_alpha32 \
    --host 127.0.0.1 --port 8000 \
    --max-model-len 150000 --gpu-memory-utilization 0.90 \
    --dtype bfloat16 --enable-prefix-caching

# Terminal 2 — run the full paper-eval pipeline (build → RET → bulk → postprocess → plots)
bash "$WORK/<repo>/Phase 4/run_paper_eval_Llama-3-8B-Instruct-262k.sh"
```

The `run_paper_eval_*.sh` script automates: building the 4024-circuit paper
evaluation set + the 3-mixed-k RET set, running the chain inference in
concurrency-8 bulk mode, running the chain again in single-instance mode for
RET timings, post-processing into Phase 1/2-compatible JSONs, and producing
every figure that appears in `Our Results/Phase 4/`.

Repeat for the LLaMA 3.1 8B base model using `run_paper_eval_LLaMA-3.1-8B.sh`.

---

## 11. Serve the final Phase 4 model with vLLM (for downstream applications)

Once Phase 4 is fine-tuned and merged, the resulting model can be served as a
standalone OpenAI-compatible chat-completion endpoint that any other
application can talk to (including the **`Prototype Interaction Application/`**
in this repository).

The simplest way is to use the bundled `run_agent_proxy.sh`:

```bash
bash "$WORK/<repo>/Phase 4/run_agent_proxy.sh"
```

This script brings up two long-running processes:

1. **vLLM** on `http://127.0.0.1:8000` — serves the merged Phase 4 model directly.
2. **agent_proxy** on `http://0.0.0.0:8080/v1` — an OpenAI-compatible facade that wraps the three-agent pipeline so a chat client only sees a single endpoint that takes a QASM circuit and streams back the per-agent reasoning + final probability distribution.

Point any OpenAI-compatible client (Open WebUI, LibreChat, Cline, custom code,
or the **`Prototype Interaction Application/`** in this repository) at:

- **Base URL:** `http://<host>:8080/v1`
- **Model name:** `Phase4_Quantum_Agents`

Environment variables you can override (sensible defaults are in the script):
`MERGED_MODEL`, `VLLM_PORT`, `PROXY_PORT`, `MAX_MODEL_LEN`.

Press **Ctrl-C** to stop both processes cleanly. vLLM startup logs go to
`/tmp/agent_proxy_vllm.log`.

If you only need the raw vLLM endpoint (without the multi-agent facade), use
the direct vLLM command shown in §10.3 instead.

---

## 12. Acknowledgements

This work is built on top of the **GroverGPT+** research:

> M. Chen, *et al.*, "Symbolic analysis of Grover search algorithm via
> Chain-of-Thought reasoning and quantum-native tokenization," *npj Quantum
> Information*, vol. 12, no. 1, art. 48, 2026.
>
> Source code and dataset: <https://github.com/JimXiong16/GroverGPT-2>

We also acknowledge the open-source LLaMA-Factory, vLLM, LangChain, Qiskit,
and PyTorch projects, without which this work would not have been possible.
