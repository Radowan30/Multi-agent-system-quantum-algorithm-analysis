# Reproduce Results

Step-by-step guide for re-running every phase of the thesis *"LLM Based
Multi-Agent System for Improving Quantum Algorithm Symbolic Analysis"* on your
own machine. Following it end to end produces the same fine-tuned models, the
same evaluation metrics and the same figures as the **`Our Results/`** folder.

Each phase has one script that runs it from training to plots. The sections
below set up everything those scripts need, in order.

| Phase | What it produces | Script | Time on an RTX PRO 6000 |
|---|---|---|---|
| 1 | Recreated GroverGPT+ (LLaMA 3 8B) | `Phase 1/run_phase1.sh` | ≈5 h training + evaluation |
| 2 | Two larger-context fine-tunes (Llama-3-8B-Instruct-262k, LLaMA 3.1 8B) | `Phase 2/<model>/run_phase2.sh` | ≈5 h training + 10–40 h evaluation per model |
| 3 | Prompt-engineered three-agent pipeline on four LLM variations | `Phase 3/run_phase3.sh` | ≈1 h |
| 4 | Fine-tuned three-agent system (two base models) | `Phase 4/run_phase4_<model>.sh`, `Phase 4/run_paper_eval_<model>.sh` | ≈4 h training + ≈1 day evaluation per model |
| 4 (ablation) | Monolithic model trained on the same data | `Phase 4/Monolithic Model/run_<model>.sh` | ≈3.5 h training + ≈1 day evaluation per model |

---

## 1. Prerequisites

### Hardware

- **GPU:** an NVIDIA GPU with at least **80 GB of VRAM** and Compute Capability 8.0+. The thesis was run on an **NVIDIA RTX PRO 6000 Blackwell Workstation Edition (96 GB)**.
- **CPU:** any modern x86-64 CPU.
- **RAM:** **64 GB recommended**. A vLLM instance uses roughly 25–35 GB of host RAM. Do not run two vLLM processes at the same time; the evaluation scripts start and stop their own vLLM server.
- **Disk:** at least **500 GB** free (model weights, adapters, merged models, evaluation outputs).

### Software

- **OS:** Linux (the thesis used Ubuntu 24.04 LTS inside WSL2 on Windows 11; native Ubuntu works the same).
- **NVIDIA driver:** one that supports the **CUDA 12.8 runtime** (driver ≥ 555). Check with `nvidia-smi`.
- **Python 3.11** (we used 3.11.14), **Git**, **wget**, **build-essential**, and an **unrar** tool for the dataset archive.
- A **Hugging Face account** with access to the Meta Llama models (accept their licences on the website, then `huggingface-cli login`).

### Acknowledgement

This work builds directly upon the **GroverGPT+** research. The original code
and the QASM dataset are released by the original authors at
[https://github.com/JimXiong16/GroverGPT-2](https://github.com/JimXiong16/GroverGPT-2).
We thank them for releasing both openly.

---

## 2. Choose a working directory and clone this repository

Everything that is downloaded or generated lives under one working directory,
`$WORK`. **Set `WORK` in every terminal you use** (or add the line to
`~/.bashrc`): all scripts read it.

```bash
export WORK=$HOME/grover-multiagent-reproduction
mkdir -p "$WORK" && cd "$WORK"
git clone https://github.com/Radowan30/Multi-agent-system-quantum-algorithm-analysis.git
export RR="$WORK/Multi-agent-system-quantum-algorithm-analysis/Reproduce Results"
```

`$RR` is used in the commands below as a shorthand for this folder. The layout
you will end up with:

```
$WORK/
├── Multi-agent-system-quantum-algorithm-analysis/   this repository
├── venv-train/  venv-eval/  venv-inference/         Python environments (section 3)
├── LLaMA-Factory/                                   training toolkit; datasets in LLaMA-Factory/data/ (section 4)
├── data_MMS/                                        GroverGPT+ QASM circuits: grover_n2/, grover_n3/, … (section 5)
├── models/                                          downloaded base models + "-quantum" training copies (section 6)
├── eval_circuits/manifests/                         evaluation circuit manifests (section 7)
├── saves/                                           LoRA adapters and merged models (written by the phase scripts)
└── results/                                         evaluation outputs, one folder per phase
```

The LLaMA-Factory configs in this repository use paths relative to
`$WORK/LLaMA-Factory` (`../models/...`, `../saves/...`), so they work without
editing as long as you keep this layout.

Inside this folder:

```
Reproduce Results/
├── paths.sh               shared locations, sourced by every run script
├── envs/                  pip requirements for the three venvs + dataset registration
├── tokenizer/             quantum-native tokenizer: token list, chat template, model preparation
├── evaluation_pipeline/   shared evaluation code: circuit manifests, metrics, Phase 1/2 runners, plots
├── Phase 1/               GroverGPT+ recreation: train/merge configs + run_phase1.sh
├── Phase 2/               one folder per base model: train/merge configs + run_phase2.sh
├── Phase 3/               prompt-engineered agents, prompts, Phase 3 circuit set, run_phase3.sh
└── Phase 4/               agent datasets, train/merge configs, multi-agent evaluation,
    └── Monolithic Model/  the monolithic ablation
```

---

## 3. Set up the three Python virtual environments

Three separate Python 3.11 environments keep training, evaluation and serving
dependencies apart. All are pinned for **CUDA 12.8** with PyTorch 2.11.

| venv | Used for |
|---|---|
| `venv-train` | LLaMA-Factory training and merging, tokenizer preparation |
| `venv-eval` | all evaluation (vLLM, Qiskit, LangChain), post-processing and plots |
| `venv-inference` | serving the final model for the prototype application |

```bash
cd "$WORK"
for v in venv-train venv-eval venv-inference; do
    python3.11 -m venv "$v"
    "$v/bin/python" -m ensurepip --upgrade
    "$v/bin/pip" install --upgrade pip wheel setuptools
    "$v/bin/pip" install --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
done
venv-train/bin/pip     install -r "$RR/envs/venv-train_requirements.txt"
venv-eval/bin/pip      install -r "$RR/envs/venv-eval_requirements.txt"
venv-inference/bin/pip install -r "$RR/envs/venv-inference_requirements.txt"
```

Some packages (e.g. `vllm`, `bitsandbytes`) are large and can take 10–20
minutes each. Check each venv:

```bash
for v in venv-train venv-eval venv-inference; do
    "$WORK/$v/bin/python" -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
done
```

All three should print `2.11.0+cu128 12.8 True`.

---

## 4. Install LLaMA-Factory and register the datasets

LLaMA-Factory is used for every fine-tuning and merging step. We used commit
`b5afabe` (version `0.9.5.dev0`); check out that commit for an exact
reproduction.

```bash
cd "$WORK"
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
git checkout b5afabe
"$WORK/venv-train/bin/pip" install -e ".[torch,metrics]"

# Add the Grover datasets to LLaMA-Factory's data/dataset_info.json
"$WORK/venv-train/bin/python" "$RR/envs/register_datasets.py"
```

`register_datasets.py` adds six entries (the two GroverGPT+ datasets, the three
Phase 4 agent datasets and the monolithic ablation dataset) and keeps
everything else in the file. The data files themselves are added in section 5.

---

## 5. Get the data

**GroverGPT+ dataset.** Download from the authors' repository
([https://github.com/JimXiong16/GroverGPT-2](https://github.com/JimXiong16/GroverGPT-2),
`data` folder):

- `data_MMS.rar`: the raw QASM circuits. Extract it so that the circuit folders
  sit directly in `$WORK/data_MMS/` (`$WORK/data_MMS/grover_n2/`, `grover_n3/`, …).
- `Grover_FullCircuit_2_7_MMS.json` and `Grover_Oracle_2_10_MMS.json`: the
  training sets for Phases 1 and 2.

```bash
cp Grover_FullCircuit_2_7_MMS.json Grover_Oracle_2_10_MMS.json "$WORK/LLaMA-Factory/data/"
```

**Phase 4 agent datasets.** These ship with this repository:

```bash
cp "$RR/Phase 4/dataset/"Grover_Agent*_2_7_MMS.json "$WORK/LLaMA-Factory/data/"
```

You can regenerate them from `data_MMS` instead; the script is deterministic
and reproduces the shipped files byte for byte:

```bash
"$WORK/venv-eval/bin/python" "$RR/Phase 4/dataset/generate_agent_datasets.py" --out_dir "$WORK/LLaMA-Factory/data"
```

The monolithic ablation dataset is built automatically by its run script
(section 12).

---

## 6. Download the base models and prepare their training copies

Download the three base models into `$WORK/models/`. **These downloaded
directories are never modified**: Phase 3 evaluates the untouched ("untrained")
models, and the evaluation uses their original tokenizer as the baseline for
the compression metrics.

```bash
mkdir -p "$WORK/models" && cd "$WORK/models"
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir Meta-Llama-3-8B-Instruct   # Phase 1
huggingface-cli download gradientai/Llama-3-8B-Instruct-262k --local-dir Llama-3-8B-Instruct-262k   # Phases 2-4
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct    --local-dir Llama-3.1-8B-Instruct      # Phases 2-4
```

**Quantum-native tokenizer.** GroverGPT+ extends the LLaMA 3 tokenizer with
QASM tokens. The released extension code builds its vocabulary from only 14
sample circuits (96 new tokens, against the 266 stated in the paper); we
re-ran the same rule on all 4,956 circuits in `data_MMS`, which gives **139**
new tokens. `tokenizer/quantum_tokens.json` lists them in the order we added
them, so your new token IDs match ours.

For each base model, `prepare_quantum_model.py` writes a separate
`<model>-quantum` directory that the training configs use. It contains the
extended tokenizer, the simple LLaMA 3 chat template that LLaMA-Factory's
`llama3` template trains with, the model configs, and links to the original
weight files (no 16 GB copy). For LLaMA 3.1 8B it also sets the end-of-sequence
ids to those of LLaMA 3 (`128009` in `config.json`, `[128001, 128009]` in
`generation_config.json`) and replaces LLaMA 3.1's longer default chat
template, which would otherwise not match training.

```bash
cd "$RR/tokenizer"
# Optional: rebuild the QASM corpus so the script can re-derive and check the token set
"$WORK/venv-train/bin/python" generate_input_list.py
for m in Meta-Llama-3-8B-Instruct Llama-3-8B-Instruct-262k Llama-3.1-8B-Instruct; do
    "$WORK/venv-train/bin/python" prepare_quantum_model.py \
        --base_model "$WORK/models/$m" --output_dir "$WORK/models/$m-quantum" \
        --inputs_list "$WORK/tokenizer/inputs_list_all_data_MMS.json"
done
```

(Leave out `--inputs_list` if you skipped `generate_input_list.py`.) Each run
ends with `vocabulary: 128395 tokens (+139), chat template: simple llama3`.

---

## 7. Generate the evaluation circuit manifests

All published results use circuits drawn from `data_MMS` with the GroverGPT+
per-n counts: min(max(100, 2ⁿ), available), i.e. 4, 36, 100, 100, 100, 128,
256 and then 300 circuits for n = 2, 3, …, 9 onwards (4,024 circuits for
n = 2–19). Selection is deterministic (sorted file list, seed 42), so these
manifests are identical to ours.

```bash
cd "$RR/evaluation_pipeline"
"$WORK/venv-eval/bin/python" generate_eval_circuits.py --mode full   --n_min 2 --n_max 9  --circuit_source paper   # Phase 1
"$WORK/venv-eval/bin/python" generate_eval_circuits.py --mode full   --n_min 2 --n_max 19 --circuit_source paper   # Phases 2-4
"$WORK/venv-eval/bin/python" generate_eval_circuits.py --mode oracle --n_min 2 --n_max 20 --circuit_source paper   # Phases 1-2
```

The manifests go to `$WORK/eval_circuits/manifests/`. The phase scripts also
create any missing manifest themselves.

---

## 8. Phase 1 — Recreate GroverGPT+

Fine-tunes LLaMA 3 8B Instruct with LoRA on the two GroverGPT+ datasets using
the paper's Table II settings (lora_alpha 32, see the comments in
`Phase 1/train.yaml`), then evaluates full-circuit inputs (n = 2–9, the most
that fits the 8,192-token window) and oracle-only inputs (n = 2–20).

```bash
bash "$RR/Phase 1/run_phase1.sh"
```

The script trains and merges (`$WORK/saves/Meta-Llama-3-8B-Instruct/merged/GroverGPT+_alpha32`),
checks the merged model's chat template, runs both evaluations, writes
`cot_traces.md` (model output next to the ground-truth reasoning, one circuit
per (n, k) cell), computes the Oracle Extraction Accuracy and draws the plots.

Results in `$WORK/results/phase-1/`:

| File | `Our Results/Phase 1/` |
|---|---|
| `cr_srr_paper.png` | Figure 9 |
| `sa_cf_full_paper.png`, `sa_cf_oracle_paper.png` | Figures 10, 11 |
| `ret_paper.png` | Figure 12 |
| `oracle_accuracy_dual.png` | Figure 13 |
| `results_full_2_9_paper.json`, `results_oracle_2_20_paper.json`, `oracle_extraction_*.json` | the JSON files |

---

## 9. Phase 2 — Fine-tune larger-context LLMs

Same datasets and settings as Phase 1 on two larger-context bases, evaluated on
full-circuit inputs over n = 2–19 (n = 20 circuits, about 145k tokens, exceed
LLaMA 3.1 8B's 128k window, so both models stop at 19) and oracle-only inputs
over n = 2–20.

```bash
bash "$RR/Phase 2/Llama-3-8B-Instruct-262k/run_phase2.sh"
bash "$RR/Phase 2/LLaMA 3.1 8B/run_phase2.sh"
```

Results in `$WORK/results/phase-2/Llama-3-8B-Instruct-262k/` and
`$WORK/results/phase-2/LLaMA-3.1-8B/`: CR/SRR (Figures 14, 15), SA/CF (16, 17),
OEA full-circuit vs oracle-only (18, 19) and RET (20 for the 262k model). For
LLaMA 3.1 8B the script also writes `ret_normalized_n3_paper.png` (Figure 21):
its n = 2 outputs run to the token limit, which inflates T(2), so RET is also
shown relative to T(3); `ret_paper.png` is Figure 22.

The full-circuit evaluation is the slow part: it took about 10 h for LLaMA 3.1 8B
and 30–40 h for the 262k model on our machine.

---

## 10. Phase 3 — Multi-agent system through prompt engineering

Phase 3 trains nothing. It runs a sequential three-agent pipeline (Oracle
Extractor → Marked-State Identifier → Probability Distribution, orchestrated
with LangChain) with few-shot prompts on four LLM variations: the untouched
Llama-3-8B-Instruct-262k and LLaMA 3.1 8B from section 6, and the two Phase 2
fine-tuned models. The prompts are in `Phase 3/prompt_templates/`.

To keep the cost of four variations manageable, Phase 3 uses one circuit per
(n, k) stratum for n = 2–19 and k = 1–3: 51 circuits (one at n = 2, two at
n = 3, three for every larger n), drawn deterministically from the full
manifest by `build_phase3_circuits.py`.

Requires Phase 2 to be finished (both merged models).

```bash
bash "$RR/Phase 3/run_phase3.sh"
```

For each variation, `run_eval.py` starts its own vLLM server, runs the chains
and stops the server, so **do not start a vLLM server yourself**. The script
then aggregates the per-agent metrics with Phase 4's post-processor
(`make_phase3_plots.py`) and draws the plots.

Results in `$WORK/results/phase-3/<variation>/`: SA/CF and the Agent 1, Agent 2
and Agent 3 accuracy plots (Figures 23–38), the result JSONs, and
`cot_agents.md` with every agent's output for every circuit.

---

## 11. Phase 4 — Multi-agent system fine-tuned on agent-specific data

The three agent datasets (section 5) rewrite each GroverGPT+ trace as three
input–output pairs, one per agent, with added reasoning steps. One model per
base is fine-tuned on all three jointly (same Table II settings, starting from
the stock base model, not the Phase 2 fine-tune); at inference the three agents
are that one model called with each agent's input.

```bash
# Llama-3-8B-Instruct-262k base
bash "$RR/Phase 4/run_phase4_Llama-3-8B-Instruct-262k.sh"       # train + merge (≈4 h)
bash "$RR/Phase 4/run_paper_eval_Llama-3-8B-Instruct-262k.sh"   # evaluation (≈1 day)

# LLaMA 3.1 8B base
bash "$RR/Phase 4/run_phase4_LLaMA-3.1-8B.sh"
bash "$RR/Phase 4/run_paper_eval_LLaMA-3.1-8B.sh"
```

The evaluation script builds the 4,024-circuit paper set and the RET set
(3 mixed-k circuits per n), runs the RET pass one chain at a time (so each is
timed on its own) and the bulk pass 8 chains at a time, post-processes
everything into Phase 1/2-style JSONs (SA, CF, RET and the per-agent
accuracies, where Agent 2 is only scored at an n where Agent 1 is correct on
every circuit, and Agent 3 likewise on Agent 2) and draws the plots. As in
Phase 3, `run_eval.py` manages its own vLLM server.

Results in `$WORK/results/phase-4/<model>/`: SA/CF, Agent 1–3 accuracy and RET
(Figures 39–48).

---

## 12. Phase 4 ablation — Monolithic model on the same data

To separate the effect of the more detailed traces from that of the
multi-agent decomposition, each base model is also trained as a single
monolithic model. Its training target is the three agent outputs joined end to
end (identical text and token count to the three agents combined), with the
bare QASM circuit as input; the settings are those of Phase 4, with
`cutoff_len` raised to 5000 so that no target is truncated. The evaluation uses
the same circuits, metrics and post-processor as Phase 4; each output is split
back into the three agent sections for the per-section scores.

```bash
bash "$RR/Phase 4/Monolithic Model/run_Llama-3-8B-Instruct-262k.sh" all   # or: train, then eval
bash "$RR/Phase 4/Monolithic Model/run_LLaMA-3.1-8B.sh" all
```

The `train` stage builds `Grover_Monolithic_2_7_MMS.json` in
`$WORK/LLaMA-Factory/data/` on first use (the build verifies that every example
has exactly the characters and token count of its three agent targets).
Results in `$WORK/results/phase-4/monolithic_<model>/`.

---

## 13. Serve the final Phase 4 model

The **`Prototype Interaction Application/`** folder contains an
OpenAI-compatible proxy that wraps the three-agent pipeline behind one chat
endpoint, and a web interface for it. With `WORK` set, its launcher finds the
merged Phase 4 model and `venv-inference` on its own:

```bash
bash "$WORK/Multi-agent-system-quantum-algorithm-analysis/Prototype Interaction Application/agent_proxy_backend/run_agent_proxy.sh"
```

See that folder's `README.md` for the interface and for connecting other chat
clients.

---

## 14. Notes on reproducibility

Checked by re-running this repository's code against our original artefacts:

- the three evaluation manifests, the Phase 3, Phase 4 and RET circuit sets,
  the three agent datasets and the monolithic dataset are identical to ours;
- the models prepared by `prepare_quantum_model.py` tokenize every input exactly
  like the ones our models were trained from (same token IDs and special
  tokens), with the same chat template and configs, and train and merge
  correctly in LLaMA-Factory;
- every training config loads in LLaMA-Factory with the expected train /
  validation split (2,703 / 301 for Phases 1–2, 2,840 / 316 for Phase 4,
  946 / 106 for the monolithic models);
- re-running the post-processing and plotting on our saved model outputs
  reproduces the published result JSONs and figures. A few figures differ only
  cosmetically (legend position), because the plotting code was tidied after
  those thesis figures were made;
- running the evaluation code on subsets with our own trained models reproduces
  our saved outputs: exactly for the Phase 4 multi-agent and monolithic chains
  run one at a time, and for almost every circuit in the batched Phase 1–3 runs.
  In batched runs vLLM occasionally picks a different token where two are
  nearly tied, so a few outputs differ between any two runs, even of the
  original code.

Fine-tuning and LLM inference on a different GPU, driver or library build are
not bit-for-bit deterministic either, so expect small numerical differences in
the retrained models' scores.

---

## 15. Acknowledgements

This work is built on top of the **GroverGPT+** research:

> M. Chen, *et al.*, "Symbolic analysis of Grover search algorithm via
> Chain-of-Thought reasoning and quantum-native tokenization," *npj Quantum
> Information*, vol. 12, no. 1, art. 48, 2026.
>
> Source code and dataset: <https://github.com/JimXiong16/GroverGPT-2>

We also acknowledge the open-source LLaMA-Factory, vLLM, LangChain, Qiskit,
and PyTorch projects, without which this work would not have been possible.
