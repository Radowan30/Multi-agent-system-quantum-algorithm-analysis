# Our Results

This folder contains the full set of result figures and underlying numerical
data files for the four research phases of the thesis *"LLM Based AI Agents
for Improving Quantum Algorithm Simulation"*. Every figure here is one that is
referenced and discussed in the thesis (Figure 9 – Figure 48), and every
result JSON is the raw output of the evaluation that produced the corresponding
figures.

## Folder layout

```
Our Results/
├── Phase 1/                                 — Recreated GroverGPT+ baseline
├── Phase 2/
│   ├── Llama-3-8B-Instruct-262k/            — Larger-context fine-tuned model
│   └── LLaMA 3.1 8B/                        — Larger-context fine-tuned model
├── Phase 3/                                 — Multi-agent system (prompt-engineered)
│   ├── Untrained Llama-3-8B-Instruct-262k/
│   ├── Untrained LLaMA 3.1 8B/
│   ├── Phase 2 Fine-Tuned Llama-3-8B-Instruct-262k/
│   └── Phase 2 Fine-Tuned LLaMA 3.1 8B/
└── Phase 4/                                 — Multi-agent system (fine-tuned per agent)
    ├── Llama-3-8B-Instruct-262k/
    └── LLaMA 3.1 8B/
```

## File-naming convention

- Each figure file is named with the figure number used in the thesis followed
  by a short description, e.g. `Figure 10 - SA and CF on full-circuit inputs.png`.
- Numerical results sit alongside the figures as JSON, with shorter names
  (`results_full.json`, `oracle_extraction.json`, `marked_state_accuracy.json`,
  `agent3_accuracy.json`). These are the exact files emitted by the evaluation
  pipeline.

## Metrics (summary)

| Metric | Where it lives in the JSONs |
|---|---|
| Search Accuracy (SA), Classical Fidelity (CF), and per-n parse-failure counts | `results_full.json` and `results_oracle.json` (per-n means + std) |
| Oracle Extraction Accuracy (OEA) for Agent 1 / monolithic models | `oracle_extraction.json` |
| Agent 2 Accuracy — marked-state identification | `marked_state_accuracy.json` (Phase 3 / Phase 4 only) |
| Agent 3 Accuracy — probability distribution | `agent3_accuracy.json` (Phase 3 / Phase 4 only) |
| Relative Execution Time (RET) | `ret` and `ret_std` keys inside `results_full.json` (Phase 1, 2, 4) |
| Compression Ratio (CR) / Sequence Reduction Ratio (SRR) | `cr` / `srr` keys inside `results_full.json` (Phase 1, 2) |

To recreate any of these results from scratch, see the **`Reproduce Results/`**
folder in the repository root.
