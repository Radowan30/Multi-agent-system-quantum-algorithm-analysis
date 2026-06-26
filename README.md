# LLM Based AI Agents for Improving Quantum Algorithm Simulation

Companion repository for the Final Year Project thesis of the same name. It
contains the full set of result artifacts, the code and configuration needed
to reproduce every phase of the research end-to-end, and a working chat-style
prototype interface for the final multi-agent system.

This work builds on the **GroverGPT+** research:

> M. Chen, *et al.*, "Symbolic analysis of Grover search algorithm via
> Chain-of-Thought reasoning and quantum-native tokenization,"
> *npj Quantum Information*, vol. 12, no. 1, art. 48, 2026.
> Original code + dataset: <https://github.com/JimXiong16/GroverGPT-2>

## Repository contents

```
.
├── Our Results/                       Result figures and underlying data files
│                                      for all four research phases.
├── Reproduce Results/                 Code, configs and step-by-step instructions
│                                      to recreate every phase from scratch.
└── Prototype Interaction Application/ A local chat-style frontend that talks to
                                       the fine-tuned Phase 4 multi-agent system.
```

Each folder has its own `README.md` with detailed instructions.

## Quick links

- 📊 **Browse the results:** [`Our Results/README.md`](./Our%20Results/README.md)
- 🔁 **Reproduce everything from scratch:** [`Reproduce Results/README.md`](./Reproduce%20Results/README.md)
- 💬 **Run the prototype locally:** [`Prototype Interaction Application/README.md`](./Prototype%20Interaction%20Application/README.md)

## What the four phases are

1. **Phase 1 — Baseline.** Recreate the `GroverGPT+` model by fine-tuning LLaMA 3 8B on the GroverGPT+ dataset.
2. **Phase 2 — Larger context.** Fine-tune two larger-context LLMs (Llama-3-8B-Instruct-262k and LLaMA 3.1 8B) with the same recipe so circuits up to n = 19 can fit.
3. **Phase 3 — Multi-agent (prompts only).** Split the analysis across three specialised agents (Oracle Extractor → Marked-State Identifier → Probability Distribution) via prompt engineering, evaluated on four LLM variations.
4. **Phase 4 — Multi-agent (fine-tuned).** Fine-tune each agent on its own sub-task using a programmatically augmented dataset.

The headline result is in **Phase 4**: in-distribution Search Accuracy reaches
1.0 with near-zero variance for both base models. The out-of-distribution
performance still degrades — the multi-agent system does not, on its own,
solve out-of-distribution generalisation. The per-agent reasoning traces are
also richer and easier for a human reader to follow than the monolithic
baseline.

## Acknowledgements

The research dataset and the original GroverGPT+ training recipe were openly
released by the GroverGPT+ authors at the GitHub link above — thank you. The
project also relies on open-source LLaMA-Factory, vLLM, LangChain, Qiskit,
React + Vite, FastAPI, and PyTorch.

## Licence

Code in this repository is released under the MIT licence; see `LICENSE`.
Datasets and model weights remain under their respective original licences
(Meta LLaMA community licence, GroverGPT+ dataset licence, etc.).
