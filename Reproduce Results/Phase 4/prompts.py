"""
Phase 4 minimal prompt builders.

Phase 4 is the fine-tuned counterpart to Phase 3's prompt-engineered
agents. The model has been trained on inputs that ARE EXACTLY what these
builders return — bare QASM for Agent 1, the marker-prefixed Oracle block
for Agent 2, the marker-prefixed marked-state list for Agent 3 — so no
prompt scaffolding is needed at inference. The task definition, output
format, reasoning structure and stopping behavior are all carried by the
trained weights.

This module shadows Phase 3's `prompts.py` when `Multi Agent System Phase
4/run_eval.py` puts its own directory ahead of Phase 3's on sys.path. The
shared orchestrator code in Phase 3 then imports these minimal builders
instead of the prompt-template-loading ones.

Each builder takes the same argument the Phase 3 builder took
(`A1Result.text` for Agent 2, `A2Result.text` for Agent 3) and returns
the raw string that the merged Phase 4 model was trained to consume.
"""


def build_agent1_prompt(qasm: str) -> str:
    """Agent 1 input — bare QASM, identical to the `input` field used at training."""
    return qasm.rstrip()


def build_agent2_prompt(parsed_a1_text: str) -> str:
    """Agent 2 input — `=== Extracted Oracle Gate Definition ===\\n<gate block>`.

    `parsed_a1_text` is already prefixed with that marker by Phase 3's
    `parse_agent1_output`, which is the same format the training data uses.
    """
    return parsed_a1_text.rstrip()


def build_agent3_prompt(parsed_a2_text: str) -> str:
    """Agent 3 input — `=== Final Marked States ===\\n<bitstrings>`.

    `parsed_a2_text` is already prefixed with that marker by Phase 3's
    `parse_agent2_output`, matching the training format.
    """
    return parsed_a2_text.rstrip()
