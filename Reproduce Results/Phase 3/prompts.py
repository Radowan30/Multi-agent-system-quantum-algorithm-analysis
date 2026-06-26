"""
Loads the three agent prompt templates and builds the final prompt string
sent to each agent.

Each `build_agentN_prompt(...)` takes whatever the previous parser returned
as `.text` (i.e. marker-prefixed text) and slots it into the template's
`{USER_INPUT}` placeholder. The few-shot examples in each template show
input in the same marker-prefixed format, so the model sees consistent
structure across the few-shot examples and the actual query.

Templates live as markdown files in `prompt_templates/`. The Agent 1 template
contains the placeholder `{EXAMPLE_N7_K3_QASM}` which is pre-substituted once
with the full n=7 t=3 QASM from `prompt_templates/example_n7_k3_full_qasm.txt`.

Public API:
    build_agent1_prompt(qasm: str) -> str
    build_agent2_prompt(parsed_a1_text: str) -> str
    build_agent3_prompt(parsed_a2_text: str) -> str
"""

from pathlib import Path

_TEMPLATES_DIR = Path(__file__).parent / "prompt_templates"


def _load_template(name: str) -> str:
    return (_TEMPLATES_DIR / name).read_text(encoding="utf-8")


_AGENT1_TEMPLATE = _load_template("agent1_oracle_extractor.md")
_AGENT2_TEMPLATE = _load_template("agent2_marked_state_identifier.md")
_AGENT3_TEMPLATE = _load_template("agent3_probability_distribution.md")
_EXAMPLE_N7_QASM = _load_template("example_n7_k3_full_qasm.txt").rstrip()

# Pre-substitute the static n=7 QASM into Agent 1's template once.
_AGENT1_TEMPLATE = _AGENT1_TEMPLATE.replace("{EXAMPLE_N7_K3_QASM}", _EXAMPLE_N7_QASM)


def build_agent1_prompt(qasm: str) -> str:
    """Build the Agent 1 prompt. Input is raw QASM — no marker prefix."""
    return _AGENT1_TEMPLATE.replace("{USER_INPUT}", qasm.rstrip())


def build_agent2_prompt(parsed_a1_text: str) -> str:
    """Build the Agent 2 prompt.

    `parsed_a1_text` is `A1Result.text` — already prefixed with
    `=== Extracted Oracle Gate Definition ===`.
    """
    return _AGENT2_TEMPLATE.replace("{USER_INPUT}", parsed_a1_text.rstrip())


def build_agent3_prompt(parsed_a2_text: str) -> str:
    """Build the Agent 3 prompt.

    `parsed_a2_text` is `A2Result.text` — already prefixed with
    `=== Final Marked States ===`.
    """
    return _AGENT3_TEMPLATE.replace("{USER_INPUT}", parsed_a2_text.rstrip())
