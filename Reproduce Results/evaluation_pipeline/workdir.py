"""
Locations outside the repository (data, generated circuits, models, results).

Everything derives from the WORK environment variable, the working directory
set up in Reproduce Results/README.md (section 2). The layout under WORK is:

    $WORK/data_MMS/                  raw GroverGPT+ QASM circuits (grover_n2/, grover_n3/, ...)
    $WORK/eval_circuits/manifests/   evaluation circuit manifests (generate_eval_circuits.py)
    $WORK/eval_circuits/data_MMS_eval/  freshly generated circuits ('strict' source only)
    $WORK/models/                    base models
    $WORK/saves/                     LoRA adapters and merged models
    $WORK/results/                   evaluation outputs
"""

import os


def work_dir() -> str:
    work = os.environ.get("WORK")
    if not work:
        raise RuntimeError(
            "The WORK environment variable is not set. Set it to your working "
            "directory (see Reproduce Results/README.md, section 2), e.g.\n"
            "    export WORK=$HOME/grover-multiagent-reproduction"
        )
    return work


def data_mms_dir() -> str:
    return os.path.join(work_dir(), "data_MMS")


def manifests_dir() -> str:
    return os.path.join(work_dir(), "eval_circuits", "manifests")


def data_mms_eval_dir() -> str:
    return os.path.join(work_dir(), "eval_circuits", "data_MMS_eval")


def models_dir() -> str:
    return os.path.join(work_dir(), "models")


def saves_dir() -> str:
    return os.path.join(work_dir(), "saves")


def results_dir() -> str:
    return os.path.join(work_dir(), "results")
