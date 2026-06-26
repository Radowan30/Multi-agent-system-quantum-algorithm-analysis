"""
Per-circuit ChainResult dataclass + shared helpers used by both orchestrators
(vanilla and LangChain). The orchestrators express the A1→A2→A3 pipeline
differently — vanilla as raw asyncio, LangChain as composed Runnables — but
both populate the same ChainResult shape so downstream code (metrics, cot
logger, results JSON) doesn't care which orchestrator produced the result.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ChainResult:
    """Per-circuit result of running A1→A2→A3."""

    success: bool
    failure_stage: Optional[str] = None  # "A1" | "A2" | "A3" | None

    # Raw model outputs
    a1_output: Optional[str] = None
    a2_output: Optional[str] = None
    a3_output: Optional[str] = None

    # Parsed structured payloads (None until that agent succeeds)
    extracted_oracle: Optional[str] = None
    marked_states: Optional[List[str]] = None
    probability_dict: Optional[Dict[str, float]] = None

    # Per-agent wall-clock timings (seconds)
    a1_time_s: float = 0.0
    a2_time_s: float = 0.0
    a3_time_s: float = 0.0

    # Total chain wall-clock from circuit input received to final dict produced.
    # This is the per-circuit RET measurement for the Phase 3 pipeline.
    total_time_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "failure_stage": self.failure_stage,
            "a1_output": self.a1_output,
            "a2_output": self.a2_output,
            "a3_output": self.a3_output,
            "extracted_oracle": self.extracted_oracle,
            "marked_states": self.marked_states,
            "probability_dict": self.probability_dict,
            "a1_time_s": self.a1_time_s,
            "a2_time_s": self.a2_time_s,
            "a3_time_s": self.a3_time_s,
            "total_time_s": self.total_time_s,
        }
