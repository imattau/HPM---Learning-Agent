"""Initial rule-based dialogue act controller."""
from __future__ import annotations


class GroundedActController:
    """Answer when evidence is selected; otherwise make uncertainty explicit."""

    def choose(self, plan) -> str:
        if plan.act == "clarify" and len(plan.options) >= 2:
            return "clarify"
        return "abstain" if plan.act != "answer" or not plan.claims else "answer"
