"""Initial rule-based dialogue act controller."""
from __future__ import annotations


class GroundedActController:
    """Answer when evidence is selected; otherwise make uncertainty explicit."""

    def choose(self, plan) -> str:
        return "abstain" if plan.act == "abstain" or not plan.claims else "answer"
