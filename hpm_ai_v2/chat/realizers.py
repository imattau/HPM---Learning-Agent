"""Extractive realization for source-backed claims."""
from __future__ import annotations

from hpm_ai_v2.chat.schema import ResponsePlan
from hpm_ai_v2.chat.procedures import render_schema


class ExtractiveRealizer:
    name = "extractive"

    def realize(self, plan: ResponsePlan) -> str:
        if plan.act != "answer" or not plan.claims:
            return "I couldn't find an answer in the provided material."
        claim = plan.claims[0]
        citation = claim.citations[0]
        return (
            f"{claim.text}\n\n"
            f"Source: {citation.title or citation.source_id} "
            f"({citation.source_id}, characters {citation.start}–{citation.end})."
        )


class SchemaRealizer:
    """Format cited extractive claims as definitions, summaries, outlines or cards."""

    name = "schema-extractive"

    def realize(self, plan: ResponsePlan) -> str:
        return render_schema(plan)
