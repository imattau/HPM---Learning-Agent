"""Extractive realization for source-backed claims."""
from __future__ import annotations

from hpm_ai_v2.chat.schema import ResponsePlan
from hpm_ai_v2.chat.procedures import render_schema


def clarify_text(plan: ResponsePlan) -> str:
    labels = "; ".join(f"{i}. {label}" for i, (_sid, label) in enumerate(plan.options, start=1))
    return f"Your question matches more than one source. Which do you mean? {labels}"


class ExtractiveRealizer:
    name = "extractive"

    def realize(self, plan: ResponsePlan) -> str:
        if plan.act == "clarify":
            return clarify_text(plan)
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
