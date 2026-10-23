"""Fixed, evidence-preserving response schemas for initial study workflows.

These renderers are hand-authored templates, not learned/promoted L4 patterns.
Each factual back/claim remains an exact cited source sentence.
"""
from __future__ import annotations

from hpm_ai_v2.chat.schema import ResponsePlan


def render_schema(plan: ResponsePlan) -> str:
    claims = plan.claims
    if plan.act == "clarify":
        from hpm_ai_v2.chat.realizers import clarify_text
        return clarify_text(plan)
    if not claims:
        return "I couldn't find an answer in the provided material."
    if plan.fmt == "define":
        claim = claims[0]
        return f"Definition: {claim.text}\n\n{_source(claim)}"
    if plan.fmt in {"summarize", "outline"}:
        heading = "Summary" if plan.fmt == "summarize" else "Outline"
        points = [f"{i}. {claim.text}\n   {_source(claim)}" for i, claim in enumerate(claims, start=1)]
        return f"{heading}\n" + "\n".join(points)
    if plan.fmt == "flashcards":
        title = claims[0].citations[0].title or claims[0].citations[0].source_id
        cards = []
        for index, claim in enumerate(claims, start=1):
            cards.append(
                f"Card {index}\nFront: Recall one key point from {title}.\n"
                f"Back: {claim.text}\n{_source(claim)}"
            )
        return "Study cards\n" + "\n\n".join(cards)
    claim = claims[0]
    return f"{claim.text}\n\n{_source(claim)}"


def _source(claim) -> str:
    citation = claim.citations[0]
    label = citation.title or citation.source_id
    return (
        f"Source: {label} ({citation.source_id}, "
        f"characters {citation.start}–{citation.end})."
    )
