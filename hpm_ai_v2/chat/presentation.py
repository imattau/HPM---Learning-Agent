"""Turn a ResponsePlan into structured display blocks for a chat frontend.

Pure presentation: block text is exactly claim text plus fixed labels. Nothing
here adds a fact, and a claim with no citation cannot be rendered.
"""
from __future__ import annotations

from hpm_ai_v2.chat.schema import ResponsePlan

ABSTAIN_TEXT = "I couldn't find an answer in the provided material."
CLARIFY_TEXT = "Your question matches more than one source. Which do you mean?"


class UngroundedClaim(ValueError):
    """Raised when a claim reaches the presentation layer without a citation."""


def to_blocks(plan: ResponsePlan) -> tuple[list[dict], list[dict]]:
    """Return ``(blocks, citations)``; blocks reference citations by number."""
    if plan.act == "clarify":
        options = [{"source_id": sid, "label": label} for sid, label in plan.options]
        return [{"type": "clarify", "text": CLARIFY_TEXT, "options": options}], []
    if plan.act != "answer" or not plan.claims:
        return [{"type": "abstain", "text": ABSTAIN_TEXT, "reason": plan.reason}], []

    numbers: dict[tuple[str, int, int], int] = {}
    citations: list[dict] = []

    def item(claim) -> dict:
        if not claim.citations:
            raise UngroundedClaim(f"claim {claim.claim_id} has no citation")
        cites = []
        for citation in claim.citations:
            key = (citation.source_id, citation.start, citation.end)
            if key not in numbers:
                numbers[key] = len(citations) + 1
                citations.append({
                    "n": numbers[key],
                    "source_id": citation.source_id,
                    "title": citation.title or citation.source_id,
                    "start": citation.start,
                    "end": citation.end,
                    "text": citation.text,
                })
            cites.append(numbers[key])
        return {"claim_id": claim.claim_id, "text": claim.text, "cites": cites}

    items = [item(claim) for claim in plan.claims]
    if plan.fmt == "define":
        blocks = [{"type": "paragraph", "title": "Definition", "items": items[:1]}]
    elif plan.fmt in {"summarize", "outline"}:
        kind = "bullet_list" if plan.format_style == "bullets" else "numbered_list"
        title = "Summary" if plan.fmt == "summarize" else "Outline"
        blocks = [{"type": kind, "title": title, "items": items}]
    elif plan.fmt == "flashcards":
        label = plan.claims[0].citations[0].title or plan.claims[0].citations[0].source_id
        blocks = [{"type": "flashcards", "title": "Study cards", "items": [
            {"front": f"Recall one key point from {label}.", **entry} for entry in items
        ]}]
    elif plan.fmt == "compare":
        blocks = [{"type": "compare", "title": "Comparison from the provided material", "items": [
            {"label": citations[entry["cites"][0] - 1]["title"], **entry} for entry in items
        ]}]
    else:
        blocks = [{"type": "paragraph", "items": items[:1]}]
    return blocks, citations
