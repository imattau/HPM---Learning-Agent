"""Extractive realization for source-backed claims."""
from __future__ import annotations

from hpm_ai_v2.chat.schema import Realization, RealizedSentence, ResponsePlan
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


class PlanBoundRealizer:
    """Render every planned claim verbatim and retain its citation mapping."""

    name = "plan-bound-extractive"

    def realize_with_map(self, plan: ResponsePlan) -> Realization:
        if plan.act == "clarify":
            return Realization(clarify_text(plan), ())
        if plan.act != "answer" or not plan.claims:
            return Realization("I couldn't find an answer in the provided material.", ())

        sentences = []
        references = []
        prior_source_ids: set[str] = set()
        prefixes = []
        for index, claim in enumerate(plan.claims, start=1):
            citation_ids = tuple(f"citation-{index}-{i}" for i, _ in enumerate(claim.citations, start=1))
            source_ids = {citation.source_id for citation in claim.citations}
            if index == 1:
                prefixes.append("The cited material states")
            elif source_ids & prior_source_ids:
                prefixes.append("The same source also states")
            else:
                prefixes.append("Another cited source states")
            prior_source_ids.update(source_ids)
            sentences.append(RealizedSentence(
                text=claim.text, claim_ids=(claim.claim_id,), citation_ids=citation_ids,
            ))
            references.append(f"[{index}] " + "; ".join(
                f"{citation.title or citation.source_id} ({citation.source_id}, "
                f"characters {citation.start}–{citation.end})"
                for citation in claim.citations
            ))
        rendered = [f'{prefixes[i - 1]}: “{sentence.text}” [{i}]'
                    for i, sentence in enumerate(sentences, start=1)]
        if plan.fmt in {"summarize", "outline"}:
            heading = "Summary" if plan.fmt == "summarize" else "Outline"
            if plan.format_style == "bullets":
                rendered = [f"• {line}" for line in rendered]
            else:
                rendered = [f"{i}. {line}" for i, line in enumerate(rendered, start=1)]
            body = heading + "\n" + "\n".join(rendered)
        elif plan.fmt == "compare":
            body = "Comparison from the provided material\n\n" + "\n\n".join(rendered)
        elif plan.fmt == "define":
            body = "Definition: " + " ".join(rendered)
        else:
            body = " ".join(rendered)
        return Realization(body + "\n\nSources:\n" + "\n".join(references), tuple(sentences))

    def realize(self, plan: ResponsePlan) -> str:
        return self.realize_with_map(plan).text
