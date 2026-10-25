"""Exact source-span verifier for extractive responses."""
from __future__ import annotations


class GroundingVerifier:
    def __init__(self, sources):
        self.sources = sources

    def verify(self, plan) -> bool:
        if plan.act != "answer":
            return not plan.claims
        if not plan.claims:
            return False
        for claim in plan.claims:
            if claim.kind != "source" or not claim.citations:
                return False
            for citation in claim.citations:
                source = self.sources.get(citation.source_id)
                if source is None or not (0 <= citation.start <= citation.end <= len(source.text)):
                    return False
                if source.text[citation.start:citation.end] != citation.text:
                    return False
                if citation.text != claim.text:
                    return False
        return True

    def verify_realization(self, plan, realization) -> bool:
        """Require every factual output sentence to map to an exact planned claim."""
        if not self.verify(plan):
            return False
        if plan.act != "answer":
            return not realization.sentences
        if len(realization.sentences) != len(plan.claims):
            return False

        claims = {claim.claim_id: claim for claim in plan.claims}
        seen = set()
        for index, sentence in enumerate(realization.sentences, start=1):
            if len(sentence.claim_ids) != 1:
                return False
            claim_id = sentence.claim_ids[0]
            if claim_id != plan.claims[index - 1].claim_id:
                return False
            claim = claims.get(claim_id)
            if claim is None or claim_id in seen or sentence.text != claim.text:
                return False
            seen.add(claim_id)
            expected_ids = {f"citation-{index}-{i}" for i in range(1, len(claim.citations) + 1)}
            if not sentence.citation_ids or set(sentence.citation_ids) != expected_ids:
                return False
            rendered_forms = (
                f'{sentence.text} [{index}]',
                f'“{sentence.text}” [{index}]',
            )
            if not any(rendered in realization.text for rendered in rendered_forms):
                return False
        return seen == set(claims)
