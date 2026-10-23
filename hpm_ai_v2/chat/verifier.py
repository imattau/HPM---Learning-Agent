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
