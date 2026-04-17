"""Renders an HFN text passage node back to its original string."""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from hfn.hfn import HFN
from hpm_ai_v2.domains.text_domain import TextDomainConfig


class TextRenderer:
    """Converts HFN passage node → original passage text string."""

    def __init__(self, config: TextDomainConfig) -> None:
        self.config = config

    def render(self, node: "HFN") -> str:
        metadata = getattr(node, "metadata", {})
        idx = metadata.get("passage_idx")
        if idx is not None and 0 <= idx < len(self.config._passages):
            return self.config.get_passage(idx)
        concept_slice = node.mu[self.config.S_DIM: self.config.S_DIM + self.config.DIM]
        top_indices = concept_slice.argsort()[::-1][:10]
        words = [self.config.concepts[i] for i in top_indices if concept_slice[i] > 0]
        return " ".join(words)
