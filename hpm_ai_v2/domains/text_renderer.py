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
        relation_type = getattr(node, "relation_type", None)
        
        # 1. Base case: Passage with raw text (backward compatibility)
        idx = metadata.get("passage_idx")
        if idx is not None and 0 <= idx < len(self.config._passages):
            return self.config.get_passage(idx)
            
        # 2. Primitive Rendering (no children)
        if relation_type == "character":
            return metadata.get("char", node.id.replace("CHAR_DIGIT_", "").replace("CHAR_", ""))
        elif relation_type == "pos_macro":
            return f"[{metadata.get('tag', 'POS')}]"
            
        # 3. Fractal Rendering: Recursive reconstruction from children
        children = node.children()
        
        if relation_type == "webpage":
            return f"[Webpage: {metadata.get('url', 'Unknown URL')}]"
        elif relation_type == "search":
            return f"[Search Query: {metadata.get('query', 'Unknown')}]"
            
        if children:
            if relation_type == "sentence":
                # Render word nodes
                words = [self.render(c) for c in children]
                return " ".join(words).replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" !", "!")
            elif relation_type in ["paragraph", "document", "topic"]:
                # Render sentence or paragraph nodes
                parts = [self.render(c) for c in children]
                return " ".join(parts)
            elif relation_type == "spelling":
                # Render character nodes for a word
                chars = [self.render(c) for c in children]
                return "".join(chars)
            elif relation_type == "hyperlink":
                # Render hyperlink source -> target
                if len(children) >= 2:
                    return f"[Hyperlink: {self.render(children[0])} -> {self.render(children[1])}]"
                return "[Hyperlink]"

        # 4. Fallback: Top words from concept vector (mu)
        concept_slice = node.mu[self.config.S_DIM: self.config.S_DIM + self.config.DIM]
        top_indices = concept_slice.argsort()[::-1][:10]
        words = [self.config.concepts[i] for i in top_indices if concept_slice[i] > 1e-4]
        return " ".join(words)
