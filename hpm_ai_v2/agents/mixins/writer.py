"""WriterMixin: adds natural language generation capabilities to HFN agents."""
from __future__ import annotations
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
import numpy as np
from hfn.hfn import HFN

if TYPE_CHECKING:
    from hpm_ai_v2.agents.reader_agent import ReaderAgent

class WriterMixin:
    """
    Mixin for HFN agents that adds text generation capabilities.
    Uses ReaderAgent's understanding to produce natural language.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # reader_agent is expected to be provided to the main Agent class

    def generate_sentence(self, predicate: str, agent: str, patient: str) -> str:
        """
        Generate a natural language sentence from SRL roles.
        Uses POS heuristics for better grammar.
        """
        v = predicate.lower()
        
        # Simple conjugation for 3rd person singular
        if v not in ["is", "are", "was", "were", "has", "have", "had", "can", "will"]:
            # Check if already pluralized
            if not v.endswith(('s', 'ies')):
                if v.endswith(('x', 'ch', 'sh')):
                    v += "es"
                elif v.endswith('y') and len(v) > 1 and v[-2] not in 'aeiou':
                    v = v[:-1] + "ies"
                else:
                    v += "s"
                
        # Handle 'the' in roles if they already contain it
        def add_article(role):
            r = role.lower()
            if any(r.startswith(art + " ") for art in ["the", "a", "an"]):
                return role
            return f"the {role}"

        a = add_article(agent)
        p = add_article(patient)
        
        # Capitalize first letter
        sentence = f"{a} {v} {p}."
        return sentence[0].upper() + sentence[1:]

    def answer_natural(self, question: str) -> str:
        """
        Answer a question with a full natural language sentence.
        Uses ReaderAgent's hierarchical answering logic.
        """
        if not hasattr(self, "reader_agent") or self.reader_agent is None:
            return "I don't have a reader agent to help me understand."
            
        answer_phrase = self.reader_agent.answer_question_hierarchical(question)
        if not answer_phrase:
            # Check if we can request knowledge
            return "I don't know the answer yet."
            
        # Wrap in a natural sentence
        return f"The answer is {answer_phrase}."

    def generate_summary(self, doc_node: HFN, max_sentences: int = 3) -> str:
        """
        Extract salient sentences from a document node to form a summary.
        """
        sentences = []
        for para in doc_node.children():
            if getattr(para, "relation_type", None) == "paragraph":
                for sent_node in para.children():
                    if getattr(sent_node, "relation_type", None) == "sentence":
                        score = self._sentence_concept_score(sent_node)
                        sentences.append((score, sent_node))
                        
        # Sort by score descending
        sentences.sort(key=lambda x: x[0], reverse=True)
        
        # Take top-k and render
        summary_nodes = [sn for _, sn in sentences[:max_sentences]]
        # We render using ReaderAgent's renderer if available
        renderer = getattr(self, "renderer", None)
        if renderer:
            return " ".join([renderer.render(sn) for sn in summary_nodes])
        return " ".join([sn.id for sn in summary_nodes])

    def _sentence_concept_score(self, sent_node: HFN) -> float:
        """Score sentence importance based on its concept density (mu)."""
        # Sentences with higher L2/L3 weights or more central concepts score higher
        concept_slice = sent_node.mu[self.config.S_DIM: self.config.S_DIM + self.config.DIM]
        return float(np.sum(concept_slice))
