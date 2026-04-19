"""NLPAuditor: Utility for measuring linguistic quality and coherence of HPM-generated text."""
from __future__ import annotations
import re
from collections import Counter
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from hpm_ai_v2.agents.reader_agent import ReaderAgent


class NLPAuditor:
    """
    Evaluates the linguistic "Aesthetics" and structural coherence of text.
    Used as a Critic to select high-quality generation candidates.
    """
    def __init__(self, reader_agent: Optional[ReaderAgent] = None):
        self.reader_agent = reader_agent

    def score_text(self, text: str) -> Dict[str, float]:
        """
        Calculate a comprehensive set of linguistic quality scores.
        """
        if not text.strip():
            return {"total_utility": 0.0}

        scores = {
            "aesthetics": self._score_aesthetics(text),
            "diversity": self._score_diversity(text),
            "coherence": self._score_coherence(text) if self.reader_agent else 0.5
        }
        
        # Unified Utility = (Aesthetics + Diversity + Coherence) / 3
        scores["total_utility"] = (scores["aesthetics"] + scores["diversity"] + scores["coherence"]) / 3.0
        return scores

    def _score_aesthetics(self, text: str) -> float:
        """
        Scores casing, punctuation, and basic grammar heuristics.
        Range: [0, 1]
        """
        points = 0
        total_possible = 4

        # 1. Capitalization (starts with capital)
        if text[0].isupper():
            points += 1
            
        # 2. Terminal Punctuation (ends with . ? !)
        if text.strip()[-1] in ".?!":
            points += 1
            
        # 3. Punctuation Density (not too little, not too much)
        # Ideal: 1 punctuation mark per 5-15 words
        words = text.split()
        punct_count = len(re.findall(r"[.,!?;:]", text))
        if words:
            ratio = punct_count / len(words)
            if 0.05 <= ratio <= 0.25:
                points += 1
                
        # 4. No excessive whitespace
        if "  " not in text:
            points += 1

        return points / total_possible

    def _score_diversity(self, text: str) -> float:
        """
        Lexical Diversity via Type-Token Ratio (TTR).
        Prevents repetitive "looping" generation.
        Range: [0, 1]
        """
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return 0.0
            
        num_types = len(set(tokens))
        num_tokens = len(tokens)
        
        # TTR = Types / Tokens
        ttr = num_types / num_tokens
        
        # Penalize very short sentences too (harder to be diverse)
        if num_tokens < 5:
            ttr *= 0.8
            
        return min(1.0, ttr)

    def _score_coherence(self, text: str) -> float:
        """
        Uses the ReaderAgent's Affective Utility (Surprise) as a critic.
        Higher surprise for a known domain = lower coherence.
        Range: [0, 1]
        """
        if not self.reader_agent:
            return 0.5
            
        # We don't want the critic to actually LEARN (mutate its weights),
        # so we use a read-only observation or a temporary clone.
        # For now, we measure how well the text is explained by existing nodes.
        
        # 1. Tokenize and encode
        from hpm_ai_v2.domains.text_domain import tokenise
        tokens = tokenise(text)
        if not tokens:
            return 0.0
            
        # 2. Measure "Familiarity" (Max overlap with any existing node)
        mu = self.reader_agent.config.encode_passage(text)
        candidates = self.reader_agent.forest.retrieve(mu, k=1)
        
        if not candidates:
            return 0.1 # Completely unfamiliar
            
        best_node = candidates[0]
        dist = np.linalg.norm(best_node.mu - mu)
        
        # Sigmoid-style mapping of distance to coherence
        # dist ~ 0 (identical) -> coherence = 1.0
        # dist ~ 2.0 (different) -> coherence = 0.0
        coherence = float(np.exp(-dist))
        
        return min(1.0, max(0.0, coherence))
