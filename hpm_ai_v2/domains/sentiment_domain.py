"""
SentimentDomainConfig — manifold structure for emotion and opinion analysis.
"""
from __future__ import annotations
from typing import List, Optional
from hpm_ai_v2.domains.base import DomainConfig

class SentimentDomainConfig(DomainConfig):
    """Manifold structure for sentiment and emotion."""
    def __init__(self, concepts: Optional[List[str]] = None, s_dim: int = 20):
        if concepts is None:
            concepts = [
                "LEXICON_SCORE",      # L1: Look up a word's sentiment score
                "IS_NEGATION",        # L1: Detect negation (not, never)
                "IS_INTENSIFIER",     # L1: Detect intensifiers (very, extremely)
                "SENTENCE_SCORE",      # L1: Aggregate word scores
                "NORMALIZE",           # L1: Clamp result to [-1, 1]
                "CLASSIFY",            # L1: Map to pos/neg/neutral
                "EXPLAIN",             # L1: Return contributing tokens
                "OP_SENTIMENT",        # L2: Sentiment operation
                "STATE_SENTIMENT",     # L3: Sentiment state
                "EMOTION_JOY", "EMOTION_ANGER", "EMOTION_SADNESS",
                "SENTIMENT_POS", "SENTIMENT_NEG", "SENTIMENT_NEUTRAL"
            ]
        super().__init__(concepts, s_dim=s_dim)

    @property
    def domain_type(self) -> str:
        return "sentiment"
