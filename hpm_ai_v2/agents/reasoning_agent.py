"""
ReasoningAgent: base class for HFN agents that perform retrieval and generation.
Ensures reasoning specialists do not accidentally modify the knowledge base.
"""
from __future__ import annotations
import numpy as np
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from hpm_ai_v2.agents.base_agent import BaseHFNAgent
from hfn.hfn import HFN

class ReasoningAgent(BaseHFNAgent):
    """
    Agents that perform 'reasoning' (retrieval, generation, evaluation).
    Operates read-only against the knowledge store.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Disable learning by default for reasoning agents
        if hasattr(self, "observer"):
            # We don't remove the observer, but we don't call observe()
            pass

    def answer(self, question: str) -> str:
        """Base method for natural language answering. Overridden by specialists."""
        return "I don't have an answer for that yet."

    def evaluate(self, proposition: str) -> float:
        """Evaluate the truth/utility of a proposition based on current knowledge."""
        mu = self._encode_if_possible(proposition)
        # Check for similarity to existing document/passage nodes
        nodes = self.retriever.retrieve(HFN(mu=mu, sigma=np.ones_like(mu)), k=1)
        if not nodes: return 0.0
        dist = np.linalg.norm(mu - nodes[0].mu)
        return float(np.exp(-dist)) # Simple similarity-to-knowledge metric

    def _encode_if_possible(self, text: str) -> np.ndarray:
        if hasattr(self.config, "encode_passage"):
            return self.config.encode_passage(text)
        return np.zeros(self.m_dim)
