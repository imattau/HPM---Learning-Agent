"""ReaderAgent: observes text passages and retrieves relevant ones for queries."""
from __future__ import annotations
from typing import Optional, List
import numpy as np
from hfn.hfn import HFN
from hpm_ai_v2.agents.base_agent import BaseHFNAgent
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.domains.text_renderer import TextRenderer
from hpm_ai_v2.utils.oracle.text_oracle import TextOracle
from hpm_ai_v2.utils.text_fetcher import fetch_passages


class ReaderAgent(BaseHFNAgent):
    """
    HFN-native agent that reads text/webpages and retrieves relevant passages.

    Usage:
        agent = ReaderAgent(config)
        agent.observe_url("https://example.com")
        result = agent.query("what is machine learning?")
    """

    def __init__(self, config: TextDomainConfig, **kwargs) -> None:
        renderer = TextRenderer(config)
        super().__init__(config, renderer=renderer, **kwargs)
        self.oracle = TextOracle(config)
        self.counting_oracle.wrapped = self.oracle

    def observe_passage(self, text: str) -> None:
        """Encode a single passage as an HFN node and register in forest."""
        idx = self.config.register_passage(text)
        mu = self.config.encode_passage(text)
        sigma = np.ones(self.config.m_dim) * 0.1
        node = HFN(mu=mu, sigma=sigma, id=f"passage_{idx}", use_diag=True)
        node.metadata = {"passage_idx": idx, "text": text}
        self.observer.register(node, protected=False, initial_weight=1.0)
        self.patterns[f"passage_{idx}"] = node

    def observe_document(self, text: str, min_length: int = 40) -> int:
        """Split text into passages and observe each. Returns passage count."""
        passages = fetch_passages(text=text, min_length=min_length)
        for p in passages:
            self.observe_passage(p)
        return len(passages)

    def observe_url(self, url: str, min_length: int = 40) -> int:
        """Fetch URL, parse into passages, observe each. Returns passage count."""
        passages = fetch_passages(url=url, min_length=min_length)
        for p in passages:
            self.observe_passage(p)
        return len(passages)

    def query(self, question: str, top_k: int = 1) -> Optional[str]:
        """Retrieve the most relevant passage for the given query string."""
        if not self.config._passages:
            return None
        query_mu = self.config.encode_passage(question)
        query_node = HFN(mu=query_mu, sigma=np.ones(self.config.m_dim), use_diag=True)
        candidates = self.retriever.retrieve(query_node, k=top_k)
        if not candidates:
            return None
        return self.renderer.render(candidates[0])

    def query_top_k(self, question: str, k: int = 3) -> List[str]:
        """Retrieve top-k most relevant passages for the query."""
        if not self.config._passages:
            return []
        query_mu = self.config.encode_passage(question)
        query_node = HFN(mu=query_mu, sigma=np.ones(self.config.m_dim), use_diag=True)
        candidates = self.retriever.retrieve(query_node, k=k)
        return [self.renderer.render(c) for c in candidates]
