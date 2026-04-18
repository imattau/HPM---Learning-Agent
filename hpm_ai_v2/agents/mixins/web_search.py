"""WebSearchMixin: adds autonomous web searching to agents."""
from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import numpy as np
from hfn.hfn import HFN

class WebSearchMixin:
    """
    Mixin for HFN agents that adds web search capabilities.
    Represents search queries and results as HFN nodes.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def search_web(self, query: str) -> HFN:
        """
        Perform a web search and return a search_query HFN node.
        The node's children are the resulting webpage nodes.
        """
        print(f"      [WEB] Searching: {query}")
        
        # 1. Create search_query node
        mu = self.config.encode_search_query(query)
        import uuid
        query_id = f"search_query_{uuid.uuid4().hex[:8]}"
        query_node = HFN(
            mu=mu,
            sigma=np.ones(self.m_dim) * 0.1,
            id=query_id,
            use_diag=True
        )
        query_node.metadata = {"type": "search_query", "query": query}
        query_node.relation_type = "search"
        
        # 2. Generate mock results (webpage nodes)
        # In a real system, this would call a Search API
        mock_urls = [
            f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
            f"https://www.britannica.com/topic/{query.replace(' ', '-')}"
        ]
        
        for url in mock_urls:
            # Create a webpage node (stub)
            target_mu = self.config.encode_url(url)
            result_id = f"webpage_{uuid.uuid4().hex[:8]}"
            result_node = HFN(mu=target_mu, sigma=np.ones(self.m_dim)*0.5, id=result_id, use_diag=True)
            result_node.metadata = {"type": "webpage", "url": url, "status": "unfetched"}
            
            # Add WEB_SEARCH_RESULT concept
            idx = self.config.concept_idx.get("WEB_SEARCH_RESULT")
            if idx is not None:
                result_node.mu[self.config.S_DIM + idx] = 1.0
                
            self.observer.register(result_node, protected=False)
            self.patterns[result_id] = result_node
            
            # Link to query node
            query_node.add_child(result_node)
            
        self.observer.register(query_node, protected=False)
        self.patterns[query_id] = query_node
        
        return query_node
