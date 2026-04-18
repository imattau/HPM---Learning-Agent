"""Web domain: defines primitives and encoding for web-scale resources."""
from __future__ import annotations
import numpy as np
from typing import List, Dict
from hpm_ai_v2.domains.base import DomainConfig

class WebDomainConfig(DomainConfig):
    """
    Configuration for web resources.
    Defines L1 primitives for web operations and handles encoding for web nodes.
    """
    def __init__(self, s_dim: int = 20):
        # L1 Primitives (Operations)
        primitives = [
            "WEB_HTTP_GET",
            "WEB_HTML_PARSE",
            "WEB_LINK_EXTRACT",
            "WEB_SEARCH_QUERY",
            "WEB_SEARCH_RESULT",
            "WEB_STATUS_200",
            "WEB_STATUS_404",
            "WEB_STATUS_500",
            "WEB_HYPERLINK"
        ]
        super().__init__(primitives, s_dim=s_dim)

    def encode_url(self, url: str) -> np.ndarray:
        """
        Encode a URL into the web manifold.
        Currently uses a simple identity encoding + concept activation for WEB_HTTP_GET.
        """
        mu = np.zeros(self.m_dim)
        # Activate WEB_HTTP_GET concept
        idx = self.concept_idx.get("WEB_HTTP_GET")
        if idx is not None:
            mu[self.S_DIM + idx] = 1.0
        
        # We could also hash the URL into the State part (S_DIM) for unique identification
        import hashlib
        h = int(hashlib.md5(url.encode()).hexdigest(), 16)
        for i in range(self.S_DIM):
            mu[i] = ((h >> (i * 8)) & 0xFF) / 255.0
            
        return mu

    def encode_search_query(self, query: str) -> np.ndarray:
        """Encode a search query into the web manifold."""
        mu = np.zeros(self.m_dim)
        idx = self.concept_idx.get("WEB_SEARCH_QUERY")
        if idx is not None:
            mu[self.S_DIM + idx] = 1.0
            
        # Mix in a hash of the query
        import hashlib
        h = int(hashlib.md5(query.encode()).hexdigest(), 16)
        for i in range(self.S_DIM):
            mu[i] = ((h >> (i * 8)) & 0xFF) / 255.0
            
        return mu
