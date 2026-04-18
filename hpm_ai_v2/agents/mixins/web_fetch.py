"""WebFetchMixin: adds autonomous web fetching and link extraction to agents."""
from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import numpy as np
from hfn.hfn import HFN
from hpm_ai_v2.utils.text_fetcher import fetch_url, strip_html

class WebFetchMixin:
    """
    Mixin for HFN agents that adds web fetching capabilities.
    Represents webpages as HFN nodes and supports structural strategies.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.web_history: List[str] = [] # list of URLs visited

    def fetch_webpage(self, url: str) -> HFN:
        """
        Fetch a webpage and return its HFN node representation.
        Encapsulates fetching strategy and status tracking.
        """
        # Create a stub node first
        mu = self.config.encode_url(url)
        import uuid
        webpage_id = f"webpage_{uuid.uuid4().hex[:8]}"
        node = HFN(
            mu=mu,
            sigma=np.ones(self.m_dim) * 0.1,
            id=webpage_id,
            use_diag=True
        )
        node.metadata = {
            "type": "webpage",
            "url": url,
            "status": "unfetched"
        }
        node.relation_type = "webpage"
        self.observer.register(node, protected=False)
        self.patterns[webpage_id] = node
        
        # Now fetch it
        self.fetch_page(node)
        return node

    def fetch_page(self, webpage_node: HFN) -> str:
        """
        Fetch the content of a webpage node.
        Updates node metadata with status and text. Returns text.
        """
        url = webpage_node.metadata.get("url")
        if not url: return ""
        
        print(f"      [WEB] Fetching: {url}")
        try:
            text = fetch_url(url)
            status = 200
        except Exception as e:
            print(f"      [WEB] Error fetching {url}: {e}")
            text = ""
            status = 404
            
        # Update node metadata
        webpage_node.metadata["text"] = text
        webpage_node.metadata["status"] = status
        
        # Update mu with status concept
        status_concept = f"WEB_STATUS_{status}"
        if status_concept in self.config.concept_idx:
            idx = self.config.concept_idx[status_concept]
            webpage_node.mu[self.config.S_DIM + idx] = 1.0
            
        self.web_history.append(url)
        return text

    def extract_links_hierarchical(self, webpage_node: HFN) -> List[HFN]:
        """
        Extract links from a webpage and create hyperlink nodes.
        Demonstrates structural composition.
        """
        url = webpage_node.metadata.get("url")
        text = webpage_node.metadata.get("text", "")
        
        # Simple regex-based link extraction for demonstration
        import re
        links = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        
        hyperlink_nodes = []
        for target_url in links:
            # Create a target webpage node (stub, not fetched yet)
            target_mu = self.config.encode_url(target_url)
            import uuid
            target_id = f"webpage_{uuid.uuid4().hex[:8]}"
            target_node = HFN(mu=target_mu, sigma=np.ones(self.m_dim)*0.5, id=target_id, use_diag=True)
            target_node.metadata = {"type": "webpage", "url": target_url, "status": "unfetched"}
            self.observer.register(target_node, protected=False)
            self.patterns[target_id] = target_node
            
            # Create a hyperlink node (Fractal composition)
            link_mu = self.config.get_concept_vector("WEB_HYPERLINK")
            link_id = f"hyperlink_{uuid.uuid4().hex[:8]}"
            link_node = HFN(mu=link_mu, sigma=np.ones(self.m_dim)*0.1, id=link_id, use_diag=True)
            link_node.add_child(webpage_node)
            link_node.add_child(target_node)
            link_node.relation_type = "hyperlink"
            
            self.observer.register(link_node, protected=False)
            self.patterns[link_id] = link_node
            hyperlink_nodes.append(link_node)
            
        return hyperlink_nodes
