"""WebAgent: specialised agent for web-scale resource discovery and interaction."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np
from hfn.hfn import HFN
from hpm_ai_v2.agents.base_agent import BaseHFNAgent
from hpm_ai_v2.agents.mixins.web_fetch import WebFetchMixin
from hpm_ai_v2.agents.mixins.web_search import WebSearchMixin
from hpm_ai_v2.domains.web_domain import WebDomainConfig

class WebAgent(BaseHFNAgent, WebFetchMixin, WebSearchMixin):
    """
    HFN-native agent for the web domain.
    Learns patterns of web interaction and builds a structural graph of web resources.
    """
    def __init__(self, config: Optional[WebDomainConfig] = None, **kwargs) -> None:
        if config is None:
            config = WebDomainConfig()
        super().__init__(config, **kwargs)

    def save_agent(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        super().save_state(str(path / "agent_state.pkl"))
        
        meta = {
            "web_history": self.web_history,
            "concepts": self.config.concepts,
            "s_dim": self.config.S_DIM
        }
        with open(path / "web_meta.json", "w") as f:
            json.dump(meta, f)

    @classmethod
    def load_agent(cls, directory: str) -> "WebAgent":
        path = Path(directory)
        with open(path / "web_meta.json") as f:
            meta = json.load(f)
        config = WebDomainConfig(s_dim=meta["s_dim"])
        config.concepts = meta["concepts"]
        
        agent = cls(config, cold_dir=str(path))
        agent.web_history = meta.get("web_history", [])
        agent.load_state(str(path / "agent_state.pkl"))
        return agent
