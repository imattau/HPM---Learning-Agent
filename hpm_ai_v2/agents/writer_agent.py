"""WriterAgent: specialised agent for natural language synthesis and meta-cognitive discovery."""
from __future__ import annotations
from typing import Optional, List, Dict, Tuple
import numpy as np
from hfn.hfn import HFN
from hpm_ai_v2.agents.base_agent import BaseHFNAgent
from hpm_ai_v2.agents.mixins.writer import WriterMixin
from hpm_ai_v2.domains.text_domain import TextDomainConfig

class WriterAgent(BaseHFNAgent, WriterMixin):
    """
    HFN-native agent for text generation.
    Collaborates with ReaderAgent and WebAgent to synthesize knowledge.
    """
    def __init__(self, config: TextDomainConfig, reader_agent: "ReaderAgent", **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.reader_agent = reader_agent

    def request_knowledge(self, topic: str) -> bool:
        """
        Request missing knowledge from the Web-Reader pipeline.
        Triggers discovery if the current knowledge base lacks information.
        """
        print(f"      [WRITER] Requesting knowledge about: '{topic}'")
        if not self.reader_agent or not self.reader_agent.web_agent:
            print("      [WRITER] Error: Reader or Web agent not available for request.")
            return False
            
        # 1. Use WebAgent to search
        results = self.reader_agent.web_agent.search(topic, num_results=1)
        if not results:
            print(f"      [WRITER] No information found for '{topic}'.")
            return False
            
        webpage_node = results[0]
        # 2. Use WebAgent to fetch content
        text = self.reader_agent.web_agent.fetch_page(webpage_node)
        
        # 3. Use ReaderAgent to ingest
        if text:
            print(f"      [WRITER] Found information, Reader ingesting...")
            self.reader_agent.ingest_text(text, title=topic, webpage_node=webpage_node)
            return True
        return False

    def save_agent(self, directory: str) -> None:
        super().save_state(f"{directory}/writer_state.pkl")

    @classmethod
    def load_agent(cls, directory: str, config: TextDomainConfig, reader_agent: "ReaderAgent") -> "WriterAgent":
        agent = cls(config, reader_agent, cold_dir=directory)
        agent.load_state(f"{directory}/writer_state.pkl")
        return agent
