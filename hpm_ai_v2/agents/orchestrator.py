"""
AgentOrchestrator: Centralised management and synchronization for HPM-native agents.
Handles agent registration, forest sharing, and dimension broadcasting.
"""
from __future__ import annotations
import os
from typing import Dict, List, Optional, Type, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from hfn.forest import Forest
    from hpm_ai_v2.agents.base_agent import BaseHFNAgent

class AgentOrchestrator:
    """
    The central hub for a 'society' of HPM agents.
    Ensures all agents are dimensionally synchronized and can discover each other.
    """
    def __init__(self, forest: Forest) -> None:
        self.forest = forest
        self.agents: Dict[str, BaseHFNAgent] = {}
        self.types: Dict[str, List[str]] = {} # type -> list of agent_ids

    def register(self, agent: BaseHFNAgent, agent_id: Optional[str] = None) -> str:
        """Register an agent with the orchestrator."""
        if agent_id is None:
            agent_id = f"{agent.__class__.__name__.lower()}_{len(self.agents)}"
        
        self.agents[agent_id] = agent
        
        # Track by type
        a_type = getattr(agent.config, "domain_type", "base")
        if a_type not in self.types:
            self.types[a_type] = []
        self.types[a_type].append(agent_id)
        
        # Link agent back to orchestrator
        agent.orchestrator = self
        
        # Ensure initial dimension sync
        if hasattr(self.forest, "_D") and self.forest._D is not None:
            if agent.m_dim != self.forest._D:
                print(f"      [ORCHESTRATOR] Syncing initial dimension for '{agent_id}' to D={self.forest._D}")
                agent.reindex(self.forest._D, agent.s_dim)
        
        return agent_id

    def get_agent(self, agent_id: str) -> Optional[BaseHFNAgent]:
        return self.agents.get(agent_id)

    def find_specialists(self, domain_type: str) -> List[BaseHFNAgent]:
        """Find all registered agents of a specific domain type."""
        ids = self.types.get(domain_type, [])
        return [self.agents[aid] for aid in ids]

    def broadcast_reindex(self, new_dim: int, s_dim: int) -> None:
        """Synchronize all registered agents to a new dimensionality."""
        print(f"      [ORCHESTRATOR] Broadcasting reindex to {len(self.agents)} agents (D={new_dim})...")
        for aid, agent in self.agents.items():
            agent.reindex(new_dim, s_dim)
        print(f"      [ORCHESTRATOR] Broadcast complete.")

    def shutdown(self) -> None:
        """Cleanup all agents."""
        for agent in self.agents.values():
            if hasattr(agent, "save_state"):
                agent.save_state()

    def discover_knowledge(self, query: str, k: int = 5) -> List[HFN]:
        """
        Search across all registered forests in the global registry.
        Enables cross-domain and cross-experiment knowledge discovery.
        """
        from hpm_ai_v2.registry import get_registry
        from hfn.tiered_forest import TieredForest
        reg = get_registry()
        
        all_results = []
        for domain_id, entry in reg.get_all_entries().items():
            path = entry.get("path")
            if not path or not os.path.exists(path):
                continue
                
            # Temporarily load forest to search
            # We use D=None to autodetect
            temp_forest = TieredForest(D=None, cold_dir=path)
            
            # Use a dummy config or the orchestrator's main agent to encode
            # For now, we assume the query is already in the shared forest dimension
            # and we use standard Euclidean retrieval in the subspace.
            main_agent = next(iter(self.agents.values()), None)
            if not main_agent: continue
            
            mu = main_agent.config.encode_passage(query)
            results = temp_forest.retrieve(mu, k=k)
            all_results.extend(results)
            
        # Re-rank all combined results
        main_agent = next(iter(self.agents.values()), None)
        if main_agent:
            mu = main_agent.config.encode_passage(query)
            all_results.sort(key=lambda n: float(np.sum((n.mu - mu)**2)))
            
        return all_results[:k]
