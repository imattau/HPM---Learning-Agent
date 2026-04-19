"""
agent_registry.py - Global registry of HPM agents that can be discovered and invoked.
"""

from typing import Dict, Optional, List, Any

class AgentRegistry:
    """Global registry of available HPM agents."""
    _agents: Dict[str, Any] = {}  # agent_name -> agent info
    
    @classmethod
    def register(cls, name: str, agent: Any, description: str = ""):
        """
        Register an HPM agent so it can be discovered and used by other agents.
        """
        cls._agents[name] = {
            'instance': agent,
            'description': description,
            'input_keys': getattr(agent, 'required_observation_keys', ["input"]),
            'output_key': getattr(agent, 'output_key', 'output')
        }
    
    @classmethod
    def create_pattern(cls, name: str) -> Optional[Any]:
        """Create an AgentPattern that wraps a registered agent."""
        if name not in cls._agents:
            return None
        info = cls._agents[name]
        from .base import AgentPattern
        return AgentPattern(
            agent=info['instance'],
            agent_name=name,
            cost=0.5,  # Agents are more expensive than primitive tools
            description=info['description']
        )
    
    @classmethod
    def list_agents(cls) -> List[str]:
        """Return list of available agent names."""
        return list(cls._agents.keys())
    
    @classmethod
    def get_agent_info(cls, name: str) -> Optional[Dict]:
        """Return metadata for an agent."""
        return cls._agents.get(name)

    @classmethod
    def call(cls, name: str, **kwargs) -> Any:
        """Directly call a registered agent by name."""
        if name not in cls._agents:
            raise ValueError(f"Agent '{name}' not registered.")
        agent_info = cls._agents[name]
        agent = agent_info['instance']
        
        # Determine interface by inspecting agent
        if hasattr(agent, 'invoke'):
            return agent.invoke(kwargs)
        elif hasattr(agent, 'process'):
            return agent.process(kwargs)
        elif hasattr(agent, 'predict'):
            return agent.predict(kwargs)
        elif hasattr(agent, 'step'):
            return agent.step(kwargs)
        else:
            raise AttributeError(f"Agent {name} has no callable interface.")

    @classmethod
    def clear(cls):
        """Clear registry."""
        cls._agents.clear()
