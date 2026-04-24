import numpy as np
from typing import List, Any, Optional
from hpm_ai_v4.meta import HPMMetaLayer
from hpm_ai_v4.io.adapters import InputAdapter, OutputAdapter

class TotalHPMSystem:
    """
    Complete HPM cognitive architecture integrating I/O, Core, and Meta layers.
    """
    def __init__(self, input_adapter: InputAdapter, output_adapter: OutputAdapter, 
                 env: Any, num_agents: int = 3):
        self.input_adapter = input_adapter
        self.output_adapter = output_adapter
        self.meta_layer = HPMMetaLayer(env, num_agents=num_agents, obs_dim=input_adapter.obs_dim)

    def step(self, raw_input: Any) -> Optional[int]:
        """Perform one complete cognitive cycle from raw input to action."""
        # 1. Input Processing: Raw Data -> Discrete Tokens
        obs_seq = self.input_adapter.to_observations(raw_input)
        
        # 2. Sequential Cognitive Processing (Learning, Social, Institutional)
        for obs in obs_seq:
            self.meta_layer.run_step(obs)
            
        # 3. Decision / Action: Deliberative act via primary agent's reasoning layer
        agents = self.meta_layer.agent_pool.agents
        if agents:
            primary_agent = agents[0]
            prediction = primary_agent.act()
            
            # 4. Output Adaptation
            self.output_adapter.act(prediction, context=None)
            return prediction
        return None

    def run_loop(self, raw_input_stream: List[Any]):
        """Run the cognitive architecture over a stream of raw data."""
        for raw in raw_input_stream:
            self.step(raw)
            
    def get_summary(self):
        """Provide a summary of the system's current cognitive state."""
        self.meta_layer.report()
