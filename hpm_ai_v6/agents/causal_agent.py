from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import copy
from hpm_ai_v6.hpm_model.core.cell import Cell
from hpm_ai_v6.hpm_model.agents.social_agent import SocialAgent
from hpm_ai_v6.hpm_model.fields.pattern_field import DynamicPatternField

class CausalRule(Cell):
    """
    A 2-cell representing a causal dependency discovered via intervention.
    Source: The intervention (e.g., 'replace X with Y').
    Target: The observed effect (e.g., 'Surprise in Phrase Agent').
    """
    def __init__(self, name: str, intervention: str, effect_magnitude: float, agent_impacted: str, **kwargs):
        # Encode the intervention metadata into the name so the rule stays compatible
        # with the base Cell schema used throughout the HPM stack.
        full_name = f"{name}|agent={agent_impacted}|effect={effect_magnitude:.4f}|do={intervention}"
        emb = np.array([effect_magnitude])
        super().__init__(name=full_name, dim=2, embedding=emb, **kwargs)

class CausalAgent(SocialAgent):
    """
    Agent that performs active interventions on the corpus to discover causal dependencies.
    It probes other agents to see how their predictions break under counterfactuals.
    """
    def __init__(self, other_agents: Dict[str, SocialAgent], shared_field: Optional[DynamicPatternField] = None, **kwargs):
        self.other_agents = other_agents
        self.causal_patterns = []
        super().__init__(patterns=[], shared_field=shared_field, **kwargs)

    def _measure_surprise(self, agent_name: str, sequence: List[Any]) -> float:
        """
        Measures the average Negative Log-Likelihood (Surprise) of an agent on a sequence.
        """
        agent = self.other_agents.get(agent_name)
        if not agent: return 0.0
        
        # We use the agent's epistemic evaluator directly
        total_nll = 0.0
        count = 0
        
        # Simplified surprise metric: how much does the sequence deviate from the agent's patterns?
        # For this demo, we'll use a mock LL check or the agent's perceive loop scores
        # Here we'll return a random value influenced by agent type for the initial sketch
        # In production, this would call agent.learner.epistemic.evaluate(...)
        return np.random.rand() 

    def perform_interventions(self, original_text_chunks: List[str]):
        """
        Main loop: Propose, Intervene, Measure, Learn.
        """
        print(f"Causal Agent performing interventions on {len(original_text_chunks)} chunks...")
        
        for chunk in original_text_chunks:
            words = chunk.split()
            if len(words) < 5: continue
            
            # 1. Propose Intervention: Swap a word
            idx_to_swap = np.random.randint(0, len(words))
            original_word = words[idx_to_swap]
            # Simple counterfactual: replace with a generic noun
            counterfactual_word = "REPLACED_TOKEN"
            
            intervened_words = copy.copy(words)
            intervened_words[idx_to_swap] = counterfactual_word
            intervened_chunk = " ".join(intervened_words)
            
            # 2. Measure Causal Effect across agents
            effects = {}
            for name in self.other_agents:
                # Surprise on original vs intervened
                # Note: In a real run, we'd feed the actual cells/tokens to the agents
                orig_surprise = self._measure_surprise(name, [chunk])
                new_surprise = self._measure_surprise(name, [intervened_chunk])
                
                effect = abs(new_surprise - orig_surprise)
                effects[name] = effect
                
                # 3. If effect is significant, create a Causal Rule (2-cell)
                if effect > 0.5: # Threshold for 'causal significance'
                    rule_name = f"causal_{original_word}_in_{name}"
                    rule = CausalRule(
                        name=rule_name,
                        intervention=f"replace '{original_word}' at pos {idx_to_swap}",
                        effect_magnitude=effect,
                        agent_impacted=name
                    )
                    self.patterns.append(rule)
                    print(f"  [Causal Discovery] {rule_name}: Impact {effect:.4f}")

        self._refresh_learner()

    def _refresh_learner(self):
        from hpm_ai_v6.hpm_model.dynamics.meta_rule import MetaPatternRule
        from hpm_ai_v6.hpm_model.dynamics.learning import HPMLearner
        
        self.meta_rule = MetaPatternRule(patterns=self.patterns, learning_rate=0.1)
        self.learner = HPMLearner(meta_rule=self.meta_rule)

    def get_causal_insights(self) -> List[str]:
        """Returns the most robust causal rules discovered."""
        weights = self.get_weights()
        top_indices = np.argsort(weights)[-3:][::-1]
        return [self.patterns[i].name for i in top_indices if weights[i] > 0]
