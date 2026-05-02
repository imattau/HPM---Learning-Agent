import numpy as np
from typing import List, Optional, Any
from hpm_ai_v4.agents.agent import HPMAgent
from hpm_ai_v4.tools.substrate import ExternalSubstrate
from hpm_ai_v4.repository import PatternRepository
from hpm_ai_v4.field import InstitutionalField
from hpm_ai_v4.curriculum import CurriculumScheduler
from hpm_ai_v4.social import SocialNetwork
from hpm_ai_v4.reflection import ReflectionEngine
from hpm_ai_v4.tools.dictionary import DictionaryValidator
from hpm_ai_v4.tools.grammar import GrammarValidator
from hpm_ai_v4.tools.pattern_equivalence import PatternEquivalenceIndex

class MetaReasoner:
    """Reflection and multi-agent coordination layer."""
    def __init__(self, agent_pool: 'AgentPool'):
        self.agent_pool = agent_pool

    def detect_inconsistencies(self, agent: HPMAgent) -> List[str]:
        """Examine explanations from top patterns for conflicting predictions."""
        # Get top 2 patterns by weight
        patterns = sorted(agent.patterns, key=lambda p: p.weight, reverse=True)[:2]
        if len(patterns) < 2: return []
        
        inconsistencies = []
        # Predict next using the agent's recent history
        p1_pred = patterns[0].predict_next(agent.obs_buffer[-20:])
        p2_pred = patterns[1].predict_next(agent.obs_buffer[-20:])
        
        if p1_pred != p2_pred:
            inconsistencies.append(f"Conflict: Pattern {patterns[0].id} and {patterns[1].id} predict different outcomes.")
                
        return inconsistencies

class AgentPool:
    """Manages a population of HPM agents."""
    def __init__(self, num_agents: int = 5, external_substrate: Optional[ExternalSubstrate] = None,
                 obs_dim: int = 2, dictionary: Optional[DictionaryValidator] = None,
                 grammar: Optional[GrammarValidator] = None,
                 equivalence_index: Optional[PatternEquivalenceIndex] = None):
        self.substrate = external_substrate if external_substrate else ExternalSubstrate()
        self.agents = [HPMAgent(external_substrate=self.substrate, obs_dim=obs_dim, 
                                dictionary=dictionary, grammar=grammar,
                                equivalence_index=equivalence_index) for _ in range(num_agents)]

    def step(self, observation: int):
        """Synchronized step for all agents in the pool."""
        for agent in self.agents:
            agent.perceive_and_learn(observation)

class HPMMetaLayer:
    """
    Unified orchestrator for social, institutional, and developmental layers.
    """
    def __init__(self, env: Any, num_agents: int = 5, obs_dim: int = 2, 
                 dictionary: Optional[DictionaryValidator] = None,
                 grammar: Optional[GrammarValidator] = None):
        self.env = env # Still needed for curriculum/reflection to know context, but not for stepping
        self.substrate = ExternalSubstrate()
        self.equivalence_index = PatternEquivalenceIndex()
        self.agent_pool = AgentPool(num_agents=num_agents, external_substrate=self.substrate, 
                                    obs_dim=obs_dim, dictionary=dictionary, grammar=grammar,
                                    equivalence_index=self.equivalence_index)
        
        self.repository = PatternRepository(equivalence_index=self.equivalence_index)
        self.institution = InstitutionalField(dictionary=dictionary, grammar=grammar)
        self.curriculum = CurriculumScheduler(env)
        self.social_network = SocialNetwork()
        self.reflection = ReflectionEngine(self.agent_pool, self.repository)
        
        self.global_step = 0
        self.observations = []

    def run_step(self, obs: int) -> int:
        """Execute a single global step using an external observation."""
        # 1. Perception and individual learning
        self.observations.append(obs)
        self.agent_pool.step(obs)
        
        # 2. Pattern Repository updates (harvest high-density)
        if self.global_step % 10 == 0:
            self.repository.update(self.agent_pool)
            
        # 3. Institutional validation (peer review)
        if self.global_step % 50 == 0 and self.global_step > 0:
            val_seq = self.observations[-30:]
            for agent in self.agent_pool.agents:
                for p in agent.patterns:
                    # Provide social reliability bonus back to the agent
                    bonus = self.institution.evaluate(p, val_seq)
                    # Mapping boost [-0.3, 0.5] to a score [0, 1] for the social evaluator
                    score = np.clip(0.5 + bonus, 0, 1)
                    agent.external_social_scores[p.id] = score

        # 4. Social field propagation
        if self.global_step % 5 == 0:
            self.social_network.update(self.agent_pool)
            
        # 5. Developmental curriculum adjustment
        if self.global_step % 20 == 0:
            self.curriculum.update(self.agent_pool)
            
        # 6. Meta-reflection and intervention
        self.reflection.step(self.global_step)
        
        # 7. Persistence
        if self.global_step % 100 == 0:
            self.substrate.broadcast([p for a in self.agent_pool.agents for p in a.patterns])
            
        self.global_step += 1
        return obs

    def _group_patterns_by_id(self) -> List[List[Any]]:
        """Collect patterns from different agents with the same ID."""
        groups = {}
        for agent in self.agent_pool.agents:
            for p in agent.patterns:
                if p.id not in groups: groups[p.id] = []
                groups[p.id].append(p)
        return list(groups.values())

    def report(self):
        """Report summary of population-level cognitive state."""
        all_patterns = [p for a in self.agent_pool.agents for p in a.patterns]
        if not all_patterns: return
        
        avg_complexity = np.mean([p.complexity for p in all_patterns])
        avg_epistemic = np.mean([-p.running_loss for p in all_patterns])
        repo_size = len(self.repository.stored_patterns)
        
        print(f"Step {self.global_step}: | avg complexity={avg_complexity:.2f} | "
              f"avg epistemic={avg_epistemic:.2f} | repo={repo_size} | "
              f"curriculum stage={self.curriculum.stage}")
