import numpy as np
import copy
from typing import List, Dict, Any, Optional

from hpm_ai_v4.pattern import HierarchicalPattern, FlatPattern
from hpm_ai_v4.operators.parallel import ParallelPatternPool, update_pattern_resident
from hpm_ai_v4.evaluators.metrics import total_score
from hpm_ai_v4.operators.dynamics import compute_conflict_matrix, meta_pattern_update, recombine
from hpm_ai_v4.field import PatternField, InstitutionalField
from hpm_ai_v4.tools.substrate import ExternalSubstrate
from hpm_ai_v4.agents.reasoning import Reasoner
from hpm_ai_v4.tools.dictionary import DictionaryValidator
from hpm_ai_v4.tools.grammar import GrammarValidator

class DevelopmentalStage:
    """Modulates evaluator focus based on current population complexity."""
    LEVELS = ['surface', 'local', 'relational', 'abstract', 'generative']
    
    def __init__(self, agent: 'HPMAgent'):
        self.agent = agent
        self.level_idx = 0   # start at surface level

    @property
    def level(self) -> str:
        return self.LEVELS[self.level_idx]

    def update(self, patterns: List[HierarchicalPattern], global_step: int):
        # Determine average complexity in current population, weighted by replicator weight
        if not patterns: return
        
        weights = np.array([p.weight for p in patterns])
        complexities = np.array([p.complexity for p in patterns])
        avg_complexity = np.sum(weights * complexities) / (np.sum(weights) + 1e-12)
        
        # Progression logic: advance level as complexity grows
        if avg_complexity > 1.5 and self.level_idx < 1:
            self.level_idx = 1   # local structural
        if avg_complexity > 2.0 and self.level_idx < 2:
            self.level_idx = 2   # relational
        if avg_complexity > 2.5 and self.level_idx < 3:
            self.level_idx = 3   # abstract
        if avg_complexity > 2.8 and self.level_idx < 4:
            self.level_idx = 4   # generative
            
        # Modulate evaluator weightings based on developmental stage
        # (Following Section 7.4 of the framework)
        if self.level_idx == 0:
            self.agent.beta_aff = 0.2   # focus on surface feedback (social)
            self.agent.gamma_soc = 0.6
        elif self.level_idx == 1:
            self.agent.beta_aff = 0.4
            self.agent.gamma_soc = 0.4
        elif self.level_idx == 2:
            self.agent.beta_aff = 0.6
            self.agent.gamma_soc = 0.2
        else:
            self.agent.beta_aff = 0.8   # focus on curiosity and generative discovery
            self.agent.gamma_soc = 0.1

class HPMAgent:
    """The central HPM learner, integrating patterns, evaluators, and fields."""
    def __init__(self, num_initial_patterns: int = 5, external_substrate: Optional[ExternalSubstrate] = None,
                 obs_dim: int = 2, num_workers: int = 1, 
                 dictionary: Optional[DictionaryValidator] = None,
                 grammar: Optional[GrammarValidator] = None):
        self.obs_dim = obs_dim
        self.dictionary = dictionary
        self.grammar = grammar
        self.patterns = []
        for i in range(num_initial_patterns):
            p = HierarchicalPattern(pattern_id=i, obs_dim=obs_dim)
            p.weight = 0.01  # Initial hierarchical patterns start as weak hypotheses
            self.patterns.append(p)
            
        # Include one flat pattern for comparative baseline (strong initial weight)
        flat_p = FlatPattern.flat(num_initial_patterns, obs_dim=obs_dim)
        flat_p.weight = 0.95
        self.patterns.append(flat_p)

        self.external = external_substrate if external_substrate else ExternalSubstrate()
        self.field = PatternField()
        self.development = DevelopmentalStage(self)
        self.reasoner = Reasoner(self, dictionary=dictionary, grammar=grammar)
        
        self.step_counter = 0
        self.obs_buffer = []
        self.external_social_scores = {} # pattern_id -> reliability score [0, 1]
        
        # Default evaluator weights (will be modulated by development)
        self.beta_aff = 0.4
        self.gamma_soc = 0.3
        
        self._pool = ParallelPatternPool(num_workers=num_workers)

    def gossip_with_substrate(self, substrate: ExternalSubstrate):
        """Retrieve a random pattern from the collective substrate and inject it into the local population."""
        other = substrate.get_random_pattern()
        if other is not None and other.id not in [p.id for p in self.patterns]:
            # Incorporate as a low-weight hypothesis to avoid population destabilization
            new_p = copy.deepcopy(other)
            # Re-id to avoid local collisions if necessary, or just keep global ID
            new_p.weight = 0.05
            self.patterns.append(new_p)

    def perceive_and_learn(self, obs: int, feedback: Optional[Dict[str, Any]] = None):
        """Update patterns based on a new observation."""
        context_before = list(self.obs_buffer[-20:])
        self.obs_buffer.append(obs)
        if len(self.obs_buffer) > 100:
            self.obs_buffer = self.obs_buffer[-100:]

        # Reasoning feedback: update per-pattern loss based on prediction error
        self.reasoner.observe_outcome(obs, context_before, metadata=feedback)

        # 1. Update Social Context (Pattern Field) - Move up for workers
        field_freq = self.field.update(self.patterns)

        # 2. Parallel per-pattern update + score computation
        worker_params = {
            'learning_rate': 0.02,
            'lambda_l': 0.1,
            'adapt_window': 20,
            'beta_aff': self.beta_aff,
            'gamma_soc': self.gamma_soc,
            'external_soc_map': self.external_social_scores,
            'do_param_update': (self.step_counter % 5 == 0),
        }

        if self._pool.num_workers == 1:
            results = []
            for p in self.patterns:
                result = update_pattern_resident(p, self.obs_buffer, field_freq, worker_params)
                result['pattern_id'] = p.id
                result['complexity'] = p.complexity
                result['latent_dim'] = p.latent_dim
                result['obs_dim'] = p.obs_dim
                result['B'] = p.B.copy()
                result['running_loss'] = float(p.running_loss)
                result['weight'] = float(p.weight)
                if p.complexity >= 2 or p.latent_dim > 1:
                    result['A'] = p.A.copy()
                    result['pi'] = p.pi.copy()
                results.append(result)
        else:
            results = self._pool.map_patterns(
                self.patterns, self.obs_buffer, field_freq, worker_params
            )

        # 3. Write updated state back into pattern objects and collect totals
        result_by_id = {r['pattern_id']: r for r in results}
        totals = {}
        for p in self.patterns:
            r = result_by_id[p.id]
            if p.complexity >= 2:
                p.A = r['A']
                p.B = r['B']
                p.pi = r['pi']
            else:
                p.B = r['B']
            p.running_loss = r['running_loss']
            totals[p.id] = r['total_score']

        # 4. Meta Pattern Update (Replicator Dynamics with Conflict)
        k_mat = compute_conflict_matrix(self.patterns)
        meta_pattern_update(self.patterns, totals, eta=0.1, beta_c=0.03,
                            k_matrix=k_mat, decay=0.005)

        # 5. Population Pruning
        self.patterns = [p for p in self.patterns if p.weight > 1e-4]

        # 6. Recombination / Innovation
        if self.step_counter > 0 and self.step_counter % 100 == 0 and len(self.patterns) >= 2:
            weights = np.array([p.weight for p in self.patterns])
            if np.sum(weights) > 0:
                probs = weights / np.sum(weights)
                parents_idx = np.random.choice(len(self.patterns), size=2, p=probs, replace=False)
                
                child = recombine(self.patterns[parents_idx[0]], self.patterns[parents_idx[1]])
                if child is not None:
                    # Assign new id
                    current_ids = [p.id for p in self.patterns]
                    child.id = max(current_ids) + 1 if current_ids else 0
                    child.weight = 0.05
                    self.patterns.append(child)

        # 7. Persistence / Substrate Sharing & Gossip
        if self.step_counter % 20 == 0:
            self.external.broadcast(self.patterns)
            self.gossip_with_substrate(self.external)
        elif self.step_counter % 10 == 0:
            self.external.broadcast(self.patterns)

        # 8. Developmental Update
        self.development.update(self.patterns, self.step_counter)

        self.step_counter += 1

    def load_library(self, path: str, reset_weights: bool = True) -> int:
        """Load patterns from a serialised library file.

        Replaces self.patterns with the loaded population.
        If reset_weights=True, normalises all weights to 1/N so no single
        prior pattern dominates at the start of the new learning session.
        Returns N (number of patterns loaded).
        """
        from hpm_ai_v4.tools.serializer import PatternSerializer
        self.patterns = PatternSerializer.load(path)
        if reset_weights and self.patterns:
            w = 1.0 / len(self.patterns)
            for p in self.patterns:
                p.weight = w
        return len(self.patterns)

    def act(self, goal: Optional[int] = None) -> int:
        """Select an action (next observation to aim for) using the reasoning layer."""
        if not self.obs_buffer:
            return 0
            
        relevant = self.reasoner.get_relevant_patterns(self.obs_buffer, top_k=3)
        
        if goal is not None:
            plan = self.reasoner.plan(goal_state=goal, horizon=3)
            if plan:
                return plan[0]
                
        # Fallback to compositional predictive inference
        blended = self.reasoner.compose_predictions(relevant, self.obs_buffer)
        return int(np.argmax(blended))
