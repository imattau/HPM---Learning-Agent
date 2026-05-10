"""
Multi‑agent social convergence test (Prediction 9.5) v2.
Incorporates:
- Initial stochasticity (weights and embeddings)
- Noisy environment (stochastic grammar)
- Exponential forgetting & replication institution
"""

import numpy as np
from copy import deepcopy
from collections import defaultdict
import os
import sys

# Ensure the testbed is in the path
sys.path.append(os.path.join(os.getcwd(), 'hpm_ai_v6/testbed'))

from toy_hpm_v4 import (
    Cell, MetaPatternRule, learn_step, hierarchical_total_score,
    make_1cell, DynamicPatternField
)

# Define 0-cells locally
apple = Cell(0, np.array([1.0, 0.2]), name="apple")
banana = Cell(0, np.array([0.8, 0.9]), name="banana")
cherry = Cell(0, np.array([0.1, 1.2]), name="cherry")
objects = [apple, banana, cherry]

def get_initial_patterns():
    return [
        Cell(dim=1, embedding=np.random.randn(2), source=apple, target=banana, name="A_to_B"),
        Cell(dim=1, embedding=np.random.randn(2), source=banana, target=cherry, name="B_to_C"),
        Cell(dim=1, embedding=np.random.randn(2), source=cherry, target=apple, name="C_to_A")
    ]

# ------------------------------------------------------------
# 1. Noisy environment (stochastic grammar)
# ------------------------------------------------------------
def generate_noisy_episode(length=20, noise_level=0.2):
    phases = ['A', 'B', 'C'] * (length // 3 + 1)
    seq = []
    for phase in phases[:length]:
        if phase == 'A':
            next_obj = apple if (not seq or seq[-1] != apple) else banana
        elif phase == 'B':
            next_obj = banana if (not seq or seq[-1] != banana) else cherry
        else: # C
            next_obj = cherry if (not seq or seq[-1] != cherry) else apple
            
        if np.random.rand() < noise_level:
            next_obj = np.random.choice([o for o in objects if o != next_obj])
        seq.append(next_obj)
    return seq

# ------------------------------------------------------------
# 2. Shared pattern field
# ------------------------------------------------------------
class SharedPatternField:
    def __init__(self, pattern_names, initial_bias=0.1, decay=0.9, influence_rate=0.1):
        self.amplification = {name: initial_bias for name in pattern_names}
        self.decay = decay
        self.influence_rate = influence_rate

    def update(self, all_agent_weights_dicts):
        avg_weights = defaultdict(float)
        for agent_w in all_agent_weights_dicts:
            for name, w in agent_w.items():
                avg_weights[name] += w
        n_agents = len(all_agent_weights_dicts)
        for name in avg_weights:
            avg_weights[name] /= n_agents
            inc = self.influence_rate * avg_weights[name]
            self.amplification[name] = self.amplification[name] * self.decay + inc

    def get_amplification(self, pattern_name):
        return self.amplification.get(pattern_name, 0.0)

# ------------------------------------------------------------
# 3. Agent with stochastic initialisation
# ------------------------------------------------------------
class HPMAgent:
    def __init__(self, base_patterns, learning_rate=0.2, gamma_soc=0.5, 
                 social_field=None, agent_id=0, forget_decay=0.99):
        self.patterns = []
        for p in base_patterns:
            noisy_emb = p.emb + np.random.randn(*p.emb.shape) * 0.1
            self.patterns.append(Cell(p.dim, noisy_emb, p.src, p.tgt, p.name))
            
        self.mpr = MetaPatternRule(self.patterns, learning_rate, decay=forget_decay)
        # Random initial weights
        self.mpr.weights = np.random.dirichlet(np.ones(len(self.patterns)))
        
        self.gamma_soc = gamma_soc
        self.social_field = social_field
        self.id = agent_id

    def learn_episode(self, observation_seq, objects_list, consensus_vec):
        class FieldWrapper:
            def __init__(self, agent): self.agent = agent
            def get_amplification(self, name):
                return self.agent.social_field.get_amplification(name) if self.agent.social_field else 0.0
            def update(self, *args): pass
            
        local_field = FieldWrapper(self)
        learn_step(
            self.mpr, local_field, observation_seq, self.patterns + objects_list,
            consensus_vec, embed_lr=0.05, gamma_soc=self.gamma_soc
        )

    def get_weights_dict(self):
        return {p.name: self.mpr.weights[i] for i, p in enumerate(self.patterns)}

# ------------------------------------------------------------
# 4. Replication Institution (Pruning)
# ------------------------------------------------------------
def replication_institution(agents, prune_ratio=0.1):
    for agent in agents:
        # Simple pruning of lowest weight patterns
        weights = agent.mpr.weights
        threshold = np.percentile(weights, prune_ratio * 100)
        for i in range(len(weights)):
            if weights[i] < threshold:
                weights[i] = 0.0
        total = np.sum(weights)
        if total > 0: agent.mpr.weights = weights / total

# ------------------------------------------------------------
# 5. Simulation
# ------------------------------------------------------------
def run_simulation(n_agents=5, n_episodes=20, noise_level=0.2):
    curiosity_vec = np.array([0.5, 0.5])
    consensus_vec = np.array([1.0, 0.0])
    base_patterns = get_initial_patterns()
    
    conditions = {
        'Social (shared field)': {'gamma_soc': 0.8, 'share_field': True},
        'Individual (no field)': {'gamma_soc': 0.1, 'share_field': False}
    }

    results = {}
    for cond_name, params in conditions.items():
        print(f"\nRunning: {cond_name}")
        pattern_names = [p.name for p in base_patterns]
        shared_field = SharedPatternField(pattern_names) if params['share_field'] else None
        agents = [HPMAgent(base_patterns, gamma_soc=params['gamma_soc'], 
                           social_field=shared_field, agent_id=i) for i in range(n_agents)]

        var_history = []
        for ep in range(n_episodes):
            for agent in agents:
                episode = generate_noisy_episode(length=15, noise_level=noise_level)
                agent.learn_episode(episode, objects, consensus_vec)

            if shared_field:
                shared_field.update([agent.get_weights_dict() for agent in agents])
            
            if ep % 5 == 0:
                replication_institution(agents)

            # Compute variance of weights across agents
            n_pats = len(base_patterns)
            variances = [np.var([a.mpr.weights[i] for a in agents]) for i in range(n_pats)]
            avg_var = np.mean(variances)
            var_history.append(avg_var)
            if ep % 5 == 0:
                print(f"  Episode {ep}: Avg Weight Variance = {avg_var:.6f}")

        results[cond_name] = var_history

    print("\nSocial Convergence Summary:")
    for cond, hist in results.items():
        print(f"  {cond:25}: Start Var={hist[0]:.6f}, End Var={hist[-1]:.6f}")

if __name__ == "__main__":
    run_simulation(n_agents=10, n_episodes=40, noise_level=0.3)
