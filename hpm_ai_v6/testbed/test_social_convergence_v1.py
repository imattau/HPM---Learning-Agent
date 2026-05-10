"""
Multi‑agent social convergence test (HPM Prediction 9.5).
Requires: toy_hpm_v3.py (or the core classes Cell, MetaPatternRule, learn_step, etc.)
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
    make_1cell
)

# Define 0-cells locally
apple = Cell(0, np.array([1.0, 0.2]), name="apple")
banana = Cell(0, np.array([0.8, 0.9]), name="banana")
cherry = Cell(0, np.array([0.1, 1.2]), name="cherry")
objects = [apple, banana, cherry]

def get_initial_patterns():
    p1 = [
        Cell(dim=1, embedding=np.random.randn(2), source=apple, target=banana, name="A_to_B"),
        Cell(dim=1, embedding=np.random.randn(2), source=banana, target=cherry, name="B_to_C"),
        Cell(dim=1, embedding=np.random.randn(2), source=cherry, target=apple, name="C_to_A")
    ]
    return p1

# ------------------------------------------------------------
# 1. Environment: a hidden hierarchical grammar
# ------------------------------------------------------------
def generate_episode(length=20):
    """
    Generates a sequence of 0‑cells (objects) that follows a hidden hierarchical pattern.
    """
    phases = ['A', 'B', 'C'] * (length // 3 + 1)
    seq = []
    for phase in phases[:length]:
        if phase == 'A':
            if len(seq) == 0: seq.append(apple)
            else:
                last = seq[-1]
                seq.append(banana if last == apple else apple)
        elif phase == 'B':
            if len(seq) == 0: seq.append(banana)
            else:
                last = seq[-1]
                seq.append(cherry if last == banana else banana)
        else:  # phase C
            if len(seq) == 0: seq.append(cherry)
            else:
                last = seq[-1]
                seq.append(apple if last == cherry else cherry)
    return seq

# ------------------------------------------------------------
# 2. Social field (shared across agents)
# ------------------------------------------------------------
class SharedPatternField:
    """Implements a dynamic field that amplifies patterns popular across agents."""
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
# 3. Agent definition
# ------------------------------------------------------------
class HPMAgent:
    def __init__(self, patterns, learning_rate=0.2, conflict_scale=0.0,
                 beta_aff=0.5, gamma_soc=0.5, density_bias=0.1,
                 social_field=None, agent_id=0):
        self.patterns = deepcopy(patterns)
        self.mpr = MetaPatternRule(self.patterns, learning_rate, conflict_scale)
        self.beta_aff = beta_aff
        self.gamma_soc = gamma_soc
        self.density_bias = density_bias
        self.social_field = social_field
        self.id = agent_id

    def learn_episode(self, observation_seq, objects_list, curiosity_vec, consensus_vec):
        # We'll use a local field wrapper that respects the shared social field
        class FieldWrapper:
            def __init__(self, agent): self.agent = agent
            def get_amplification(self, name):
                if self.agent.social_field:
                    return self.agent.social_field.get_amplification(name)
                return 0.0
            def update(self, *args): pass # Shared field updated globally
            
        local_field = FieldWrapper(self)
        learn_step(
            self.mpr, local_field, observation_seq, self.patterns + objects_list,
            consensus_vec, embed_lr=0.05
        )

    def get_weights_dict(self):
        return {p.name: self.mpr.weights[i] for i, p in enumerate(self.patterns)}

# ------------------------------------------------------------
# 4. Convergence metric
# ------------------------------------------------------------
def compute_convergence(agents):
    n_patterns = len(agents[0].patterns)
    variances = []
    for i in range(n_patterns):
        weights_i = [agent.mpr.weights[i] for agent in agents]
        variances.append(np.var(weights_i))
    return np.mean(variances)

# ------------------------------------------------------------
# 5. Main simulation
# ------------------------------------------------------------
def run_social_convergence_test(n_agents=5, n_episodes=20, episode_length=20):
    curiosity_vec = np.array([0.5, 0.5])
    consensus_vec = np.array([1.0, 0.0])
    
    conditions = {
        'Social (shared field)': {'gamma_soc': 0.8, 'share_field': True},
        'Individual (no field)': {'gamma_soc': 0.1, 'share_field': False}
    }

    results = {}
    for cond_name, params in conditions.items():
        print(f"\nRunning: {cond_name}")
        init_patterns = get_initial_patterns()
        pattern_names = [p.name for p in init_patterns]
        shared_field = SharedPatternField(pattern_names) if params['share_field'] else None

        agents = [HPMAgent(init_patterns, gamma_soc=params['gamma_soc'], social_field=shared_field, agent_id=i) 
                  for i in range(n_agents)]

        convergence_history = []
        for ep in range(n_episodes):
            episode = generate_episode(episode_length)
            for agent in agents:
                agent.learn_episode(episode, objects, curiosity_vec, consensus_vec)

            if shared_field:
                shared_field.update([agent.get_weights_dict() for agent in agents])

            conv = compute_convergence(agents)
            convergence_history.append(conv)
            if ep % 5 == 0:
                print(f"  Episode {ep}: convergence (variance) = {conv:.6f}")

        results[cond_name] = convergence_history

    print("\nSocial Convergence Summary:")
    for cond, hist in results.items():
        print(f"  {cond:25}: Initial Var={hist[0]:.6f}, Final Var={hist[-1]:.6f}")

if __name__ == "__main__":
    run_social_convergence_test(n_agents=8, n_episodes=30, episode_length=15)
