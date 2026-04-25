import numpy as np
from hpm_ai_v4.pattern import HierarchicalPattern, FlatPattern
from hpm_ai_v4.evaluators.metrics import epistemic_score, affective_score, social_score, total_score
from hpm_ai_v4.operators.dynamics import compute_conflict_matrix, meta_pattern_update, recombine

np.random.seed(42)   # for reproducibility

# ============================================================================
# Environment: True 2‑state HMM (the "world" the agent learns)
# ============================================================================
class TrueEnvironment:
    def __init__(self):
        # True transition and emission matrices
        self.A_true = np.array([[0.8, 0.2],
                                [0.2, 0.8]])
        self.B_true = np.array([[0.9, 0.1],
                                [0.1, 0.9]])
        self.state = 0

    def reset(self):
        self.state = np.random.choice([0, 1], p=[0.5, 0.5])

    def step(self):
        obs = np.random.choice([0, 1], p=self.B_true[self.state])
        self.state = np.random.choice([0, 1], p=self.A_true[self.state])
        return int(obs)

# ============================================================================
# Agent (contains the pattern population)
# ============================================================================
class HPMAgentLocal:
    def __init__(self, num_hier=3, num_flat=3):
        self.patterns = []
        # hierarchical patterns
        for i in range(num_hier):
            p = HierarchicalPattern(i, latent_dim=2, obs_dim=2)
            p.weight = 1.0 / (num_hier + num_flat)
            self.patterns.append(p)
        # flat patterns
        for i in range(num_hier, num_hier+num_flat):
            p = FlatPattern(i, obs_dim=2)
            p.weight = 1.0 / (num_hier + num_flat)
            self.patterns.append(p)
        self.obs_buffer = []
        self.step_counter = 0

    def perceive_and_learn(self, obs, field_freq):
        self.obs_buffer.append(obs)
        if len(self.obs_buffer) > 200:
            self.obs_buffer = self.obs_buffer[-200:]

        # 1) Update each pattern
        for p in self.patterns:
            if hasattr(p, 'update_parameters_online'):
                p.update_parameters_online(self.obs_buffer, window_size=30)
            elif hasattr(p, 'observe'):
                p.observe(obs)
            p.update_running_loss(self.obs_buffer[-30:])

        # 2) Compute total scores
        totals = {}
        for p in self.patterns:
            totals[p.id] = total_score(p, self.obs_buffer, field_freq)

        # 3) Compute conflict matrix
        k_mat = compute_conflict_matrix(self.patterns)

        # 4) Replicator update
        meta_pattern_update(self.patterns, totals, eta=0.1, beta_c=0.03,
                            k_matrix=k_mat, decay=0.005)

        # 5) Recombination every 25 steps
        if self.step_counter > 0 and self.step_counter % 25 == 0:
            weights = np.array([p.weight for p in self.patterns])
            if np.sum(weights) > 0:
                probs = weights / np.sum(weights)
                idx = np.random.choice(len(self.patterns), size=2, p=probs, replace=False)
                parent_a = self.patterns[idx[0]]
                parent_b = self.patterns[idx[1]]
                child = recombine(parent_a, parent_b)
                if child is not None:
                    child.id = max([p.id for p in self.patterns] + [0]) + 1
                    child.creation_step = self.step_counter
                    self.patterns.append(child)

        # 6) Prune low‑weight patterns
        self.patterns = [p for p in self.patterns if p.weight > 0.001]

        self.step_counter += 1

# ============================================================================
# Main simulation
# ============================================================================
def run_simulation(steps=500):
    env = TrueEnvironment()
    env.reset()
    agent = HPMAgentLocal(num_hier=3, num_flat=3)

    # For logging
    history = []

    for step in range(steps):
        obs = env.step()
        # Field frequencies: simply the current weight distribution
        weights = np.array([p.weight for p in agent.patterns])
        field_freq = {p.id: w / (np.sum(weights)+1e-12) for p, w in zip(agent.patterns, weights)}
        agent.perceive_and_learn(obs, field_freq)

        if step % 50 == 0 or step == steps-1:
            # Get best pattern
            best = max(agent.patterns, key=lambda p: p.weight)
            print(f"Step {step:3d}: best pattern id={best.id}, "
                  f"type={'hier' if (best.complexity >= 2 or best.latent_dim > 1) else 'flat'}, "
                  f"weight={best.weight:.3f}, ep_score={epistemic_score(best):.3f}, "
                  f"comp={best.compression() if (best.complexity >= 2 or best.latent_dim > 1) else 0:.3f}, "
                  f"pop_size={len(agent.patterns)}")
            history.append((step, best.id, best.weight, epistemic_score(best)))

    # Final analysis
    print("\n--- Final population ---")
    for p in agent.patterns:
        print(f"ID {p.id}: weight={p.weight:.3f}, "
              f"type={'hier' if (p.complexity >= 2 or p.latent_dim > 1) else 'flat'}, "
              f"ep_score={epistemic_score(p):.3f}")
    return history

if __name__ == "__main__":
    history = run_simulation(500)
