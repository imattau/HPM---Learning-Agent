"""
SP42: Experiment 18 — Long-Horizon Goal Reasoning (Depth Test)

Tests reasoning stability over chains of up to 20 steps with distractors.
"""
import numpy as np
import time
from typing import List, Optional, Tuple, Dict

from hfn.hfn import HFN
from hfn.forest import Forest
from hfn.observer import Observer
from hfn.decoder import Decoder
from hfn.retriever import GoalConditionedRetriever
from hfn.evaluator import Evaluator

# --- 1. ENVIRONMENT ---

class GraphEnvironment:
    """A graph-based environment with linear paths and dead-end distractors."""
    def __init__(self, dim=10):
        self.dim = dim
        self.state = np.zeros(dim)
        # transitions: state_bytes -> { action_id -> next_state_vec }
        self.transitions: Dict[bytes, Dict[int, np.ndarray]] = {}

    def reset(self, state_vec=None):
        self.state = state_vec.copy() if state_vec is not None else np.zeros(self.dim)
        return self.state

    def step(self, action_id: int) -> np.ndarray:
        s_bytes = self.state.tobytes()
        if s_bytes in self.transitions and action_id in self.transitions[s_bytes]:
            self.state = self.transitions[s_bytes][action_id].copy()
        return self.state

    def generate_chain(self, depth: int, distractor_prob: float = 0.3):
        """Generate a linear chain of length 'depth' plus dead-end branches."""
        self.transitions = {}
        current_state = np.zeros(self.dim)
        
        for d in range(depth):
            s_bytes = current_state.tobytes()
            self.transitions[s_bytes] = {}
            
            # 1. THE VALID ACTION (move Dimension d % dim)
            next_state = current_state.copy()
            next_state[d % self.dim] += 1.0
            self.transitions[s_bytes][d % self.dim] = next_state
            
            # 2. DISTRACTORS (random dead ends)
            if np.random.rand() < distractor_prob:
                distractor_action = (d + 5) % self.dim
                if distractor_action not in self.transitions[s_bytes]:
                    # Shortcut: jump forward 2 steps in the chain (geometrically)
                    # but don't add any outward transitions from this state.
                    dead_end_state = current_state.copy()
                    dead_end_state[d % self.dim] += 1.0
                    dead_end_state[(d + 1) % self.dim] += 1.0
                    self.transitions[s_bytes][distractor_action] = dead_end_state
            
            current_state = next_state
            
        return np.zeros(self.dim), current_state # return start and goal

# --- 2. THE AGENT ---

class LongHorizonAgent:
    """Enhanced UnifiedAgent with backtracking capabilities."""
    def __init__(self, dim=10):
        self.dim = dim
        self.m_dim = 3 * dim # [State, Action, Delta]
        self.forest = Forest()
        
        # Observation Retriever (Unbiased)
        self.obs_retriever = GoalConditionedRetriever(
            self.forest, 
            target_slice=slice(2*dim, 3*dim), 
            target_weight=10.0 
        )
        
        self.observer = Observer(
            forest=self.forest,
            retriever=self.obs_retriever,
            tau=1.0,
            residual_surprise_threshold=1.2,
            alpha_gain=0.5, 
            beta_loss=0.05,
            node_use_diag=True
        )
        
        # Planning Retriever (Weight-Aware)
        self.plan_retriever = GoalConditionedRetriever(
            self.forest, 
            target_slice=slice(2*dim, 3*dim), # Focus on Delta
            target_weight=20.0,
            weight_provider=lambda nid: self.observer.get_weight(nid)
        )

    def perceive(self, state_t, action_id, state_t1):
        """Update world model from a transition."""
        vec = np.zeros(self.m_dim)
        vec[:self.dim] = state_t
        vec[self.dim + action_id] = 5.0 
        vec[2*self.dim:] = (state_t1 - state_t) * 5.0
        return self.observer.observe(vec)

    def inject_rules(self, env: GraphEnvironment):
        """Force the agent to 'know' the environment rules."""
        for s_bytes, actions in env.transitions.items():
            s_t = np.frombuffer(s_bytes, dtype=np.float64)
            for a_id, s_t1 in actions.items():
                self.perceive(s_t, a_id, s_t1)

    def plan(self, current_state, goal_state, max_steps=40) -> Tuple[List[int], int, float]:
        """Multi-Step Planning with DFS backtracking."""
        visited = set()
        nodes_explored = 0
        
        def solve(state, path, steps_left):
            nonlocal nodes_explored
            
            s_bytes = state.tobytes()
            if s_bytes in visited:
                return None
            visited.add(s_bytes)
            
            dist_to_goal = np.linalg.norm(goal_state - state)
            if dist_to_goal < 0.1:
                return path
            
            if steps_left <= 0:
                return None
                
            query_mu = np.zeros(self.m_dim)
            query_mu[:self.dim] = state
            query_mu[2*self.dim:] = (goal_state - state) * 5.0
            
            query_node = HFN(mu=query_mu, sigma=np.ones(self.m_dim), id="plan_query", use_diag=True)
            candidates = self.plan_retriever.retrieve(query_node, k=10)
            nodes_explored += len(candidates)
            
            def state_match_score(node):
                s_dist = np.linalg.norm(node.mu[:self.dim] - state)
                w = self.observer.get_weight(node.id)
                return s_dist / (w + 1e-6)

            candidates.sort(key=state_match_score)
            
            for rule in candidates:
                if np.linalg.norm(rule.mu[:self.dim] - state) > 0.5:
                    continue
                    
                action_id = int(np.argmax(rule.mu[self.dim:2*self.dim]))
                delta = rule.mu[2*self.dim:] / 5.0
                next_state = state + delta
                
                res = solve(next_state, path + [action_id], steps_left - 1)
                if res is not None:
                    return res
            
            return None

        full_path = solve(current_state, [], max_steps)
        return full_path if full_path else [], nodes_explored, 0.0

# --- 3. EXPERIMENT RUNNER ---

def run_experiment():
    print("--- SP42: Experiment 18 — Long-Horizon Goal Reasoning ---\n")
    
    DIM = 10
    DEPTHS = [3, 5, 10, 20]
    TRIALS = 10
    
    print(f"{'Depth':<6} | {'Success':<10} | {'Steps':<8} | {'Explored':<10} | {'Drift':<8} | {'Status'}")
    print("-" * 65)
    
    for depth in DEPTHS:
        successes = 0
        all_steps = []
        all_explored = []
        all_drifts = []
        
        for _ in range(TRIALS):
            env = GraphEnvironment(DIM)
            start_vec, goal_vec = env.generate_chain(depth, distractor_prob=0.4)
            
            valid_chain = [np.frombuffer(s, dtype=np.float64) for s in env.transitions.keys()]
            valid_chain.append(goal_vec)
            
            agent = LongHorizonAgent(DIM)
            agent.inject_rules(env)
            
            plan, explored, _ = agent.plan(start_vec, goal_vec, max_steps=depth + 20)
            
            env.reset(start_vec)
            trial_drifts = []
            for a in plan:
                env.step(a)
                min_dist = min([np.linalg.norm(env.state - v) for v in valid_chain])
                trial_drifts.append(min_dist)
            
            final_dist = np.linalg.norm(env.state - goal_vec)
            if final_dist < 0.1:
                successes += 1
                all_steps.append(len(plan))
                all_explored.append(explored)
                all_drifts.append(np.mean(trial_drifts) if trial_drifts else 0.0)
            else:
                all_explored.append(explored)
                # For failure, we still calculate drift of whatever path was taken
                all_drifts.append(np.mean(trial_drifts) if trial_drifts else 1.0)
                
        avg_success = successes / TRIALS
        avg_steps = np.mean(all_steps) if all_steps else 0
        avg_explored = np.mean(all_explored)
        avg_drift = np.mean(all_drifts)
        
        status = "PASSED" if avg_success >= 0.8 else "PARTIAL" if avg_success > 0.5 else "FAIL"
        print(f"{depth:<6} | {avg_success:<10.1%} | {avg_steps:<8.1f} | {avg_explored:<10.1f} | {avg_drift:<8.2f} | {status}")

    print("\n--- PERFORMANCE SUMMARY ---")
    print("Goal: Test if reasoning remains stable beyond short chains.")
    print("Success Criterion: Graceful degradation and stable branching.")

if __name__ == "__main__":
    run_experiment()
