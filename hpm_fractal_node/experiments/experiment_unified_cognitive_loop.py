"""
SP41: Experiment 17 — Unified Cognitive Loop Test (The Core Agent)

Integrates all HFN capabilities into a single autonomous agent:
Curiosity -> Multi-Step Planning -> Action -> Failure Detection -> Belief Revision -> Re-Simulation.

The agent navigates an evolving 10D world where rules can shift.
"""
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from hfn.hfn import HFN
from hfn.forest import Forest
from hfn.observer import Observer
from hfn.decoder import Decoder
from hfn.retriever import GoalConditionedRetriever
from hfn.evaluator import Evaluator

# --- 1. ENVIRONMENT ---

class EvolvingEnvironment:
    """A dynamic 10D continuous world with shifting rules."""
    def __init__(self, dim=10):
        self.dim = dim
        self.state = np.zeros(dim)
        # Default rules: Action i increments Dimension i by 1.0
        self.rules = {i: i for i in range(dim)}
        self.rule_deltas = {i: 1.0 for i in range(dim)}

    def reset(self, state=None):
        self.state = state if state is not None else np.zeros(self.dim)
        return self.state

    def step(self, action_id: int) -> np.ndarray:
        """Apply action and return the new state."""
        if action_id in self.rules:
            target_dim = self.rules[action_id]
            delta = self.rule_deltas[action_id]
            self.state[target_dim] += delta
        return self.state

    def shift_rule(self, action_id: int, new_delta: float):
        """Change the world!"""
        print(f"  [ENV] RULE SHIFT: Action {action_id} now adds {new_delta}")
        self.rule_deltas[action_id] = new_delta

# --- 2. THE UNIFIED AGENT ---

class UnifiedAgent:
    """Orchestrates curiosity, planning, and revision."""
    def __init__(self, dim=10):
        self.dim = dim
        self.m_dim = 3 * dim # [State, Action, Delta]
        self.forest = Forest()
        
        # 1. Observation Retriever (Unbiased, focuses on Delta matching)
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
            adaptive_compression=True,
            node_use_diag=True
        )
        
        # 2. Planning Retriever (Weight-Aware, prefers trusted beliefs)
        self.plan_retriever = GoalConditionedRetriever(
            self.forest, 
            target_slice=slice(2*dim, 3*dim), 
            target_weight=100.0,
            weight_provider=lambda nid: self.observer.get_weight(nid)
        )
        
        self.decoder = Decoder(target_forest=self.forest, sigma_threshold=0.1)

    def perceive(self, state_t, action_id, state_t1):
        """Observe a transition and update the world model."""
        st = np.array(state_t)
        st1 = np.array(state_t1)
        
        vec = np.zeros(self.m_dim)
        vec[:self.dim] = st
        vec[self.dim + action_id] = 5.0 
        vec[2*self.dim:] = (st1 - st) * 5.0
        
        res = self.observer.observe(vec)
        
        # FALSIFICATION: If we are surprised, penalize nodes that expected a different outcome
        if res.residual_surprise > self.observer.tau:
            query_mu = np.zeros(self.m_dim)
            query_mu[:self.dim] = st
            query_mu[self.dim + action_id] = 5.0
            
            query_node = HFN(mu=query_mu, sigma=np.ones(self.m_dim), id="falsify_query", use_diag=True)
            candidates = self.obs_retriever.retrieve(query_node, k=5)
            for c in candidates:
                if c.id not in [n.id for n in res.explanation_tree]:
                    self.observer.penalize_id(c.id, penalty=0.5)
                    
        return res

    def dream(self, n_dreams=20):
        """Self-Curiosity: explore the state space internally."""
        print(f"  [AGENT] Starting Curiosity Phase ({n_dreams} dreams)...")
        for d in range(n_dreams):
            # Placeholder for generative play
            pass

    def plan(self, current_state, goal_state, max_steps=10) -> List[int]:
        """Multi-Step Planning: find a sequence of actions to reach the goal."""
        plan_actions = []
        sim_state = current_state.copy()
        
        for _ in range(max_steps):
            target_delta = goal_state - sim_state
            if np.linalg.norm(target_delta) < 0.1:
                break
                
            query_mu = np.zeros(self.m_dim)
            query_mu[:self.dim] = sim_state
            query_mu[2*self.dim:] = target_delta * 5.0
            
            query_node = HFN(mu=query_mu, sigma=np.ones(self.m_dim), id="plan_query", use_diag=True)
            candidates = self.plan_retriever.retrieve(query_node, k=5)
            
            if not candidates:
                break
                
            rule = candidates[0]
            action_id = np.argmax(rule.mu[self.dim:2*self.dim])
            plan_actions.append(int(action_id))
            sim_state += rule.mu[2*self.dim:] / 5.0
            
        return plan_actions

# --- 3. THE CAPSTONE EXPERIMENT ---

def run_experiment():
    print("--- SP41: Experiment 17 — Unified Cognitive Loop Test ---\n")
    
    DIM = 10
    env = EvolvingEnvironment(DIM)
    agent = UnifiedAgent(DIM)
    
    # PHASE 1: Bootstrap Model
    print("PHASE 1: Learning the World...")
    for a in range(DIM):
        for _ in range(5):
            s_t = env.reset().copy() 
            s_t1 = env.step(a).copy()
            agent.perceive(s_t, a, s_t1)
        
    print(f"  World model stable. Forest Size: {len(list(agent.forest.active_nodes()))}")
    
    # PHASE 2: Goal Pursuit
    print("\nPHASE 2: Goal Pursuit (Multi-Step Reasoning)...")
    env.reset()
    start_state = env.state.copy()
    goal_state = np.array([2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    plan = agent.plan(start_state, goal_state)
    print(f"  Plan generated: {plan}")
    
    # Execute Plan
    print("  Executing Plan...")
    for a in plan:
        s_t = env.state.copy()
        s_t1 = env.step(a).copy()
        agent.perceive(s_t, a, s_t1)
        
    final_dist = np.linalg.norm(env.state - goal_state)
    print(f"  Final State: {env.state[:3]}... Dist to Goal: {final_dist:.4f}")
    
    # PHASE 3: Environmental Shift
    print("\nPHASE 3: Environmental Shift (Rule Change)...")
    # Action 0 now subtracts 1 instead of adding 1.
    env.shift_rule(0, -1.0)
    # Give the agent a new way to reach the goal: Action 9 now adds 1 to Dim 0
    env.rules[9] = 0
    env.shift_rule(9, 1.0)
    
    # PHASE 4: Adaptation
    print("\nPHASE 4: Adaptation (Continuous Cognitive Loop)...")
    
    max_attempts = 10
    success = False
    
    for attempt in range(max_attempts):
        print(f"\n  --- Attempt {attempt + 1} ---")
        env.reset()
        
        # 1. Plan
        new_plan = agent.plan(env.state, goal_state)
        print(f"  Plan: {new_plan}")
        
        # 2. Act & Perceive
        print("  Executing Plan...")
        plan_failed = False
        for a in new_plan:
            s_t = env.state.copy()
            s_t1 = env.step(a).copy()
            res = agent.perceive(s_t, a, s_t1)
            
            if res.residual_surprise > 1.0:
                print(f"    [SURPRISE] Action {a} failed expectation! Surprise: {res.residual_surprise:.4f}")
                plan_failed = True
                
        final_dist = np.linalg.norm(env.state - goal_state)
        print(f"  Final State: {env.state[:3]}... Dist to Goal: {final_dist:.4f}")
        
        if final_dist < 0.1:
            success = True
            print("\n[SUCCESS] Unified Cognitive Loop: Goal Reached after adaptation!")
            print("The agent successfully revised its beliefs and discovered a new path to the target.")
            break
            
        # 3. Explore if stuck
        if plan_failed or final_dist >= 0.1:
            print("  [EXPLORE] Goal not reached. Exploring environment to find new paths...")
            s_t = env.reset().copy()
            for _ in range(20):
                a = np.random.randint(0, DIM)
                s_t1 = env.step(a).copy()
                agent.perceive(s_t, a, s_t1)
                s_t = s_t1
                
    if not success:
        print("\n[FAIL] Agent failed to adapt and reach the goal within attempt limit.")

if __name__ == "__main__":
    run_experiment()
