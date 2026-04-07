"""
SP43: Experiment 19 — Adversarial Belief Revision (Truth Under Conflict)

Tests whether the HFN system can unlearn entrenched, high-confidence but incorrect beliefs.
"""
import numpy as np
import time
from typing import List, Dict, Tuple, Optional

from hfn.hfn import HFN
from hfn.forest import Forest
from hfn.observer import Observer
from hfn.retriever import GoalConditionedRetriever

# --- 1. ENVIRONMENT ---

class AdversarialEnvironment:
    """A simple sequence environment that flips its rules after an initial phase."""
    def __init__(self, dim=10):
        self.dim = dim
        self.phase = 1 # 1: Misleading (A->B->C), 2: True (A->B->D)
        
        # Define states as vectors
        self.A = np.zeros(dim)
        self.B = np.zeros(dim)
        self.B[0] = 1.0 # State B
        
        self.C = np.zeros(dim)
        self.C[0] = 1.0
        self.C[1] = 1.0 # State C: Delta is [0, 1, 0...]
        
        self.D = np.zeros(dim)
        self.D[0] = 1.0
        self.D[2] = 1.0 # State D: Delta is [0, 0, 1...]
        
        self.state = self.A.copy()

    def reset(self):
        self.state = self.A.copy()
        return self.state

    def step(self, action_id: int = 0) -> np.ndarray:
        """
        Transition logic:
        If State is A: Move to B
        If State is B: 
           Phase 1: Move to C
           Phase 2: Move to D
        """
        if np.allclose(self.state, self.A):
            self.state = self.B.copy()
        elif np.allclose(self.state, self.B):
            if self.phase == 1:
                self.state = self.C.copy()
            else:
                self.state = self.D.copy()
        else:
            # Loop back to A for continuous learning
            self.state = self.A.copy()
            
        return self.state

# --- 2. THE AGENT ---

class BeliefRevisionAgent:
    def __init__(self, dim=10):
        self.dim = dim
        self.m_dim = 3 * dim # [State, Action, Delta]
        self.forest = Forest()
        
        # Simple Retriever
        self.retriever = GoalConditionedRetriever(
            self.forest,
            target_slice=slice(2*dim, 3*dim),
            target_weight=10.0
        )
        
        self.observer = Observer(
            forest=self.forest,
            retriever=self.retriever,
            tau=1.0,
            residual_surprise_threshold=1.2,
            alpha_gain=0.5,
            beta_loss=0.1,
            node_use_diag=True
        )

    def perceive(self, state_t, action_id, state_t1):
        """Update model and explicitly falsify on surprise."""
        st = np.array(state_t)
        st1 = np.array(state_t1)
        
        vec = np.zeros(self.m_dim)
        vec[:self.dim] = st
        vec[self.dim + action_id] = 5.0 # Action 0
        vec[2*self.dim:] = (st1 - st) * 5.0 # Scaled delta
        
        res = self.observer.observe(vec)
        
        # Explicit Falsification (from Exp 17)
        if res.residual_surprise > self.observer.tau:
            query_mu = np.zeros(self.m_dim)
            query_mu[:self.dim] = st
            query_mu[self.dim + action_id] = 5.0
            
            query_node = HFN(mu=query_mu, sigma=np.ones(self.m_dim), id="falsify_query", use_diag=True)
            candidates = self.retriever.retrieve(query_node, k=5)
            for c in candidates:
                if c.id not in [n.id for n in res.explanation_tree]:
                    # Penalize nodes that were expected but didn't match
                    self.observer.penalize_id(c.id, penalty=0.5)
                    
        return res

# --- 3. EXPERIMENT RUNNER ---

def run_experiment():
    print("--- SP43: Experiment 19 — Adversarial Belief Revision ---\n")
    
    DIM = 10
    env = AdversarialEnvironment(DIM)
    agent = BeliefRevisionAgent(DIM)
    
    # Trackers
    weight_trajectory_C = []
    weight_trajectory_D = []
    
    node_id_C = None
    node_id_D = None
    
    # PHASE 1: Entrenchment (A->B->C)
    print("PHASE 1: Entrenching misleading belief (A->B->C)...")
    env.phase = 1
    for i in range(50):
        env.reset()
        s_t = env.state.copy()
        s_mid = env.step(0).copy() # A -> B
        s_t1 = env.step(0).copy() # B -> C
        
        # We only care about the B->C transition for belief tracking
        res = agent.perceive(s_mid, 0, s_t1)
        
        # Identify node C on first capture
        if node_id_C is None and len(res.explanation_tree) > 0:
            node_id_C = res.explanation_tree[0].id
            
        w_c = agent.observer.get_weight(node_id_C) if node_id_C else 0.0
        weight_trajectory_C.append(w_c)
        weight_trajectory_D.append(0.0) # D doesn't exist yet
        
    print(f"  Phase 1 Complete. Belief C Weight: {weight_trajectory_C[-1]:.4f}")
    
    # PHASE 2: Conflict (A->B->D)
    print("\nPHASE 2: Introducing truth under conflict (A->B->D)...")
    env.phase = 2
    belief_shift_time = None
    residual_conflict_start = None
    residual_conflict_duration = 0
    
    for i in range(100):
        env.reset()
        s_t = env.state.copy()
        s_mid = env.step(0).copy() # A -> B
        s_t1 = env.step(0).copy() # B -> D
        
        res = agent.perceive(s_mid, 0, s_t1)
        
        # Identify node D on first capture
        if node_id_D is None and len(res.explanation_tree) > 0:
            # It must be a new node that explains the transition to D
            # (Delta is [0, 0, 1])
            new_node = res.explanation_tree[0]
            if np.isclose(new_node.mu[2*DIM + 2], 5.0): # Delta D[2] scaled
                node_id_D = new_node.id
        
        w_c = agent.observer.get_weight(node_id_C) if node_id_C else 0.0
        w_d = agent.observer.get_weight(node_id_D) if node_id_D else 0.0
        
        weight_trajectory_C.append(w_c)
        weight_trajectory_D.append(w_d)
        
        # Metrics
        if belief_shift_time is None and w_d > w_c and w_d > 0.01:
            belief_shift_time = i + 1
            print(f"  [SHIFT] Truth (D) surpassed Belief (C) at step {belief_shift_time}")
            
        if w_c > 0.05 and w_d > 0.05:
            if residual_conflict_start is None:
                residual_conflict_start = i
            residual_conflict_duration += 1
            
        if i % 10 == 0:
            print(f"  Step {i:3}: C Weight: {w_c:.4f} | D Weight: {w_d:.4f}")
            
        if w_c < 0.01 and w_d > 0.2:
            print(f"  [CONVERGED] Incorrect belief C suppressed below threshold at step {i+1}")
            break

    # --- RESULTS ---
    print("\n--- PERFORMANCE SUMMARY ---")
    print(f"Belief Shift Time:   {belief_shift_time if belief_shift_time else 'N/A'} steps")
    print(f"Residual Conflict:   {residual_conflict_duration} steps")
    print(f"Final C Weight:      {weight_trajectory_C[-1]:.6f}")
    print(f"Final D Weight:      {weight_trajectory_D[-1]:.6f}")
    
    # Analyze trajectory for failure modes
    max_c_after_shift = max(weight_trajectory_C[50:]) if len(weight_trajectory_C) > 50 else 0
    if weight_trajectory_C[-1] > 0.1:
        print("Result: FAIL - Confirmation Bias (Never abandoned incorrect belief)")
    elif belief_shift_time is None:
        print("Result: FAIL - No shift detected")
    elif residual_conflict_duration > 50:
        print("Result: PARTIAL - High residual conflict")
    else:
        print("Result: SUCCESS - Stable convergence to truth")

    if belief_shift_time and weight_trajectory_C[-1] < 0.01:
        print("\n[SUCCESS] Adversarial Belief Revision Successful.")
        print("The system successfully unlearned the entrenched bias and converged on the truth.")
    else:
        print("\n[FAIL] System failed to fully overcome the early bias.")

if __name__ == "__main__":
    run_experiment()
