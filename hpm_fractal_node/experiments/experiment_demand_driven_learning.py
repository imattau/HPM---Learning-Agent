"""
Experiment: Demand-Driven Learning (SP24).
Tests the "Fail-Learn-Retry" loop via a ResolutionRequest from the Decoder.
"""
from __future__ import annotations
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from hfn.hfn import HFN
from hfn.forest import Forest
from hfn.decoder import Decoder, ResolutionRequest
from hfn.observer import Observer

# --- Constants ---
D = 1

def execute_goal(decoder: Decoder, observer: Observer, goal: HFN, buffer: list[float], max_retries: int = 3):
    for attempt in range(max_retries):
        print(f"  Attempt {attempt + 1}...")
        result = decoder.decode(goal)
        
        if isinstance(result, list):
            print(f"  -> SUCCESS! Output: {[n.id for n in result]}")
            return result
            
        elif isinstance(result, ResolutionRequest):
            print(f"  -> STALL: Missing node near mu={result.missing_mu} with edges: {[e.relation for e in result.required_edges]}")
            print(f"  -> Triggering Observer scan...")
            
            found = False
            for obs in buffer:
                # If observation matches request mu closely enough (e.g. within 0.5)
                if np.linalg.norm(obs - result.missing_mu) < 0.5:
                    print(f"  -> OBSERVER FOUND EVIDENCE: {obs}. Creating new node.")
                    
                    # 1. Create concrete leaf
                    leaf_id = f"leaf_discovered_{int(obs)}"
                    new_leaf = HFN(mu=np.array([obs]), sigma=np.array([0.001]), id=leaf_id, use_diag=True)
                    
                    # 2. Apply requested constraints (Binding)
                    for edge in result.required_edges:
                        new_leaf.add_edge(new_leaf, edge.target, edge.relation)
                        
                    # 3. Register in Forest
                    observer.register(new_leaf)
                    found = True
                    break
                    
            if not found:
                print(f"  -> OBSERVER FAILED: No historical evidence for mu={result.missing_mu}. Blocked Hallucination.")
                return "FAILURE: Ungrounded Request"
                
    print("  -> FAILURE: Max Retries Exceeded.")
    return "FAILURE: Max Retries Exceeded"


def run_experiment():
    print("SP24: Demand-Driven Learning Experiment\n")

    # --- 1. Manifold Setup ---
    target_forest = Forest(D=D, forest_id="hand")
    # Priors: We know Red and Blue. We DON'T know Green.
    x2 = HFN(mu=np.array([2.0]), sigma=np.array([0.0001]), id="pos_red", use_diag=True)
    x5 = HFN(mu=np.array([5.0]), sigma=np.array([0.0001]), id="pos_blue", use_diag=True)
    for p in [x2, x5]: target_forest.register(p)

    concept_red = HFN(mu=np.zeros(D), sigma=np.zeros(D), id="COLOR_RED")
    concept_blue = HFN(mu=np.zeros(D), sigma=np.zeros(D), id="COLOR_BLUE")
    concept_green = HFN(mu=np.zeros(D), sigma=np.zeros(D), id="COLOR_GREEN")
    concept_yellow = HFN(mu=np.zeros(D), sigma=np.zeros(D), id="COLOR_YELLOW")

    x2.add_edge(x2, concept_red, "HAS_COLOR")
    x5.add_edge(x5, concept_blue, "HAS_COLOR")

    decoder = Decoder(target_forest=target_forest, sigma_threshold=0.01)
    # The observer acts as the curiosity engine, managing the same forest
    observer = Observer(forest=target_forest, tau=1.0) 

    # --- 2. Historical Buffer ---
    # The system has "seen" these coordinates in the past, but ignored them
    # because it had no active goals related to them.
    historical_buffer = [2.1, 4.9, 8.0, 8.2] # 8.0 is where the Green block actually is

    # --- 3. Test Cases ---
    
    print("--- Test 1: The Green Block (Valid Knowledge Gap) ---")
    # Goal: Point to the Green Block.
    goal_green = HFN(mu=np.array([8.1]), sigma=np.array([0.5]), id="goal_find_green")
    goal_green.add_edge(goal_green, concept_green, "HAS_COLOR")
    
    execute_goal(decoder, observer, goal_green, historical_buffer)
    
    print("\n--- Test 2: The Yellow Block (Hallucination Guard) ---")
    # Goal: Point to the Yellow Block at X=10.0
    # But X=10.0 has never been observed in reality (not in buffer).
    goal_yellow = HFN(mu=np.array([10.0]), sigma=np.array([0.5]), id="goal_find_yellow")
    goal_yellow.add_edge(goal_yellow, concept_yellow, "HAS_COLOR")
    
    execute_goal(decoder, observer, goal_yellow, historical_buffer)

if __name__ == "__main__":
    run_experiment()
