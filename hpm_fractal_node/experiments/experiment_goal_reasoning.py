"""
SP33: Experiment 9 — Goal-Conditioned Reasoning (Agency)

Validates the transition from passive learner to intent-driven agent.
Setup:
1. Populate Forest with diverse transformation rules.
2. Define Goal: Transform Input A to Target B.
3. Compare:
   - Passive Baseline: Search by Input A (no intent).
   - Goal-Conditioned Agent: Search by Delta (B - A) using GoalConditionedRetriever.

Metric:
- Retrieval Efficiency: Rank of the 'correct' rule in results.
- Success Rate: Does the agent find the rule that satisfies the goal?
"""
import numpy as np
import time
from hfn.hfn import HFN
from hfn.forest import Forest
from hfn.observer import Observer
from hfn.retriever import GeometricRetriever, GoalConditionedRetriever
from hfn.evaluator import Evaluator
from hfn.decoder import Decoder

def generate_transformation_forest(dim=100, n_rules=20):
    """Creates a forest of synthetic transformation rules with ambiguity."""
    forest = Forest()
    rng = np.random.RandomState(42)
    rules = []
    
    # Manifold: [Input(50), Delta(50)]
    for i in range(n_rules):
        mu = np.zeros(dim)
        group = i // 5
        mu[:50] = rng.normal(group * 10.0, 0.1, size=50)
        mu[50:] = rng.normal(i, 0.5, size=50)
        
        # CONCRETE RULE: Low sigma (0.0001) allows decoder to return it as a solution
        rule = HFN(mu=mu, sigma=np.ones(dim)*0.0001, id=f"rule_{i}", use_diag=True)
        forest.register(rule)
        rules.append(rule)
        
    return forest, rules

def run_experiment():
    print("--- SP33: Experiment 9 — Goal-Conditioned Reasoning ---\n")
    
    DIM = 100
    N_RULES = 50
    forest, rules = generate_transformation_forest(DIM, N_RULES)
    decoder = Decoder(target_forest=forest)
    
    # Pick a random rule to test
    target_rule_idx = 12
    target_rule = rules[target_rule_idx]
    
    A = target_rule.mu[:50] + np.random.normal(0, 0.01, size=50)
    Delta = target_rule.mu[50:]
    B = A + Delta 
    
    goal_mu = np.zeros(DIM)
    goal_mu[:50] = A
    goal_mu[50:] = Delta
    goal_query = HFN(mu=goal_mu, sigma=np.ones(DIM), id="goal_query", use_diag=True)
    
    # 1. Goal-Conditioned Agent Setup
    goal_retriever = GoalConditionedRetriever(forest, target_slice=slice(50, 100), target_weight=50.0)
    
    # 2. Planning Loop: Intent-driven search + Verification
    print(f"Goal: Transform A -> B. Seeking rule that satisfies delta.")
    
    candidates = goal_retriever.retrieve(goal_query, k=10)
    
    steps = 0
    success = False
    winning_rule = None
    
    for rule in candidates:
        steps += 1
        # EXECUTE: We believe this rule satisfies the goal.
        # We synthesize a 'Plan Result' that combines Input A with this Rule.
        
        # Agnostic Plan: Input + Rule Constraints
        plan_mu = np.zeros(DIM)
        plan_mu[:50] = A
        plan_mu[50:] = rule.mu[50:] # Apply rule's delta to our plan
        
        plan_node = HFN(mu=plan_mu, sigma=np.ones(DIM)*0.0001, id="plan_node", use_diag=True)
        
        # VALIDATE: Does this plan satisfy the target B?
        pred_B = plan_node.mu[:50] + plan_node.mu[50:] # A + Delta
        dist = np.linalg.norm(pred_B - B)
        
        print(f"  [DEBUG] Step {steps}: Rule {rule.id} -> dist={dist:.4f}")
        if dist < 0.5:
            success = True
            winning_rule = rule
            break
                
    # 3. Report
    print(f"\nFinal Result:")
    print(f"  Success:      {success}")
    print(f"  Steps Taken:  {steps}")
    if winning_rule:
        print(f"  Winning Rule: {winning_rule.id} (Rank {steps})")
    
    if success and steps <= 3:
        print("\n[SUCCESS] The agent efficiently solved the goal using intent-driven planning.")
    elif success:
        print("\n[PARTIAL] Goal solved, but required multiple planning iterations.")
    else:
        print("\n[FAIL] Agent failed to satisfy the goal.")

if __name__ == "__main__":
    run_experiment()
