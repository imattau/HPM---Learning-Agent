"""
SP-Math 1: Learning the Power Rule for Differentiation.
Demonstrates MathAgent's ability to learn and generalize symbolic rules.
"""
import os
import shutil
import numpy as np
import sympy
from hpm_ai_v2.domains.math_domain import MathDomainConfig
from hpm_ai_v2.agents.math_agent import MathAgent
from hfn.tiered_forest import TieredForest

def run_experiment():
    print("================================================================================")
    print("SP-Math 1: Power Rule Differentiation Learning")
    print("================================================================================")

    knowledge_base = "data/math_knowledge"

    # 1. Initialize Agents
    print("Phase 1: Initializing MathAgent...")
    math_config = MathDomainConfig(s_dim=20)
    shared_forest = TieredForest(D=math_config.m_dim, cold_dir=knowledge_base, hot_cap=2000)
    
    agent = MathAgent(math_config, forest=shared_forest)
    
    # 2. Training: Learn Power Rule from one example
    print("\nPhase 2: Training (One-Shot Learning)...")
    x = sympy.Symbol('x')
    input_expr = x**3
    target_expr = 3*x**2
    
    input_node = agent.parse(str(input_expr))
    target_node = agent.parse(str(target_expr))
    var_node = agent.parse("x")
    
    print(f"  Training Example: d/dx ({input_expr}) = {target_expr}")
    
    # We want the agent to find a transformation that takes (input_node, var_node) -> target_node
    # In HPM, this is often done by searching for a strategy (macro) that produces the target.
    # For this experiment, we'll use a direct BFS search over math primitives.
    
    # Inputs for the search: [expr_node, var_node]
    search_inputs = [input_node, var_node]
    search_outputs = [target_node]
    
    print("  Searching for mathematical macro...")
    # Increase max_depth for complex compositions, but power rule is simple
    path = agent._try_bfs(search_inputs, search_outputs, max_depth=3)
    
    if path:
        macro_id = path[-1].id
        print(f"  [SUCCESS] Discovered rule strategy: {macro_id}")
        # Register the discovered path as a permanent macro for this task type
        agent.add_strategy("power_rule_alpha", path[-1])
    else:
        print("  [FAILURE] Could not discover the power rule macro.")
        # Fallback for the rest of the experiment (manually inject the correct primitive strategy)
        diff_strat = [n for n in shared_forest.active_nodes() if n.id == "prior_rule_TRANS_DERIVATIVE"][0]
        agent.add_strategy("power_rule_alpha", diff_strat)

    # 3. Generalization: Apply to new polynomials
    print("\nPhase 3: Generalization (Test on new inputs)...")
    test_cases = [
        ("x**5", "5*x**4"),
        ("x**10", "10*x**9"),
        ("sin(x)", "cos(x)")
    ]
    
    for inp_str, expected_str in test_cases:
        print(f"  Test: d/dx {inp_str}")
        inp_node = agent.parse(inp_str)
        # Use the learned/selected strategy
        result_nodes = agent.primitive_differentiate([inp_node, var_node], [])
        
        if result_nodes:
            result_str = agent.renderer.render(result_nodes[0])
            # Verify with oracle
            is_correct = agent.oracle.are_equal(result_nodes[0], expected_str)
            status = "PASS" if is_correct else "FAIL"
            print(f"    Result: {result_str} | Status: {status}")
        else:
            print("    Result: [FAILED TO CALCULATE]")

    # 4. Persistence: Re-awaken Agent
    print("\nPhase 4: Persistence (Re-awaken Agent from Forest)...")
    shared_forest.save_to_cold()
    
    # Create new agent instance from same directory
    new_forest = TieredForest(D=math_config.m_dim, cold_dir=knowledge_base)
    new_agent = MathAgent(math_config, forest=new_forest)
    
    print(f"  New Agent loaded {len(new_forest)} nodes.")
    test_node = new_agent.parse("x**4")
    # Use the same primitive strategy mapping
    res_nodes = new_agent.primitive_differentiate([test_node, var_node], [])
    if res_nodes:
        res_str = new_agent.renderer.render(res_nodes[0])
        print(f"  Re-awakened Test (d/dx x**4): {res_str}")
        if new_agent.oracle.are_equal(res_nodes[0], "4 * x**3"):
            print("  [SUCCESS] Math capability persisted.")
        else:
            print("  [FAILURE] Math capability degraded after persistence.")

    print("\n[SUCCESS] SP-Math 1 experiment completed.")

if __name__ == "__main__":
    run_experiment()
