import numpy as np
from typing import List, Dict
from copy import deepcopy
from hpm_ai_v6.hpm_model.core.cell import Cell
from hpm_ai_v6.hpm_model.agents.social_agent import SocialAgent
from hpm_ai_v6.hpm_model.fields.pattern_field import DynamicPatternField
from hpm_ai_v6.hpm_model.environments.grammar_env import GrammarEnvironment
from hpm_ai_v6.hpm_model.environments.noisy_env import NoisyEnvironment

def run_social_convergence_experiment():
    """
    Production implementation of Prediction 9.5: 
    Social fields accelerate pattern convergence.
    """
    np.random.seed(42)
    
    # 1. Setup
    objs = [Cell(name=f"o{i}", embedding=np.random.randn(2)) for i in range(3)]
    base_pats = [
        Cell(name="A_to_B", dim=1, embedding=np.random.randn(2), source=objs[0], target=objs[1]),
        Cell(name="B_to_C", dim=1, embedding=np.random.randn(2), source=objs[1], target=objs[2]),
        Cell(name="C_to_A", dim=1, embedding=np.random.randn(2), source=objs[2], target=objs[0])
    ]
    
    n_agents = 5
    n_episodes = 20
    
    # Condition 1: Social (Shared Field)
    shared_field = DynamicPatternField(influence_rate=0.2)
    social_agents = [SocialAgent(patterns=deepcopy_patterns(base_pats), 
                               shared_field=shared_field,
                               gamma_soc=0.8) for _ in range(n_agents)]
    
    # Condition 2: Individual (No Shared Field)
    individual_agents = [SocialAgent(patterns=deepcopy_patterns(base_pats), 
                                   shared_field=None,
                                   gamma_soc=0.1) for _ in range(n_agents)]
    
    env = NoisyEnvironment(GrammarEnvironment(objs), noise_level=0.2, all_objects=objs)
    
    def get_variance(agents):
        n_pats = len(base_pats)
        vars = []
        for i in range(n_pats):
            weights = [a.get_weights()[i] for a in agents]
            vars.append(np.var(weights))
        return np.mean(vars)

    results = {"Social": [], "Individual": []}
    
    print("Running Social Convergence Experiment...")
    for ep in range(n_episodes):
        seq = env.generate_episode(15)
        consensus_vec = np.mean([a.meta_rule.get_best_pattern().embedding for a in social_agents], axis=0)
        
        # Social Loop
        for a in social_agents:
            a.perceive(seq, objs, context={"consensus_vec": consensus_vec})
        # Field Update
        all_weights = [a.meta_rule.get_weights_dict() for a in social_agents]
        # Hack to update field manually in this test runner
        social_scores = [a.learner.social.evaluate(a.get_best_pattern(), {"consensus_vec": consensus_vec}).score for a in social_agents]
        shared_field.update(social_agents[0].patterns, np.mean([a.get_weights() for a in social_agents], axis=0), 
                          np.ones(len(base_pats)) * np.mean(social_scores))
        
        # Individual Loop
        for a in individual_agents:
            a.perceive(seq, objs, context={"consensus_vec": np.zeros(2)})
            
        results["Social"].append(get_variance(social_agents))
        results["Individual"].append(get_variance(individual_agents))
        
        if ep % 5 == 0:
            print(f"  Episode {ep:2} | Social Var: {results['Social'][-1]:.6f} | Individual Var: {results['Individual'][-1]:.6f}")

    print("\nVerification of Prediction 9.5:")
    if results["Social"][-1] < results["Individual"][-1]:
        print("✅ Prediction 9.5 Supported: Social agents converged more than individual agents.")
    else:
        print("⚠️ Prediction 9.5 Not Supported. Adjust field influence or social weight.")

def deepcopy_patterns(patterns: List[Cell]) -> List[Cell]:
    return [Cell(name=p.name, dim=p.dim, embedding=p.embedding.copy(), 
                 source=p.source, target=p.target) for p in patterns]

if __name__ == "__main__":
    run_social_convergence_experiment()
