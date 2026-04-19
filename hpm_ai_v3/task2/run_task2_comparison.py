import sys, os
sys.path.append(os.path.abspath("hpm_ai_v3"))

import torch
import numpy as np
import matplotlib.pyplot as plt
from neural_pattern import RegressionPattern
from pattern_field import PatternField
from population import PatternPopulation
from evaluators import EvaluatorManager
from compiler import SubstrateCompiler
from task2_data import generate_task2_data

def run_isolated_agent(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    pop = PatternPopulation([RegressionPattern(input_dim=2) for _ in range(3)])
    eval_mgr, compiler = EvaluatorManager(), SubstrateCompiler()
    importance_history = []
    
    for spurious in [True]*50 + [False]*50:
        obs = generate_task2_data(1, spurious_active=spurious)[0]
        obs_dict = {"input": torch.tensor(obs[0]).unsqueeze(0), "target": torch.tensor([obs[1]]).unsqueeze(0)}
        pop.step(eval_mgr, obs_dict, compiler)
        top = sorted(pop.patterns, key=lambda p: p.weight, reverse=True)[0]
        importance_history.append(top.fc_x_to_z1.weight.data.abs().mean(dim=0)[1].item())
    return importance_history

def run_field_agents(num_agents=5, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    field = PatternField(num_agents=num_agents, pattern_factory=lambda: RegressionPattern(input_dim=2))
    importance_history = []
    
    for spurious in [True]*50 + [False]*50:
        obs = [generate_task2_data(1, spurious_active=spurious)[0] for _ in range(num_agents)]
        field.step_field([{"input": torch.tensor(o[0]), "target": torch.tensor([o[1]])} for o in obs])
        avg_imp = 0.0
        for agent in field.agents:
            tops = agent.get_top_patterns(k=1)
            if tops: avg_imp += tops[0].fc_x_to_z1.weight.data.abs().mean(dim=0)[1].item()
        importance_history.append(avg_imp / num_agents)
    return importance_history

if __name__ == "__main__":
    seeds = [42, 43, 44]
    iso = [run_isolated_agent(s) for s in seeds]
    fld = [run_field_agents(num_agents=5, seed=s) for s in seeds]
    
    iso_avg, fld_avg = np.mean(iso, axis=0), np.mean(fld, axis=0)
    iso_std, fld_std = np.std(iso, axis=0), np.std(fld, axis=0)
    
    steps = np.arange(100)
    plt.figure(figsize=(10, 6))
    plt.plot(steps, iso_avg, label='Isolated Agents', color='red')
    plt.fill_between(steps, iso_avg-iso_std, iso_avg+iso_std, alpha=0.2, color='red')
    plt.plot(steps, fld_avg, label='Pattern Field (Strengthened)', color='blue')
    plt.fill_between(steps, fld_avg-fld_std, fld_avg+fld_std, alpha=0.2, color='blue')
    plt.axvline(x=50, linestyle='--', color='gray', label='Spurious correlation removed')
    plt.xlabel('Training Step')
    plt.ylabel('Spurious Feature Importance')
    plt.title('HPM Task 2: Institutional Pressure Reduces Superstition')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('task2_institutional_effect.png')
    print("Done. Saved to task2_institutional_effect.png")
