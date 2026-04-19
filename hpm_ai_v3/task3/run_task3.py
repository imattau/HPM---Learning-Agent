import sys, os
sys.path.append(os.path.abspath("hpm_ai_v3"))

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from neural_pattern import RegressionPattern
from population import PatternPopulation
from evaluators import EvaluatorManager
from compiler import SubstrateCompiler
from task3.meta_pattern import MetaPattern
from task3.task3_environment import NonStationaryEnvironment

def compute_history(loss_history, entropy_history):
    losses = np.array(list(loss_history))
    if len(losses) < 10: return np.zeros(4)
    return np.array([np.mean(losses), np.var(losses), (losses[-1]-losses[0])/10, np.mean(list(entropy_history))])

def run_agent(condition, env, meta=None):
    pop = PatternPopulation([RegressionPattern(input_dim=2) for _ in range(3)])
    eval_mgr, compiler = EvaluatorManager(), SubstrateCompiler()
    loss_hist, ent_hist = deque(maxlen=50), deque(maxlen=50)
    errors, deltas, regimes, total_regret = [], [], [], 0.0
    
    env.reset()
    while True:
        x, y, info = env.step()
        if x is None: break
        obs = {"input": torch.tensor(x).float().unsqueeze(0), "target": torch.tensor([y]).float().unsqueeze(0)}
        
        if condition == 'meta':
            h = compute_history(loss_hist, ent_hist)
            delta = meta.sample({"history": torch.tensor(h).float().unsqueeze(0)})["delta_cur"].item()
            delta = np.clip(delta, 0.05, 0.95)
        else: delta = 0.8 if condition == 'fixed_high' else 0.1
        
        for p in pop.patterns: p.curiosity_reward = delta
        pop.step(eval_mgr, obs, compiler)
        
        mse = ((sum(p.sample({"input": obs["input"]})["y"] for p in pop.get_top_patterns())/3 - obs["target"])**2).item()
        errors.append(mse)
        deltas.append(delta)
        regimes.append(1 if info['regime'] == 'volatile' else 0)
        loss_hist.append(mse)
        ent_hist.append(-sum(p.weight*np.log(p.weight+1e-8) for p in pop.patterns))
        
        if condition == 'meta':
            meta.update_parameters({"history": torch.tensor(compute_history(loss_hist, ent_hist)).float().unsqueeze(0),
                                   "reward": torch.tensor([-mse]).float(), "delta_cur_taken": torch.tensor([delta]).float()})
    return errors, deltas, regimes

env = NonStationaryEnvironment()
meta = MetaPattern()
res = {c: run_agent(c, env, meta) for c in ['fixed_low', 'fixed_high', 'meta']}

fig, axes = plt.subplots(3, 1, figsize=(10, 8))
for c in ['fixed_low', 'fixed_high', 'meta']:
    axes[0].plot(res[c][0], label=c)
axes[0].set_ylabel('MSE'); axes[0].legend()
axes[1].plot(res['meta'][1], label='Meta-Curiosity'); axes[1].set_ylabel('Weight')
axes[2].fill_between(range(len(res['meta'][2])), 0, res['meta'][2], alpha=0.3)
plt.savefig('task3_meta_learning.png')
print("Task 3 experiment complete. Plot saved.")
