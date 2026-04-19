import sys, os; sys.path.append(os.path.abspath("hpm_ai_v3"))
import torch, numpy as np, matplotlib.pyplot as plt
from task6.oracle_tool_pattern import OracleToolPattern
from task6.meta_tool_pattern import MetaToolPattern
from task6.task6_data import *
from task1.regression_pattern import RegressionPattern
from population import PatternPopulation
from evaluators import EvaluatorManager
from compiler import SubstrateCompiler

def run_meta():
    pop = PatternPopulation([RegressionPattern(1,1) for _ in range(3)])
    tool = OracleToolPattern(hard_function)
    meta = MetaToolPattern()
    eval_mgr, compiler = EvaluatorManager(), SubstrateCompiler()
    
    errors, decisions = [], []
    for step in range(500):
        x, y = generate_task6_data(1)
        obs = {"input": torch.tensor(x), "target": torch.tensor(y)}
        feat = torch.tensor([x[0,0], 0.1, 1-step/500]).float()
        
        use_tool = meta.sample({"features": feat})["use_tool"].item()
        decisions.append(use_tool)
        pred = tool.sample(obs)["y"] if use_tool else sum(p.sample(obs)["y"] for p in pop.get_top_patterns(1))/1
        
        mse = (pred.item() - y[0,0])**2
        errors.append(mse)
        meta.update_parameters({"features": feat.unsqueeze(0), "reward": torch.tensor([-(mse + (0.3 if use_tool else 0))]), "use_tool_taken": torch.tensor([use_tool])})
        if not use_tool: pop.step(eval_mgr, obs, compiler)
    return errors, decisions

res = run_meta()
plt.figure(); plt.scatter(range(500), res[0], c=['red' if d else 'blue' for d in res[1]], s=2); plt.savefig('task6_tool_use.png')
print("Task 6 complete.")
