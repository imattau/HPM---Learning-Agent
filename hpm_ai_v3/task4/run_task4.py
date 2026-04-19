import sys, os
sys.path.append(os.path.abspath("hpm_ai_v3"))

import torch, numpy as np, matplotlib.pyplot as plt
from composable_pattern import ComposableRegressionPattern
from recombination import RecombinationOperator
from population import PatternPopulation
from evaluators import EvaluatorManager
from compiler import SubstrateCompiler
from task4_data import *

def run_experiment(use_rec=True):
    pop = PatternPopulation([ComposableRegressionPattern(input_dim=1, output_dim=1) for _ in range(3)])
    eval_mgr, compiler = EvaluatorManager(), SubstrateCompiler()
    
    for gen in [generate_primitive_A, generate_primitive_B]:
        for _ in range(1000):
            x, y = gen(1)
            pop.step(eval_mgr, {"input": torch.tensor(x), "target": torch.tensor(y)}, compiler)
            
    if use_rec: pop.recombination_prob = 0.5
    else: pop.recombination_prob = 0.0
    
    mse_hist = []
    for step in range(500):
        x, y = generate_composition_C(1)
        pop.step(eval_mgr, {"input": torch.tensor(x), "target": torch.tensor(y)}, compiler)
        if step % 20 == 0:
            preds = np.array([p.sample({"input": torch.tensor(x).float().unsqueeze(0)})["y"].item() for p in pop.get_top_patterns(1)])
            mse_hist.append(np.mean((preds.mean() - y)**2))
    return mse_hist

res = run_experiment(True), run_experiment(False)
plt.plot(res[0], label='HPM+Rec'); plt.plot(res[1], label='HPM-NoRec')
plt.legend(); plt.savefig('task4_creativity.png')
print("Task 4 complete.")
