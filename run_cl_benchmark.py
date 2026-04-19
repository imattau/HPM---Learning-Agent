import sys, os
sys.path.append(os.path.abspath("hpm_ai_v3"))

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import time

from hpm_ai_v3.device_utils import get_device, set_torch_threads
from hpm_ai_v3.pattern import HPMPattern
from hpm_ai_v3.classification_pattern import ClassificationPattern
from hpm_ai_v3.population import PatternPopulation
from hpm_ai_v3.evaluators import EvaluatorManager
from hpm_ai_v3.compiler import SubstrateCompiler
from data.permuted_mnist import get_permuted_mnist, generate_permutations

def evaluate_accuracy(pop, loader, device):
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        for i in range(x.size(0)):
            obs = {"input": x[i].unsqueeze(0)}
            top_patterns = pop.get_top_patterns(k=1)
            if not top_patterns: continue
            pred = top_patterns[0].predict_class(x[i].unsqueeze(0))
            if pred.item() == y[i].item(): correct += 1
            total += 1
    return correct / total if total > 0 else 0.0

def run_hpm_cl(n_tasks=3, epochs_per_task=5, population_size=5):
    device = get_device()
    HPMPattern.set_device(device)
    set_torch_threads()
    
    patterns = [ClassificationPattern(z1_dim=64, z2_dim=16).to(device) for _ in range(population_size)]
    pop = PatternPopulation(patterns, eta=0.1, beta_c=0.01)
    eval_mgr = EvaluatorManager()
    compiler = SubstrateCompiler(use_gp=False)
    
    permutations = generate_permutations(n_tasks)
    acc_matrix = np.zeros((n_tasks, n_tasks))
    
    for task_id in range(n_tasks):
        print(f"Task {task_id+1}/{n_tasks}")
        train_loader = get_permuted_mnist(task_id, permutations[task_id], batch_size=128, train=True)
        for epoch in range(epochs_per_task):
            for step, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                # Use full batch for each HPM step
                obs = {"input": x, "target": y}
                pop.step(eval_mgr, obs, compiler)
                if step > 200: break # Fewer steps but larger batches
        
        for eval_task in range(task_id + 1):
            test_loader = get_permuted_mnist(eval_task, permutations[eval_task], batch_size=100, train=False)
            acc = evaluate_accuracy(pop, test_loader, device)
            acc_matrix[task_id, eval_task] = acc
            print(f"  Accuracy on task {eval_task+1}: {acc:.4f}")
            
    avg_acc = np.mean(acc_matrix[-1, :n_tasks])
    forgetting = np.mean([max(acc_matrix[t:, t]) - acc_matrix[-1, t] for t in range(n_tasks)])
    print(f"\nFinal Avg Accuracy: {avg_acc:.4f}")
    print(f"Final Avg Forgetting: {forgetting:.4f}")

if __name__ == "__main__":
    run_hpm_cl()
