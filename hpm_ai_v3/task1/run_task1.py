import sys, os
sys.path.append(os.path.abspath("hpm_ai_v3"))

import torch
import numpy as np
from task1_data import generate_task1_data, create_torch_dataset
from regression_pattern import RegressionPattern
from population import PatternPopulation
from evaluators import EvaluatorManager
from compiler import SubstrateCompiler
from symbolic_pattern import SymbolicPattern

def train_hpm_agent(train_data, n_epochs=10, batch_size=32):
    X_train, y_train = create_torch_dataset(train_data)
    initial_patterns = [RegressionPattern(input_dim=6, output_dim=1) for _ in range(5)]
    pop = PatternPopulation(initial_patterns)
    eval_mgr = EvaluatorManager()
    compiler = SubstrateCompiler()
    
    losses = []
    for epoch in range(n_epochs):
        perm = torch.randperm(X_train.shape[0])
        for i in range(0, X_train.shape[0], batch_size):
            idx = perm[i:i+batch_size]
            obs = {"input": X_train[idx], "target": y_train[idx]}
            pop.step(eval_mgr, obs, compiler)
        
        # Loss evaluation
        with torch.no_grad():
            top = pop.get_top_patterns(k=1)[0]
            loss = ((top.sample({"input": X_train[:100]})["y"] - y_train[:100])**2).mean().item()
            losses.append(loss)
            if epoch % 5 == 0: print(f"Epoch {epoch}: loss = {loss:.4f}")
    return pop, losses

def predict_from_pop(pop, X):
    preds = []
    for i in range(X.shape[0]):
        obs = {"input": X[i].unsqueeze(0)}
        sorted_pats = sorted(pop.patterns, key=lambda p: p.weight, reverse=True)
        if not sorted_pats: preds.append(0.0); continue
        pred = sorted_pats[0].sample(obs)["y"]
        preds.append(pred.item())
    return torch.tensor(preds)

def evaluate_sensitivity(pop, base_test_data):
    X_base, _ = create_torch_dataset(base_test_data)
    X_surface = X_base.clone()
    X_surface[:, [2, 3]] = X_surface[:, [3, 2]]
    X_surface[:, [4, 5]] = X_surface[:, [5, 4]]
    X_struct = X_base.clone()
    X_struct[:, 0] = -X_struct[:, 0]
    y_struct = []
    for i in range(X_struct.shape[0]):
        a, b = X_struct[i,0].item(), X_struct[i,1].item()
        y_struct.append(a*b + np.sin(a) if a > 0 else a + b)
    y_struct = torch.tensor(y_struct).unsqueeze(1)
    with torch.no_grad():
        y_base = torch.tensor([d[1] for d in base_test_data]).unsqueeze(1)
        pred_base = predict_from_pop(pop, X_base).unsqueeze(1)
        pred_surface = predict_from_pop(pop, X_surface).unsqueeze(1)
        pred_struct = predict_from_pop(pop, X_struct).unsqueeze(1)
    return {
        "surface_delta": ((pred_surface - y_base)**2).mean().item() - ((pred_base - y_base)**2).mean().item(),
        "structural_delta": ((pred_struct - y_struct)**2).mean().item() - ((pred_base - y_base)**2).mean().item()
    }

def main():
    train_data = generate_task1_data(n_samples=2000)
    test_data = generate_task1_data(n_samples=500)
    print("Training HPM agent...")
    hpm_pop, hpm_losses = train_hpm_agent(train_data, n_epochs=20)
    
    sens = evaluate_sensitivity(hpm_pop, test_data)
    print("\nSensitivity to surface vs structural changes:")
    print(f"Surface ΔMSE: {sens['surface_delta']:.4f}")
    print(f"Structural ΔMSE: {sens['structural_delta']:.4f}")

if __name__ == "__main__":
    main()
