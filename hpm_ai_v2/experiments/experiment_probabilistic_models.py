"""
Experiment: Comparing Probabilistic Models (Flat Gaussian vs GMM)

This experiment validates the pluggable probabilistic model architecture in the HFN core.
It demonstrates that an HFN node equipped with a Gaussian Mixture Model (GMM) can effectively
learn a multi-modal concept that a single flat Gaussian cannot represent accurately.

We use a 1-D domain with two separated modes:
- Mode A: centered around 1.0
- Mode B: centered around 9.0

Success is measured by a significant reduction in average residual surprise on test data.
"""
import sys
import numpy as np
from pathlib import Path

# Ensure we can import from project root
sys.path.insert(0, str(Path(__file__).parents[2]))

from hfn.hfn import HFN
from hfn.forest import Forest
from hfn.observer import Observer
from hfn.probabilistic_models import FlatGaussianModel, GaussianMixtureModel


def run_experiment():
    print("=" * 70)
    print("Experiment: Comparing Probabilistic Models (Flat Gaussian vs GMM)")
    print("Goal: Demonstrate learning of a multi-modal concept ('parity')")
    print("=" * 70 + "\n")

    # 1-D Observations: Two separated modes
    # Mode A: centered around 1.0
    # Mode B: centered around 9.0
    train_a = [0.8, 0.9, 1.0, 1.1, 1.2]
    train_b = [8.8, 8.9, 9.0, 9.1, 9.2]
    train_data = [np.array([float(x)]) for x in train_a + train_b]
    # Shuffle training data
    rng = np.random.default_rng(42)
    rng.shuffle(train_data)

    # Test data: held-out values near the modes
    test_a = [0.7, 1.3]
    test_b = [8.7, 9.3]
    test_data = [np.array([float(x)]) for x in test_a + test_b]
    test_labels = ["Mode A"] * 2 + ["Mode B"] * 2

    # Helper to setup a small world model for a node
    def setup_node_context(node):
        forest = Forest(D=1)
        forest.register(node)
        # Set tau high so the node always "explains" the data and gets updated
        obs = Observer(forest, tau=10.0, node_use_diag=True)
        return obs

    # -----------------------------------------------------------------------
    # 1. Flat Gaussian Model
    # -----------------------------------------------------------------------
    print("Scenario 1: Flat Gaussian Model (Single Mode)")
    # Initialize at global mean
    flat_node = HFN(mu=np.array([5.0]), sigma=np.array([1.0]), id="parity_flat", use_diag=True)
    obs_flat = setup_node_context(flat_node)

    # Training: Rely on Observer's native node.update() call
    for x in train_data:
        obs_flat.observe(x)
    
    print(f"  Trained Mean: {flat_node.mu[0]:.2f}")
    
    surprises_flat = []
    for x, label in zip(test_data, test_labels):
        surprise = -flat_node.log_prob(x)
        surprises_flat.append(surprise)
    
    avg_surprise_flat = np.mean(surprises_flat)
    print(f"  Average Test Surprise: {avg_surprise_flat:.4f}\n")

    # -----------------------------------------------------------------------
    # 2. Gaussian Mixture Model (K=2)
    # -----------------------------------------------------------------------
    print("Scenario 2: Gaussian Mixture Model (K=2, Multi-Modal)")
    # Initialise two components near the expected means
    comp1 = FlatGaussianModel(mu=np.array([2.0]), sigma=np.array([1.0]), use_diag=True)
    comp2 = FlatGaussianModel(mu=np.array([8.0]), sigma=np.array([1.0]), use_diag=True)
    mixture = GaussianMixtureModel(components=[comp1, comp2], weights=[0.5, 0.5])
    
    mix_node = HFN(mu=np.array([5.0]), sigma=np.array([1.0]), prob_model=mixture, id="parity_mix", use_diag=True)
    obs_mix = setup_node_context(mix_node)

    # Training: Rely on Observer's native node.update() call
    for x in train_data:
        obs_mix.observe(x)
    
    m1 = mix_node.prob_model.components[0].mu[0]
    m2 = mix_node.prob_model.components[1].mu[0]
    w1 = mix_node.prob_model.weights[0]
    w2 = mix_node.prob_model.weights[1]
    print(f"  Trained Component 1: mu={m1:.2f}, weight={w1:.2f}")
    print(f"  Trained Component 2: mu={m2:.2f}, weight={w2:.2f}")

    surprises_mix = []
    for x, label in zip(test_data, test_labels):
        surprise = -mix_node.log_prob(x)
        surprises_mix.append(surprise)
    
    avg_surprise_mix = np.mean(surprises_mix)
    print(f"  Average Test Surprise: {avg_surprise_mix:.4f}\n")

    # -----------------------------------------------------------------------
    # RESULTS
    # -----------------------------------------------------------------------
    improvement = (avg_surprise_flat - avg_surprise_mix) / avg_surprise_flat * 100
    print("-" * 70)
    print(f"Improvement in expression: {improvement:.2f}% reduction in surprise")
    
    if avg_surprise_mix < avg_surprise_flat:
        print("[SUCCESS] GMM outperformed Flat Gaussian on multi-modal data.")
    else:
        print("[FAIL] GMM did not show improvement.")
    print("-" * 70)


if __name__ == "__main__":
    run_experiment()
