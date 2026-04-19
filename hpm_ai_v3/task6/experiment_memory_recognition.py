"""
experiment_memory_recognition.py - Memory-Augmented Recognition Task for HPM agent.
"""

import sys, os
sys.path.append(os.path.abspath("hpm_ai_v3"))

import torch
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional, Any
import random
from tqdm import tqdm

from hpm_ai_v3.tool_registry import ToolRegistry
from hpm_ai_v3.augmented_agent import AugmentedHPMAgent
from hpm_ai_v3.perception_tools import extract_resnet_features, register_perception_tools
from hpm_ai_v3.memory_tools import vector_store, vector_search, episodic_append, register_memory_tools
from hpm_ai_v3.classification_pattern import ClassificationPattern

# ----------------------------------------------------------------------
# Data Generator: Mock CIFAR-10 if download fails
# ----------------------------------------------------------------------
def create_repetition_stream(dataset_size: int = 100, 
                             repeat_prob: float = 0.3,
                             min_lag: int = 5,
                             max_lag: int = 20,
                             seed: int = 42) -> List[Tuple[torch.Tensor, int, bool]]:
    """
    Generate a stream of images with controlled repetitions.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        from torchvision import datasets, transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        full_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        use_mock = False
    except Exception as e:
        print(f"Warning: Could not load CIFAR-10 ({e}). Using mock data.")
        use_mock = True
    
    stream = []
    seen_indices = []
    
    if use_mock:
        # Generate random tensors as mock images
        mock_images = [torch.randn(3, 32, 32) for _ in range(dataset_size)]
        mock_labels = [random.randint(0, 9) for _ in range(dataset_size)]
        
        for _ in range(dataset_size):
            if len(seen_indices) > min_lag and random.random() < repeat_prob:
                lag = random.randint(min_lag, min(max_lag, len(seen_indices)))
                repeat_idx = seen_indices[-lag]
                img = mock_images[repeat_idx]
                label = mock_labels[repeat_idx]
                is_repeat = True
            else:
                idx = random.randint(0, dataset_size - 1)
                img = mock_images[idx]
                label = mock_labels[idx]
                seen_indices.append(idx)
                is_repeat = False
            stream.append((img, label, is_repeat))
    else:
        idx_count = 0
        for _ in range(dataset_size):
            if len(seen_indices) > min_lag and random.random() < repeat_prob:
                lag = random.randint(min_lag, min(max_lag, len(seen_indices)))
                repeat_idx = seen_indices[-lag]
                img, label = full_dataset[repeat_idx]
                is_repeat = True
            else:
                img, label = full_dataset[idx_count]
                seen_indices.append(idx_count)
                idx_count = (idx_count + 1) % len(full_dataset)
                is_repeat = False
            stream.append((img, label, is_repeat))
    
    return stream


# ----------------------------------------------------------------------
# Environment Wrapper
# ----------------------------------------------------------------------
class MemoryTaskEnvironment:
    def __init__(self, stream: List[Tuple[torch.Tensor, int, bool]]):
        self.stream = stream
        self.position = 0
        self.correct_classifications = 0
        self.correct_repeat_detections = 0
        self.total_repeats = 0
        
    def reset(self):
        self.position = 0
        self.correct_classifications = 0
        self.correct_repeat_detections = 0
        self.total_repeats = 0
        
    def step(self, agent_prediction: Dict) -> Dict:
        if self.position >= len(self.stream):
            return {"done": True}
        
        img, true_label, is_repeat = self.stream[self.position]
        
        reward = 0.0
        predicted_label = agent_prediction.get("predicted_label")
        predicted_repeat = agent_prediction.get("predicted_repeat", False)
        
        if predicted_label is not None:
            if predicted_label == true_label:
                reward += 1.0
                self.correct_classifications += 1
            else:
                reward -= 0.5
        
        if is_repeat:
            self.total_repeats += 1
            if predicted_repeat:
                reward += 0.5
                self.correct_repeat_detections += 1
            else:
                reward -= 0.2
        else:
            if not predicted_repeat:
                reward += 0.1
                
        self.position += 1
        
        return {
            "image": img,
            "true_label": true_label,
            "is_repeat": is_repeat,
            "reward": reward,
            "done": self.position >= len(self.stream)
        }
    
    def get_stats(self) -> Dict:
        return {
            "classification_accuracy": self.correct_classifications / self.position if self.position > 0 else 0,
            "repeat_detection_accuracy": self.correct_repeat_detections / self.total_repeats if self.total_repeats > 0 else 0,
            "position": self.position
        }


# ----------------------------------------------------------------------
# Main Experiment
# ----------------------------------------------------------------------
def run_memory_experiment():
    print("=== HPM Memory-Augmented Recognition Experiment ===\n")
    
    # Setup tools
    register_perception_tools()
    register_memory_tools()
    
    # Generate stream
    dataset_size = 200 # Reduced for speed
    stream = create_repetition_stream(dataset_size=dataset_size, repeat_prob=0.3)
    env = MemoryTaskEnvironment(stream)
    
    # Create agent
    agent = AugmentedHPMAgent(
        tool_names=["extract_features", "vector_search", "vector_store"]
    )
    
    # We'll use a simple threshold for memory recognition in this demo
    memory_threshold = 0.5
    
    rewards = []
    print(f"Running {dataset_size} steps...")
    
    for i in range(dataset_size):
        img, true_label, is_repeat = stream[i]
        
        # Step 1: Agent decides tool
        # In a full run, meta-orchestrator learns this.
        # Here we demonstrate the agent using tools sequentially.
        
        context = {"image": img}
        
        # Try to find in memory first
        features = extract_resnet_features(img)
        search_results = vector_search(features, collection="recognition_memory", top_k=1)
        
        predicted_repeat = False
        predicted_label = None
        
        if search_results["results"]:
            best = search_results["results"][0]
            if best["distance"] < memory_threshold:
                predicted_repeat = True
                predicted_label = best["metadata"].get("label")
        
        if predicted_label is None:
            # Fallback to a base classifier (mock for this script)
            predicted_label = random.randint(0, 9)
            # Store features in memory for future
            vector_store(features, {"label": true_label}, collection="recognition_memory")
            
        # Step 2: Environment provides reward
        res = env.step({
            "predicted_label": predicted_label,
            "predicted_repeat": predicted_repeat
        })
        
        rewards.append(res["reward"])
        
        if i % 50 == 0:
            stats = env.get_stats()
            print(f"  Step {i}: Reward={res['reward']:.2f}, ClassAcc={stats['classification_accuracy']:.2f}, RepeatAcc={stats['repeat_detection_accuracy']:.2f}")

    print("\nFinal Stats:")
    stats = env.get_stats()
    print(f"  Total Steps: {stats['position']}")
    print(f"  Classification Accuracy: {stats['classification_accuracy']:.3f}")
    print(f"  Repeat Detection Accuracy: {stats['repeat_detection_accuracy']:.3f}")
    print(f"  Average Reward: {np.mean(rewards):.3f}")

if __name__ == "__main__":
    run_memory_experiment()
