"""Mock task generator for the DS-1000 Benchmark (Sub-project 11)."""

import numpy as np

def generate_ds1000_tasks(n_per_library: int = 20, seed: int = 42) -> list[dict]:
    """
    Generate mock DS-1000 tasks covering all 7 pillars.
    Each task has 'train' examples (for building the prototype) and 'candidates'
    for test-time discrimination, simulating API sequences or code blocks.
    """
    rng = np.random.default_rng(seed)
    tasks = []
    
    libraries = [
        "pandas", "numpy", "matplotlib", "scipy", 
        "scikit-learn", "pytorch", "tensorflow"
    ]
    
    for lib in libraries:
        for i in range(n_per_library):
            # Simulate different transformations based on library
            if lib == "pandas":
                transform_types = ["impute_nan", "groupby_sum", "merge", "pivot_table", "rolling_mean"]
            elif lib == "numpy":
                transform_types = ["normalize", "dot_product", "reshape", "broadcast_add", "fft"]
            elif lib == "matplotlib":
                transform_types = ["scatter_plot", "histogram", "heatmap", "subplot_layout"]
            elif lib == "scipy":
                transform_types = ["optimize_minimize", "signal_filter", "sparse_solve", "stat_test"]
            elif lib == "scikit-learn":
                transform_types = ["fit_predict", "train_test_split", "scale", "feature_selection"]
            elif lib == "pytorch":
                transform_types = ["tensor_view", "grad_update", "conv2d_forward", "gather_indices"]
            elif lib == "tensorflow":
                transform_types = ["dataset_map", "eager_exec", "layer_call", "weight_assign"]
            else:
                transform_types = ["generic_op"]
                
            transform = rng.choice(transform_types)
            
            # Create a mock task structure
            task = {
                "task_id": f"ds1000_{lib}_{i}",
                "library": lib,
                "prompt": f"Apply {transform} using {lib}",
                "transform": transform,
                "train": [],
                "candidates": []
            }
            
            # Generate 3-5 training pairs (input structural features -> output structural features)
            n_train = rng.integers(3, 6)
            for _ in range(n_train):
                # We mock features as float arrays for the encoders to digest
                input_feat = rng.uniform(0, 1, size=8)
                # Output feature is a deterministically shifted version to simulate the transformation
                shift = rng.uniform(0.1, 0.5) if "impute" in transform or "normalize" in transform else rng.uniform(-0.5, -0.1)
                output_feat = input_feat + shift
                task["train"].append({"input": input_feat, "output": output_feat})
                
            # Create candidates (mocking different API sequences)
            correct_shift = shift
            correct_candidate = task["train"][-1]["input"] + correct_shift # Base it on the last train input
            task["test_input"] = task["train"][-1]["input"]
            task["test_output"] = correct_candidate
            
            candidates = [correct_candidate]
            for _ in range(4):
                wrong_shift = rng.uniform(-1, 1)
                candidates.append(task["test_input"] + wrong_shift)
                
            # Keep track of correct candidate using exact array matching
            rng.shuffle(candidates)
            correct_idx = next(i for i, c in enumerate(candidates) if np.array_equal(c, correct_candidate))
            
            task["candidates"] = candidates
            task["correct_idx"] = correct_idx
            
            tasks.append(task)
            
    return tasks
