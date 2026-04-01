"""
ARC Rosetta Dataset Generator for SP19.

Generates "Count-Governed Rotation" tasks:
- Input: 3x3 grid with N active pixels (N=1..4).
- Rule: Output is input rotated by N * 90 degrees.
- Domain Slices:
    - Spatial: 100D (10x10 delta)
    - Symbolic: 30D (Numerical invariants)
"""
from __future__ import annotations

import numpy as np
from hpm_fractal_node.arc.arc_sovereign_loader import (
    extract_spatial, extract_symbolic, COMMON_D, S_SLICE, M_SLICE, C_SLICE
)

def generate_rosetta_samples(n_per_rule: int = 20, seed: int = 42) -> list[dict]:
    """
    Generates samples for the 4 rules:
    Rule 1: 1 pixel -> Rotate 90
    Rule 2: 2 pixels -> Rotate 180
    Rule 3: 3 pixels -> Rotate 270
    Rule 4: 4 pixels -> Rotate 360 (Identity)
    """
    rng = np.random.default_rng(seed)
    samples = []
    
    for n_pixels in [1, 2, 3, 4]:
        for _ in range(n_per_rule):
            # Create unique 3x3 input grid
            input_grid = np.zeros((3, 3), dtype=int)
            indices = rng.choice(9, size=n_pixels, replace=False)
            input_grid.flat[indices] = rng.integers(1, 10, size=n_pixels) # Random colors 1-9
            
            # Apply rule: rotate k times (90 deg each)
            k = n_pixels 
            output_grid = np.rot90(input_grid, k=-k) # k=-1 is 90 deg clockwise
            
            # Extract features
            vec = np.zeros(COMMON_D)
            vec[S_SLICE] = extract_spatial(input_grid, output_grid)
            vec[M_SLICE] = extract_symbolic(input_grid, output_grid)
            # (Structural slice omitted or zeroed for this specific logic test)
            
            samples.append({
                "n_pixels": n_pixels,
                "rotation_deg": (n_pixels * 90) % 360,
                "input": input_grid,
                "output": output_grid,
                "vec": vec
            })
            
    # Shuffle
    rng.shuffle(samples)
    return samples

if __name__ == "__main__":
    samples = generate_rosetta_samples(n_per_rule=2)
    for s in samples:
        print(f"Pixels: {s['n_pixels']} -> Rot: {s['rotation_deg']} deg")
        print(f"Input:\n{s['input']}")
        print(f"Output:\n{s['output']}\n")
