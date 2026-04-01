"""
Sovereign ARC Loader.

Extracts multi-modal feature vectors from ARC examples:
- Spatial Slice: Grid delta (O - I)
- Symbolic Slice: Numerical invariants (counts, dimensions, parity)
- Structural Slice: Topological features (components, symmetry, Euler char)
"""
from __future__ import annotations

import json
import glob
from pathlib import Path
import numpy as np
from scipy import ndimage

# Fixed dimensions for the Sovereign Manifold
S_DIM = 100  # 10x10 Spatial
M_DIM = 30   # Symbolic/Math
C_DIM = 20   # Structural/Topological
COMMON_D = S_DIM + M_DIM + C_DIM

S_SLICE = slice(0, S_DIM)
M_SLICE = slice(S_DIM, S_DIM + M_DIM)
C_SLICE = slice(S_DIM + M_DIM, COMMON_D)

def extract_spatial(input_grid: np.ndarray, output_grid: np.ndarray) -> np.ndarray:
    """Returns 100D delta vector (normalized)."""
    # Resize or pad both to 10x10
    def normalize(g):
        res = np.zeros((10, 10))
        r, c = min(10, g.shape[0]), min(10, g.shape[1])
        res[:r, :c] = g[:r, :c]
        return res / 9.0

    delta = normalize(output_grid) - normalize(input_grid)
    return delta.flatten()

def extract_symbolic(input_grid: np.ndarray, output_grid: np.ndarray) -> np.ndarray:
    """Returns 30D numerical invariant vector."""
    m = np.zeros(M_DIM)
    
    # Grid dimensions (normalized by 10)
    m[0] = input_grid.shape[0] / 10.0
    m[1] = input_grid.shape[1] / 10.0
    m[2] = output_grid.shape[0] / 10.0
    m[3] = output_grid.shape[1] / 10.0
    
    # Color counts (0-9) in output
    for c in range(10):
        m[4 + c] = np.sum(output_grid == c) / (output_grid.size if output_grid.size > 0 else 1)
        
    # Delta unique colors
    u_in = len(np.unique(input_grid))
    u_out = len(np.unique(output_grid))
    m[14] = (u_out - u_in) / 10.0
    
    # Delta active pixels
    a_in = np.sum(input_grid > 0)
    a_out = np.sum(output_grid > 0)
    m[15] = (a_out - a_in) / 100.0
    
    # Parity of dimensions
    m[16] = (output_grid.shape[0] % 2)
    m[17] = (output_grid.shape[1] % 2)
    
    # Prime counts (normalized)
    primes = {2, 3, 5, 7}
    m[18] = np.sum([1 for val in output_grid.flatten() if val in primes]) / 100.0
    
    # Is identity?
    m[19] = 1.0 if np.array_equal(input_grid, output_grid) else 0.0
    
    return m

def extract_structural(input_grid: np.ndarray, output_grid: np.ndarray) -> np.ndarray:
    """Returns 20D topological feature vector."""
    c = np.zeros(C_DIM)
    
    # Use output grid for structure
    binary = (output_grid > 0).astype(int)
    if binary.size == 0:
        return c
        
    # Connected components
    labeled, n_comp = ndimage.label(binary)
    c[0] = n_comp / 10.0
    
    # Symmetry scores
    def sym_score(g, axis):
        if g.size == 0: return 0.0
        rev = np.flip(g, axis=axis)
        return np.sum(g == rev) / g.size
        
    c[1] = sym_score(binary, 0) # Vertical
    c[2] = sym_score(binary, 1) # Horizontal
    
    # Euler characteristic (simple: components - holes)
    # Using 1 - holes for binary grids
    filled = ndimage.binary_fill_holes(binary).astype(int)
    holes = np.sum(filled - binary)
    c[3] = (n_comp - holes) / 10.0
    
    # Largest component area
    if n_comp > 0:
        comp_sizes = np.bincount(labeled.ravel())[1:]
        c[4] = np.max(comp_sizes) / 100.0
    
    return c

def load_sovereign_tasks(data_dir: str = "data/ARC-AGI-2/data/training") -> list[dict]:
    """Loads ARC tasks and extracts Sovereign vectors."""
    tasks = []
    for f in sorted(glob.glob(f"{data_dir}/*.json")):
        with open(f) as jf:
            d = json.load(jf)
        puzzle_id = Path(f).stem
        
        examples = []
        for ex in d["train"]:
            i_grid = np.array(ex["input"])
            o_grid = np.array(ex["output"])
            
            vec = np.zeros(COMMON_D)
            vec[S_SLICE] = extract_spatial(i_grid, o_grid)
            vec[M_SLICE] = extract_symbolic(i_grid, o_grid)
            vec[C_SLICE] = extract_structural(i_grid, o_grid)
            
            examples.append({
                "vec": vec,
                "input": i_grid,
                "output": o_grid
            })
            
        tasks.append({
            "id": puzzle_id,
            "train": examples
        })
    return tasks
