"""
Sovereign ARC Loader (Upgraded for SP40 Contextual Manifold).

Extracts multi-modal feature vectors from ARC examples:
- I_SLICE (0-900): Input Grid (archetype context)
- D_SLICE (900-1800): Grid Delta (O - I)
- M_SLICE (1800-1830): Numerical invariants
- C_SLICE (1830-1850): Topological features
"""
from __future__ import annotations

import json
import glob
from pathlib import Path
import numpy as np
from scipy import ndimage

# Sovereign Manifold (1850D)
G_DIM = 900   # 30x30 Grid
M_DIM = 30    # Symbolic/Math
C_DIM = 20    # Structural/Topological
COMMON_D = G_DIM + G_DIM + M_DIM + C_DIM

I_SLICE = slice(0, G_DIM)
D_SLICE = slice(G_DIM, G_DIM + G_DIM)
M_SLICE = slice(G_DIM + G_DIM, G_DIM + G_DIM + M_DIM)
C_SLICE = slice(G_DIM + G_DIM + M_DIM, COMMON_D)

# Unified Spatial Manifold (1800D)
S_SLICE = slice(0, G_DIM + G_DIM) 
S_DIM = G_DIM + G_DIM

def grid_to_vec(g: np.ndarray) -> np.ndarray:
    """Crops to content and flattens to 900D with centering."""
    if g.size == 0: return np.zeros(900)
    
    # 1. Find Bounding Box
    coords = np.argwhere(g > 0)
    if coords.size == 0:
        # Zero grid, just return flattened 30x30
        res = np.zeros((30, 30))
        return res.flatten()
        
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    content = g[y0:y1, x0:x1]
    
    # 2. Center in 30x30 canvas
    res = np.zeros((30, 30))
    cr, cc = content.shape
    r_start = (30 - cr) // 2
    c_start = (30 - cc) // 2
    
    # Clip if too big
    r_size = min(cr, 30)
    c_size = min(cc, 30)
    
    res[r_start:r_start+r_size, c_start:c_start+c_size] = content[:r_size, :c_size]
    
    return (res / 9.0).flatten()

def extract_spatial(input_grid: np.ndarray, output_grid: np.ndarray) -> np.ndarray:
    """Returns 900D delta vector."""
    return grid_to_vec(output_grid) - grid_to_vec(input_grid)

def extract_symbolic(input_grid: np.ndarray, output_grid: np.ndarray) -> np.ndarray:
    """Returns 30D numerical invariant vector."""
    m = np.zeros(M_DIM)
    m[0] = input_grid.shape[0] / 30.0
    m[1] = input_grid.shape[1] / 30.0
    m[2] = output_grid.shape[0] / 30.0
    m[3] = output_grid.shape[1] / 30.0
    for c in range(10):
        m[4 + c] = np.sum(output_grid == c) / (output_grid.size if output_grid.size > 0 else 1)
    u_in, u_out = len(np.unique(input_grid)), len(np.unique(output_grid))
    m[14] = (u_out - u_in) / 10.0
    a_in, a_out = np.sum(input_grid > 0), np.sum(output_grid > 0)
    m[15] = (a_out - a_in) / 900.0
    m[16], m[17] = (output_grid.shape[0] % 2), (output_grid.shape[1] % 2)
    primes = {2, 3, 5, 7}
    m[18] = np.sum([1 for val in output_grid.flatten() if val in primes]) / 900.0
    m[19] = 1.0 if np.array_equal(input_grid, output_grid) else 0.0
    
    # ORIGIN ATTRIBUTES: Where is the content?
    coords = np.argwhere(output_grid > 0)
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        m[20] = y0 / 30.0
        m[21] = x0 / 30.0
    
    return m

def extract_structural(input_grid: np.ndarray, output_grid: np.ndarray) -> np.ndarray:
    """Returns 20D topological feature vector."""
    c = np.zeros(C_DIM)
    binary = (output_grid > 0).astype(int)
    if binary.size == 0: return c
    labeled, n_comp = ndimage.label(binary)
    c[0] = n_comp / 100.0
    def sym_score(g, axis):
        if g.size == 0: return 0.0
        rev = np.flip(g, axis=axis)
        return np.sum(g == rev) / g.size
    c[1], c[2] = sym_score(binary, 0), sym_score(binary, 1)
    filled = ndimage.binary_fill_holes(binary).astype(int)
    holes = np.sum(filled - binary)
    c[3] = (n_comp - holes) / 100.0
    if n_comp > 0:
        comp_sizes = np.bincount(labeled.ravel())[1:]
        c[4] = np.max(comp_sizes) / 900.0
    return c

def load_sovereign_tasks(data_dir: str = "data/ARC-AGI-2/data/training") -> list[dict]:
    """Loads ARC tasks and extracts Contextual 1850D vectors."""
    tasks = []
    for f in sorted(glob.glob(f"{data_dir}/*.json")):
        with open(f) as jf:
            d = json.load(jf)
        puzzle_id = Path(f).stem
        
        train_examples = []
        for ex in d["train"]:
            i_grid, o_grid = np.array(ex["input"]), np.array(ex["output"])
            vec = np.zeros(COMMON_D)
            vec[I_SLICE] = grid_to_vec(i_grid)
            vec[D_SLICE] = extract_spatial(i_grid, o_grid)
            vec[M_SLICE] = extract_symbolic(i_grid, o_grid)
            vec[C_SLICE] = extract_structural(i_grid, o_grid)
            train_examples.append({"vec": vec, "input": i_grid, "output": o_grid})
            
        test_examples = []
        for ex in d["test"]:
            i_grid = np.array(ex["input"])
            o_grid = np.array(ex["output"]) if "output" in ex else None
            vec = np.zeros(COMMON_D)
            vec[I_SLICE] = grid_to_vec(i_grid)
            vec[D_SLICE] = 0.0 # Goal target
            vec[M_SLICE] = extract_symbolic(i_grid, i_grid) # Approximate
            test_examples.append({"vec": vec, "input": i_grid, "output": o_grid})
            
        tasks.append({"id": puzzle_id, "train": train_examples, "test": test_examples})
    return tasks
