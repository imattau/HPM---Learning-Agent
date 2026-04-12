"""
Library Discovery Tools for HFN.
Includes a Query to find functions in a module and a Converter that 
probes them to determine their behavioral HFN vector.
"""

import inspect
import importlib.util
import numpy as np
from typing import List, Any, Optional, Tuple
from hfn.query import Query
from hfn.converter import Converter

# Dimensionality for this experiment
S_DIM = 25

import zlib

class BehavioralOracle:
    """
    Computes a 25D behavioral state vector using a deterministic dense projection.
    Uses fast zlib hashing to avoid SHA512 bottleneck in large probing cycles.
    """
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        # Random projection matrix [32D HashSpace -> 25D]
        self.projection = np.random.randn(32, S_DIM)
        self.projection /= np.linalg.norm(self.projection, axis=0)

    def _hash_to_vec(self, text: str) -> np.ndarray:
        # Use multiple seeds with adler32 to create 32D bit-vector
        bits = []
        for i in range(32):
            val = zlib.adler32(f"{i}:{text}".encode())
            bits.append(1.0 if val % 2 == 0 else -1.0)
        return np.array(bits)

    def compute(self, data: Any) -> np.ndarray:
        if data is None:
            return np.zeros(S_DIM)
        
        text_rep = repr(data)
        high_dim_vec = self._hash_to_vec(text_rep)
        latent = high_dim_vec @ self.projection
        
        norm = np.linalg.norm(latent)
        if norm > 0:
            latent /= norm
            
        return latent

class LibraryScannerQuery(Query):
    """Scans a Python module for public functions."""
    def __init__(self, module_name: str, module_path: str):
        self.module_name = module_name
        self.module_path = module_path
        
    def fetch(self, gap_mu: np.ndarray, context=None) -> List[str]:
        # In a real scenario, gap_mu might guide which module to scan.
        # Here we just scan the targeted mock library.
        print(f"  [DEBUG] LibraryScannerQuery.fetch called for {self.module_name}")
        try:
            spec = importlib.util.spec_from_file_location(self.module_name, self.module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            print(f"  [DEBUG] Failed to load module: {e}")
            return []
        
        functions = []
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) and not name.startswith("_"):
                # Use plain name (no sig: prefix) so they are treated as independent 
                # context observations rather than being averaged into one signature.
                functions.append(f"{self.module_name}.{name}")
        print(f"  [DEBUG] Found {len(functions)} functions in {self.module_name}")
        return functions

class LibraryProbingConverter(Converter):
    """
    Probes library functions to determine their HFN behavioral vector.
    """
    def __init__(self, oracle: BehavioralOracle):
        self.oracle = oracle
        # More diverse probes to reveal multi-modal behavior
        self.probe_inputs = [
            [1, 2, 3, 4, 5],
            [10, 10, 20, 20, 30],
            [5, 4, 3, 2, 1],
            [1, [2, 3], 4],
            [100, 200], # Large numbers
            [-1, -2, -3], # Negatives
            [True, False, True], # Bools
            [0, 0, 0] # Zeroes
        ]

    def encode(self, raw: str, D: int) -> List[np.ndarray]:
        # Accept both prefixed and non-prefixed for flexibility
        func_path = raw
        if func_path.startswith("sig: "):
            func_path = func_path[len("sig: "):].strip()
        
        # mod_name is inferred for the experiment
        func_name = func_path.split(".")[-1]
        
        # Dynamically import to probe
        try:
            import hpm_fractal_node.code.mock_tool_lib as module
            func = getattr(module, func_name)
        except Exception as e:
            # print(f"  [DEBUG] Error importing {func_path}: {e}")
            return []
            
        observations = []
        for inp in self.probe_inputs:
            try:
                inp_state = self.oracle.compute(inp)
                out = func(inp)
                out_state = self.oracle.compute(out)
                
                delta = out_state - inp_state
                
                vec = np.zeros(D)
                vec[:S_DIM] = inp_state
                
                # [25D Pre | 10D Concept | 25D Delta] structure
                if D >= 60:
                    vec[35:60] = delta
                    # Use function ID hash to seed the Concept space
                    hash_val = hash(func_name) % 100 / 100.0
                    vec[25] = hash_val
                else:
                    vec[S_DIM:S_DIM*2] = delta
                
                # Add tiny noise to simulate observation uncertainty
                vec += np.random.normal(0, 0.005, D)
                
                observations.append(vec)
            except:
                continue
        
        return observations
