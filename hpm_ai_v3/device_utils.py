import os
import torch
import multiprocessing as mp
from typing import Optional

def get_device(verbose: bool = True) -> torch.device:
    """Auto-detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print(f"[Device] Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            print("[Device] Using Apple Metal (MPS)")
    else:
        device = torch.device("cpu")
        if verbose:
            print(f"[Device] Using CPU ({os.cpu_count()} threads)")
    return device

def get_optimal_dtype(device: torch.device) -> torch.dtype:
    """Return optimal dtype for device."""
    return torch.float32

def get_parallel_context():
    """Return multiprocessing context appropriate for OS."""
    if os.name == 'nt':
        return mp.get_context('spawn')
    return mp.get_context('fork')

def set_torch_threads(n_threads: Optional[int] = None):
    """Configure CPU threading for optimal performance."""
    if n_threads is None:
        n_threads = os.cpu_count()
    torch.set_num_threads(n_threads)
    torch.set_num_interop_threads(min(4, n_threads))
