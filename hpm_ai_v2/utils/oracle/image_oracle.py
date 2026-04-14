"""
ImageOracle — computes empirical state vector for PIL image outputs.
"""
from __future__ import annotations

import numpy as np
from typing import Any, List, Optional, TYPE_CHECKING
from .base import BaseOracle

if TYPE_CHECKING:
    from hpm_ai_v2.domains.base import DomainConfig

class ImageOracle(BaseOracle):
    """Computes empirical state vector for PIL image outputs."""

    def __init__(self, config: "DomainConfig"):
        self.config = config

    def compute_state(
        self,
        outputs: List[Any],
        errors: List[Optional[str]],
        code: str = "",
    ) -> np.ndarray:
        s_dim = self.config.S_DIM
        s = np.zeros(s_dim)
        valid_outputs = [o for o, e in zip(outputs, errors) if e is None]
        if not valid_outputs:
            s[0] = 0.0
            return s
        s[0] = 1.0
        
        # For each output image (PIL Image), compute statistics
        pixels = []
        for img in valid_outputs:
            try:
                # convert to grayscale and normalize
                arr = np.array(img.convert('L')) / 255.0
                pixels.extend(arr.flatten())
            except Exception:
                continue
        
        if pixels:
            s[3] = float(np.mean(pixels))   # mean brightness
            s[4] = float(np.std(pixels))    # contrast proxy
            # simple edge presence: high variance
            s[12] = 1.0 if s[4] > 0.2 else 0.0
            
        return s
