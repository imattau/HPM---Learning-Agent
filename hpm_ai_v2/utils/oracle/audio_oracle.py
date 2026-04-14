"""
AudioOracle — computes empirical state vector for audio waveforms.
"""
from __future__ import annotations

import numpy as np
from typing import Any, List, Optional, TYPE_CHECKING
from .base import BaseOracle

if TYPE_CHECKING:
    from hpm_ai_v2.domains.base import DomainConfig

class AudioOracle(BaseOracle):
    """Computes empirical state vector for audio waveforms."""

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
        
        import librosa
        centroids = []
        rms = []
        for audio in valid_outputs:
            if isinstance(audio, np.ndarray):
                try:
                    # Use provided sample rate if available in config
                    sr = getattr(self.config, 'sample_rate', 22050)
                    cent = librosa.feature.spectral_centroid(y=audio, sr=sr)[0].mean()
                    rms_val = np.sqrt(np.mean(audio**2))
                    centroids.append(cent)
                    rms.append(rms_val)
                except Exception:
                    continue
        
        if centroids:
            # Normalized spectral centroid (0-5000Hz -> 0-1)
            s[3] = float(np.mean(centroids)) / 5000.0
            s[4] = float(np.mean(rms))
            # simple edge/transient proxy: high variance in amplitude
            s[12] = 1.0 if s[4] > 0.1 else 0.0
            
        return s
