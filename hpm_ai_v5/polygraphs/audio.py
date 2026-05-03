from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import librosa
from ..core import State
from .base import PolygraphView, PolygraphGenerator

@dataclass(slots=True)
class AudioPolygraphGenerator(PolygraphGenerator):
    """Generate MFCC and spectral views from raw audio signal (numpy array)."""
    sr: int = 22050
    n_mfcc: int = 13

    def generate(self, raw: Any, *, context: dict[str, Any] | None = None) -> list[PolygraphView]:
        if not isinstance(raw, np.ndarray):
            try:
                raw = np.array(raw, dtype=np.float32)
            except Exception:
                raise TypeError("AudioPolygraphGenerator expects a numpy array or list of floats")
        context = dict(context or {})
        context.setdefault("domain", "audio")
        
        # Compute MFCCs over frames, then flatten
        mfccs = librosa.feature.mfcc(y=raw, sr=self.sr, n_mfcc=self.n_mfcc)
        mfcc_flat = tuple(map(float, mfccs.flatten()))
        views = [PolygraphView(
            name="mfcc",
            state=State(value=mfcc_flat, context={**context, "view": "mfcc", "shape": mfccs.shape}),
            context={**context, "view": "mfcc"},
        )]
        
        # Delta (first derivative)
        delta = librosa.feature.delta(mfccs)
        delta_flat = tuple(map(float, delta.flatten()))
        views.append(PolygraphView(
            name="delta_mfcc",
            state=State(value=delta_flat, context={**context, "view": "delta"}),
            context={**context, "view": "delta"},
        ))
        
        # Energy (RMS)
        rms = librosa.feature.rms(y=raw)
        rms_flat = tuple(map(float, rms.flatten()))
        views.append(PolygraphView(
            name="energy",
            state=State(value=rms_flat, context={**context, "view": "energy"}),
            context={**context, "view": "energy"},
        ))
        return views
