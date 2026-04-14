"""
AudioRenderer — Specialized renderer for the audio domain.
"""
from __future__ import annotations
import numpy as np
from typing import Any, List, Optional, TYPE_CHECKING
from hpm_ai_v2.utils.base_renderer import Renderer
from hfn.hfn import HFN

if TYPE_CHECKING:
    from hpm_ai_v2.domains.audio_domain import AudioDomainConfig

class AudioRenderer(Renderer):
    """
    Renders audio transformation macros into Python librosa code.
    """
    def __init__(self, config: AudioDomainConfig):
        self.config = config

    def render(self, node: HFN) -> str:
        # 1. Flatten the node's structure into a sequence of primitive ops
        ops = self._extract_ops(node)
        
        # 2. Build code body
        lines = [
            "import numpy as np",
            "import librosa",
            "y = inp.copy() if hasattr(inp, 'copy') else np.array(inp)",
            "# apply operations in order"
        ]
        sr = self.config.sample_rate
        for op in ops:
            if op == "PITCH_UP_2":
                lines.append(f"y = librosa.effects.pitch_shift(y, sr={sr}, n_steps=2)")
            elif op == "PITCH_DOWN_2":
                lines.append(f"y = librosa.effects.pitch_shift(y, sr={sr}, n_steps=-2)")
            elif op == "VOLUME_UP":
                lines.append("y = y * 1.5")
            elif op == "VOLUME_DOWN":
                lines.append("y = y * 0.7")
            elif op == "TIME_STRETCH_2X":
                lines.append("y = librosa.effects.time_stretch(y, rate=2.0)")
            elif op == "LOW_PASS":
                # Simplified low-pass via librosa
                lines.append("y = librosa.effects.preemphasis(y, coef=0.97)")
            elif op == "HIGH_PASS":
                lines.append("y = librosa.effects.preemphasis(y, coef=-0.97)")
        
        lines.append("res = y")
        return "\n".join(lines)

    def render_function(self, node: HFN, func_name: str = "macro_func") -> str:
        """Render a macro as a standalone Python function definition."""
        body = self.render(node)
        indented = body.replace('\n', '\n    ')
        return f"def {func_name}(inp, inputs):\n    {indented}\n    return res"

    def _extract_ops(self, node: HFN) -> List[str]:
        """Traverse the node's inputs (multi-polygraph) to collect primitive op names."""
        ops = []
        if node.inputs:
            for child in node.inputs:
                ops.extend(self._extract_ops(child))
        else:
            # Leaf node – extract concept
            concept = self._get_concept(node)
            if concept:
                ops.append(concept)
        return ops

    def _get_concept(self, node: HFN) -> Optional[str]:
        """Extract concept name from HFN mu vector or ID."""
        for c in self.config.concepts:
            if node.id == f"prior_rule_{c}" or node.id == f"audio_op_{c}":
                return c
        
        start = self.config.S_DIM
        end = start + self.config.DIM
        vec = node.mu[start:end]
        if np.max(vec) > 0.5:
            idx = np.argmax(vec)
            return self.config.concepts[idx]
        return None
