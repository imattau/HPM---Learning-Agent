import numpy as np
from typing import List, Any
from hpm_ai_v2.utils.oracle.base import BaseOracle

class VideoOracle(BaseOracle):
    """
    Computes a 20D state vector from a list of video frames.
    """
    def __init__(self, config):
        self.config = config
        self.call_count = 0

    def compute_state(self, outputs: List[Any], errors: List[Any], code: str = "") -> np.ndarray:
        s = np.zeros(self.config.S_DIM)
        valid = [o for o, e in zip(outputs, errors) if e is None and o is not None]
        if not valid:
            return s
        
        # dim0: valid
        s[0] = 1.0
        # dim1: is_video (if list of frames)
        if isinstance(valid[0], list) and len(valid[0]) > 0:
            s[1] = 1.0
        
        all_pixels = []
        for video in valid:
            if not isinstance(video, list):
                continue
            for frame in video:
                arr = np.array(frame).astype(float) / 255.0
                all_pixels.extend(arr.flatten())
                
        if all_pixels:
            # dim3: mean brightness, dim4: std dev
            s[3] = float(np.mean(all_pixels))
            s[4] = float(np.std(all_pixels))
            
        # Code structure flags
        s[10] = 1.0 if 'for ' in code else 0.0
        s[11] = 1.0 if '.append(' in code else 0.0
        s[12] = 1.0 if 'if ' in code else 0.0
        s[13] = 1.0 if any(op in code for op in ['+=', '-=', '*=']) else 0.0
        s[14] = 1.0 if 'x = inp' in code else 0.0
        s[15] = 1.0 if 'val = ' in code else 0.0
        s[16] = 1.0 if 'res = []' in code else 0.0
        
        return s
