from __future__ import annotations
import numpy as np
from typing import List
from hfn.hfn import HFN
from hpm_ai_v2.domains.base import DomainConfig

class VideoDomainConfig(DomainConfig):
    """
    HPM domain for 64x64 grayscale video transformations.
    """
    def __init__(self, frame_size=64, num_frames=8):
        self.frame_size = frame_size
        self.num_frames = num_frames
        self.pixels_per_frame = frame_size * frame_size
        
        # Concepts for video primitives:
        concepts = [
            "VAR_INP",            # x = inp
            "LIST_INIT",          # res = []
            "FOR_EACH_FRAME",     # for frame in list(x):
            "ITEM_ACCESS",        # val = frame
            "FRAME_APPEND",       # res.append(val)
            "ROTATE_90",          # val = cv2.rotate(val, cv2.ROTATE_90_CLOCKWISE)
            "FLIP_H",             # val = cv2.flip(val, 1)
            "FLIP_V",             # val = cv2.flip(val, 0)
            "BRIGHTNESS_UP",      # val = np.clip(val + 30, 0, 255)
            "BRIGHTNESS_DOWN",    # val = np.clip(val - 30, 0, 255)
            "COND_BRIGHTNESS_HIGH", # if np.mean(val) > 128:
            "BLOCK_END",          # end of block
            "RETURN",             # return res
            "MAP_START",          # Composite: VAR_INP + LIST_INIT + FOR_EACH_FRAME + ITEM_ACCESS
            "MAP_END",            # Composite: FRAME_APPEND + BLOCK_END
        ]
        
        # State dimensions (20D):
        # 0: valid, 1: is_video, 2: num_frames, 3: mean_brightness, 4: std_dev
        # 10: loop, 11: append, 12: if, 13: mutation, 14: x=inp, 15: val=frame, 16: res=[]
        super().__init__(concepts, s_dim=20)
        self.STRUCT_DIMS = [1, 2, 3, 4] + list(range(10, 17))

def get_video_primitive_nodes(config: VideoDomainConfig) -> List[HFN]:
    """Return a set of initial prior nodes for the video domain."""
    nodes = []
    for i, c in enumerate(config.concepts):
        mu = np.zeros(config.m_dim)
        mu[config.S_DIM + i] = 5.0
        
        # Iteration motifs are strongly associated with list/video structure
        if c in {"FOR_EACH_FRAME", "FRAME_APPEND", "MAP_START", "MAP_END", "LIST_INIT"}:
            mu[1] = 1.0   # is_video
            mu[10] = 1.0  # loop flag
            mu[11] = 1.0  # append flag
            
        node = HFN(
            mu=mu,
            sigma=np.ones(config.m_dim) * 5.0,
            id=f"prior_rule_{c}",
            relation_type="prior",
            use_diag=True
        )
        nodes.append(node)
    return nodes
