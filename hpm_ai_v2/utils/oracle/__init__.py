"""
Oracle module for HPM AI v2.
"""
from .base import BaseOracle, CountingOracle
from .list_oracle import ListOracle
from .image_oracle import ImageOracle
from .audio_oracle import AudioOracle
from .graph_oracle import GraphOracle
from .video_oracle import VideoOracle

# For backward compatibility
EmpiricalOracle = ListOracle
