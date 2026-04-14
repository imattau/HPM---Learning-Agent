"""
Oracle module for HPM AI v2.
"""
from .base import BaseOracle, CountingOracle
from .list_oracle import ListOracle
from .image_oracle import ImageOracle
from .audio_oracle import AudioOracle

# For backward compatibility
EmpiricalOracle = ListOracle
