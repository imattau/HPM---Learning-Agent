"""Legacy preprocessing namespace for v5.

Adapter-composition code should prefer the adapter-layer exports. This package
remains as a compatibility surface for older imports.
"""

from .autocorrelation import AutocorrelationPreprocessor
from .base import Preprocessor, PreprocessedInput
from .differencing import DifferencingPreprocessor
from .entropy import EntropyPreprocessor
from .normalisation import NormalisationPreprocessor
from .numeric import NumericPreprocessor
from .prefix_buffer import PrefixBufferAdapter, PrefixBufferPreprocessor
from .rolling_stats import RollingStatsPreprocessor
from .state_fusion import StateFusionPreprocessor
from .symbolic import SymbolicDiscretiser

__all__ = [
    "AutocorrelationPreprocessor",
    "DifferencingPreprocessor",
    "EntropyPreprocessor",
    "NormalisationPreprocessor",
    "NumericPreprocessor",
    "PrefixBufferAdapter",
    "PrefixBufferPreprocessor",
    "RollingStatsPreprocessor",
    "StateFusionPreprocessor",
    "SymbolicDiscretiser",
    "PreprocessedInput",
    "Preprocessor",
]
