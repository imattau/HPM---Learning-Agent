"""Canonical adapter-layer feature transforms and compatibility aliases."""

from __future__ import annotations

from ..preprocessors.autocorrelation import AutocorrelationPreprocessor
from ..preprocessors.differencing import DifferencingPreprocessor
from ..preprocessors.entropy import EntropyPreprocessor
from ..preprocessors.normalisation import NormalisationPreprocessor
from ..preprocessors.numeric import NumericPreprocessor
from ..preprocessors.prefix_buffer import PrefixBufferAdapter, PrefixBufferPreprocessor
from ..preprocessors.rolling_stats import RollingStatsPreprocessor
from ..preprocessors.state_fusion import StateFusionPreprocessor
from ..preprocessors.symbolic import SymbolicDiscretiser
from .action_sequence_unpacker import ActionSequenceUnpacker
from .connected_components import ConnectedComponentsAdapter
from .delta_buffer import DeltaBufferAdapter
from .flatten_grid import FlattenGridAdapter
from .grid_postprocessor import GridPostprocessor
from .recent_buffer import RecentBufferAdapter
from .validation_only import ValidationOnlyAdapter

AutocorrelationAdapter = AutocorrelationPreprocessor
DifferencingAdapter = DifferencingPreprocessor
EntropyAdapter = EntropyPreprocessor
NormalisationAdapter = NormalisationPreprocessor
NumericAdapter = NumericPreprocessor
RollingStatsAdapter = RollingStatsPreprocessor
StateFusionAdapter = StateFusionPreprocessor
SymbolicAdapter = SymbolicDiscretiser

__all__ = [
    "ActionSequenceUnpacker",
    "AutocorrelationAdapter",
    "AutocorrelationPreprocessor",
    "ConnectedComponentsAdapter",
    "DeltaBufferAdapter",
    "DifferencingAdapter",
    "DifferencingPreprocessor",
    "EntropyAdapter",
    "EntropyPreprocessor",
    "FlattenGridAdapter",
    "GridPostprocessor",
    "NormalisationAdapter",
    "NormalisationPreprocessor",
    "NumericAdapter",
    "NumericPreprocessor",
    "PrefixBufferAdapter",
    "PrefixBufferPreprocessor",
    "RecentBufferAdapter",
    "RollingStatsAdapter",
    "RollingStatsPreprocessor",
    "StateFusionAdapter",
    "StateFusionPreprocessor",
    "SymbolicAdapter",
    "SymbolicDiscretiser",
    "ValidationOnlyAdapter",
]
