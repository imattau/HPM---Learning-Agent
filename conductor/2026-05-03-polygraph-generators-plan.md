# Polygraph Generators Expansion Plan

## Objective
Implement new `PolygraphGenerator` classes for Text, Grid, Audio, Graph, and Time Series domains, expanding the `hpm_ai_v5/polygraphs/` module.

## Motivation
Currently, only `NumericPolygraphGenerator` exists. By adding domain-specific generators, the HPM core can process and learn from different types of structured data (e.g., text tokens/POS tags, grid components, audio MFCCs) by converting them into multiple concurrent "views" of states, which are then fused via polygraph agreement.

## Implementation Steps

1. **TextPolygraphGenerator (`hpm_ai_v5/polygraphs/text.py`)**
   - Uses `spacy` to generate views: `tokens`, `pos_tags`, `dependencies`, and `lemmas`.
   - Maps linguistic structures to integers/tuples compatible with HPM `State` values.
   
2. **GridPolygraphGenerator (`hpm_ai_v5/polygraphs/grid.py`)**
   - Uses `scikit-image` and `numpy` to generate views from 2D grids.
   - Views: `flattened`, `edges` (Canny), `components` (Connected Components), and `distance` (Distance Transform).
   
3. **AudioPolygraphGenerator (`hpm_ai_v5/polygraphs/audio.py`)**
   - Uses `librosa` to process audio numpy arrays.
   - Views: `mfcc`, `delta_mfcc`, and `energy` (RMS).
   
4. **GraphPolygraphGenerator (`hpm_ai_v5/polygraphs/graph.py`)**
   - Uses `networkx` to process adjacency dicts or node/edge lists.
   - Views: `adjacency` (flattened matrix), `degrees` (degree sequence), and `betweenness` (centrality).
   
5. **TimeSeriesPolygraphGenerator (`hpm_ai_v5/polygraphs/timeseries.py`)**
   - Uses basic math/numpy to process a sliding window of a scalar stream.
   - Views: `raw`, `rolling_mean`, and `diff` (first derivative).
   
6. **Update Module Initialization (`hpm_ai_v5/polygraphs/__init__.py`)**
   - Export all newly created generators so they are accessible from `hpm_ai_v5.polygraphs`.

7. **Verification (`hpm_ai_v5/tests/test_polygraph_generators.py`)**
   - Write unit tests for each new generator to ensure they produce the correct number of views and that the state values have the expected types.

## Constraints
- **KISS & Minimal Changes:** Apply KISS (Keep It Simple, Stupid) principles and ensure only the smallest practicable changes are made to achieve the objective.
- **Core Isolation:** As with previous implementations, the `hpm_ai_v5/core` MUST NOT be modified. These generators sit solely in the `polygraphs` module.
- All generators must adhere strictly to the `PolygraphGenerator` protocol, returning a list of `PolygraphView` objects.