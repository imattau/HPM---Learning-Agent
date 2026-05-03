import numpy as np
import pytest
from hpm_ai_v5.polygraphs import (
    AudioPolygraphGenerator,
    GraphPolygraphGenerator,
    GridPolygraphGenerator,
    TextPolygraphGenerator,
    TimeSeriesPolygraphGenerator,
)

def test_text_polygraph():
    generator = TextPolygraphGenerator()
    views = generator.generate("She loves natural language processing.")
    
    view_names = {v.name for v in views}
    assert "tokens" in view_names
    assert "pos_tags" in view_names
    assert "dependencies" in view_names
    assert "lemmas" in view_names
    
    for v in views:
        assert isinstance(v.state.value, tuple)

def test_grid_polygraph():
    generator = GridPolygraphGenerator()
    grid = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
    views = generator.generate(grid)
    
    view_names = {v.name for v in views}
    assert "flattened" in view_names
    assert "edges" in view_names
    assert "components" in view_names
    assert "distance" in view_names
    
    for v in views:
        assert isinstance(v.state.value, tuple)
        assert len(v.state.value) == 9

def test_audio_polygraph():
    generator = AudioPolygraphGenerator()
    # 1 second of white noise
    audio = np.random.uniform(-1, 1, 22050).astype(np.float32)
    views = generator.generate(audio)
    
    view_names = {v.name for v in views}
    assert "mfcc" in view_names
    assert "delta_mfcc" in view_names
    assert "energy" in view_names
    
    for v in views:
        assert isinstance(v.state.value, tuple)

def test_graph_polygraph():
    generator = GraphPolygraphGenerator()
    # Adjacency dict
    graph = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    views = generator.generate(graph)
    
    view_names = {v.name for v in views}
    assert "adjacency" in view_names
    assert "degrees" in view_names
    assert "betweenness" in view_names
    
    for v in views:
        assert isinstance(v.state.value, tuple)

def test_timeseries_polygraph():
    generator = TimeSeriesPolygraphGenerator(window_size=3)
    
    # First value
    views = generator.generate(1.0)
    assert len(views) == 1 # only raw
    assert views[0].state.value == 1.0
    
    # Second value
    views = generator.generate(2.0)
    assert len(views) == 3 # raw, rolling_mean, diff
    
    view_names = {v.name for v in views}
    assert "raw" in view_names
    assert "rolling_mean" in view_names
    assert "diff" in view_names
    
    # Check values
    raw_view = next(v for v in views if v.name == "raw")
    mean_view = next(v for v in views if v.name == "rolling_mean")
    diff_view = next(v for v in views if v.name == "diff")
    
    assert raw_view.state.value == 2.0
    assert mean_view.state.value == 1.5
    assert diff_view.state.value == 1.0
