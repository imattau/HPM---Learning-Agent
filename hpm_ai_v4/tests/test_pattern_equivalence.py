import numpy as np

from hpm_ai_v4.pattern import HierarchicalPattern
from hpm_ai_v4.tools.pattern_equivalence import PatternEquivalenceIndex


def test_pattern_equivalence_index_exact_sequence_reuse():
    index = PatternEquivalenceIndex()
    seq = [0, 1, 0, 1, 2, 0, 1, 0]
    index.register_sequence(seq)

    pattern = HierarchicalPattern(pattern_id=0, latent_dim=2, obs_dim=3)
    match = index.classify(pattern, seq)

    assert match.exact_match is True
    assert match.status == "exact"
    assert match.residual_fraction == 0.0
    assert match.residual_sequence == ()


def test_pattern_equivalence_index_prefers_short_residual_for_known_pattern(monkeypatch):
    index = PatternEquivalenceIndex()
    pattern = HierarchicalPattern(pattern_id=1, latent_dim=2, obs_dim=3)
    seq = [0, 1, 0, 1, 2, 0, 1, 0, 1, 2]

    def fake_residual_observations(obs_seq, *, min_residual=4, keep_fraction=0.35):
        return list(obs_seq[-3:])

    monkeypatch.setattr(pattern, "residual_observations", fake_residual_observations)
    match = index.classify(pattern, seq)

    assert match.status in {"equivalent", "near"}
    assert len(match.residual_sequence) == 3
    assert 0.0 < match.residual_fraction < 1.0


def test_pattern_equivalence_index_detects_composed_sequence():
    index = PatternEquivalenceIndex()
    index.register_sequence([1])
    index.register_sequence([2])
    index.register_sequence([3])

    pattern = HierarchicalPattern(pattern_id=2, latent_dim=2, obs_dim=4)
    match = index.classify(pattern, [1, 2, 3])

    assert match.status == "composed"
    assert match.composition_parts == ((1,), (2,), (3,))
    assert match.composition_coverage == 1.0
    assert match.residual_sequence == ()
    assert match.residual_fraction == 0.0
