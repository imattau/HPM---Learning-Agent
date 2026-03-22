"""Tests for ArchiveForecaster."""
import pickle
from pathlib import Path

import numpy as np
import pytest

from hpm.store.contextual_store import SubstrateSignature
from hpm.store.archive_librarian import CandidateArchive
from hpm.store.archive_forecaster import ArchiveForecaster, RankedCandidate
from hpm.patterns.factory import make_pattern


def _make_sig(grid_size=(5, 5), color_count=2, object_count=1, bucket="square"):
    return SubstrateSignature(
        grid_size=grid_size,
        unique_color_count=color_count,
        object_count=object_count,
        aspect_ratio_bucket=bucket,
    )


def _make_candidate(tmp_path: Path, episode_id: str, patterns: list,
                    success: bool = True) -> CandidateArchive:
    """Write a pickle archive and return a CandidateArchive pointing to it."""
    sig = _make_sig()
    archive_path = tmp_path / f"{episode_id}.pkl"
    records = [(p, 0.9, "agent") for p in patterns]
    archive_path.write_bytes(pickle.dumps(records))
    return CandidateArchive(
        episode_id=episode_id,
        archive_path=archive_path,
        signature=sig,
        success=success,
    )


# ---------------------------------------------------------------------------
# rank tests
# ---------------------------------------------------------------------------

def test_rank_empty_candidates():
    fc = ArchiveForecaster()
    result = fc.rank([], obs=[np.zeros(4)])
    assert result == []


def test_rank_no_obs_returns_all_with_zero_nll(tmp_path):
    p = make_pattern(mu=np.ones(4), scale=np.eye(4), pattern_type="gaussian")
    c = _make_candidate(tmp_path, "ep1", [p])
    fc = ArchiveForecaster()
    result = fc.rank([c], obs=[])
    assert len(result) == 1
    assert result[0].mean_nll == 0.0
    assert result[0].candidate is c


def test_rank_below_threshold_included(tmp_path):
    # Pattern near zero: NLL should be finite and below high threshold
    p = make_pattern(mu=np.zeros(4), scale=np.eye(4), pattern_type="gaussian")
    c = _make_candidate(tmp_path, "ep1", [p])
    fc = ArchiveForecaster()
    obs = [np.zeros(4)]
    result = fc.rank([c], obs=obs, nll_threshold=50.0)
    assert len(result) == 1
    assert isinstance(result[0], RankedCandidate)


def test_rank_above_threshold_excluded(tmp_path):
    # Pattern far from obs: NLL will be very high
    p = make_pattern(mu=np.array([1000.0, 1000.0, 1000.0, 1000.0]),
                     scale=np.eye(4) * 0.0001, pattern_type="gaussian")
    c = _make_candidate(tmp_path, "ep1", [p])
    fc = ArchiveForecaster()
    obs = [np.zeros(4)]
    result = fc.rank([c], obs=obs, nll_threshold=0.001)
    assert result == []


def test_rank_sorted_ascending_nll(tmp_path):
    # Two patterns: one close to obs (low NLL), one far (higher NLL)
    p_close = make_pattern(mu=np.zeros(4), scale=np.eye(4), pattern_type="gaussian")
    p_far = make_pattern(mu=np.ones(4) * 2, scale=np.eye(4), pattern_type="gaussian")
    c1 = _make_candidate(tmp_path, "ep1", [p_far])
    c2 = _make_candidate(tmp_path, "ep2", [p_close])
    fc = ArchiveForecaster()
    obs = [np.zeros(4)]
    result = fc.rank([c1, c2], obs=obs, nll_threshold=500.0)
    assert len(result) == 2
    assert result[0].mean_nll <= result[1].mean_nll


def test_rank_skips_missing_archive(tmp_path):
    sig = _make_sig()
    missing = CandidateArchive(
        episode_id="missing",
        archive_path=tmp_path / "nonexistent.pkl",
        signature=sig,
        success=True,
    )
    fc = ArchiveForecaster()
    result = fc.rank([missing], obs=[np.zeros(4)])
    assert result == []


def test_rank_skips_corrupt_archive(tmp_path):
    sig = _make_sig()
    bad_path = tmp_path / "bad.pkl"
    bad_path.write_bytes(b"not-valid-pickle-data")
    bad = CandidateArchive(
        episode_id="bad",
        archive_path=bad_path,
        signature=sig,
        success=True,
    )
    fc = ArchiveForecaster()
    result = fc.rank([bad], obs=[np.zeros(4)])
    assert result == []


def test_rank_skips_empty_patterns(tmp_path):
    sig = _make_sig()
    empty_path = tmp_path / "empty.pkl"
    empty_path.write_bytes(pickle.dumps([]))  # empty records
    empty = CandidateArchive(
        episode_id="empty",
        archive_path=empty_path,
        signature=sig,
        success=True,
    )
    fc = ArchiveForecaster()
    result = fc.rank([empty], obs=[np.zeros(4)])
    assert result == []


def test_rank_returns_ranked_candidate_dataclass(tmp_path):
    p = make_pattern(mu=np.zeros(4), scale=np.eye(4), pattern_type="gaussian")
    c = _make_candidate(tmp_path, "ep1", [p])
    fc = ArchiveForecaster()
    result = fc.rank([c], obs=[np.zeros(4)])
    assert len(result) == 1
    r = result[0]
    assert isinstance(r, RankedCandidate)
    assert r.candidate is c
    assert isinstance(r.mean_nll, float)


def test_rank_multiple_obs(tmp_path):
    p = make_pattern(mu=np.zeros(4), scale=np.eye(4), pattern_type="gaussian")
    c = _make_candidate(tmp_path, "ep1", [p])
    fc = ArchiveForecaster()
    obs = [np.zeros(4), np.ones(4) * 0.1, np.ones(4) * 0.2]
    result = fc.rank([c], obs=obs, nll_threshold=500.0)
    assert len(result) == 1
