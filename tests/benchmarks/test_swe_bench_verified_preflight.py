"""Smoke tests for the SWE-bench Verified preflight utility."""

from __future__ import annotations

from benchmarks import swe_bench_verified_preflight as swe


class _FakeDataset(list):
    def select(self, indices):
        return _FakeDataset([self[i] for i in indices])


def test_swe_bench_verified_preflight_summary(monkeypatch):
    fake = _FakeDataset([
        {"repo": "repo-a", "problem_statement": "Fix bug A", "files_changed": ["a.py", "b.py"]},
        {"repo": "repo-b", "problem_statement": "Fix bug B", "files_changed": ["c.py"]},
        {"repo": "repo-a", "problem_statement": "Fix bug C", "files_changed": []},
    ])
    monkeypatch.setattr(swe, "load_dataset", lambda name, split: fake)

    instances = swe.load_instances(split="test", limit=2, dataset_name="fake/dataset")
    assert len(instances) == 2

    summary = swe.run(split="test", limit=3, dataset_name="fake/dataset")
    assert summary["instances"] == 3
    assert summary["repositories"] == 2
    assert summary["top_repos"][0][0] == "repo-a"
    assert summary["dataset_name"] == "fake/dataset"
