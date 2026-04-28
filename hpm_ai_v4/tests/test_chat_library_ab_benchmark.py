from hpm_ai_v4.simulations import chat_library_ab_benchmark as benchmark_module


def test_chat_library_ab_benchmark_reports_deltas_and_writes_json(tmp_path, monkeypatch):
    class DummyRun:
        def __init__(self, library_path, responses, metrics):
            self.library_path = library_path
            self.responses = responses
            self.response_stats = []
            self.prompts = ["a", "b"]
            self.metrics = metrics

    runs = {
        "small": DummyRun(
            "/tmp/small.pkl",
            ["one", "one"],
            {
                "response_diversity": 0.5,
                "adjacent_repeat_rate": 1.0,
                "avg_length": 3.0,
                "avg_plausibility": 0.2,
                "avg_repeat_score": 0.8,
                "avg_structure_score": 0.1,
                "avg_commonness_score": 0.2,
                "avg_text_signal_score": 0.3,
            },
        ),
        "large": DummyRun(
            "/tmp/large.pkl",
            ["one", "two"],
            {
                "response_diversity": 1.0,
                "adjacent_repeat_rate": 0.0,
                "avg_length": 4.0,
                "avg_plausibility": 0.4,
                "avg_repeat_score": 0.2,
                "avg_structure_score": 0.3,
                "avg_commonness_score": 0.4,
                "avg_text_signal_score": 0.5,
            },
        ),
    }

    def fake_run_single_library(**kwargs):
        if kwargs["library_path"] == "/tmp/small.pkl":
            return runs["small"]
        return runs["large"]

    monkeypatch.setattr(benchmark_module, "_run_single_library", fake_run_single_library)

    report = benchmark_module.run_chat_library_ab_benchmark(
        corpus_path=str(tmp_path / "corpus.txt"),
        prompts=["Hello.", "What can you do?"],
        small_library_path="/tmp/small.pkl",
        large_library_path="/tmp/large.pkl",
        checkpoint_dir=str(tmp_path),
    )

    assert report["small"]["metrics"]["response_diversity"] == 0.5
    assert report["large"]["metrics"]["response_diversity"] == 1.0
    assert report["delta"]["response_diversity"] == 0.5
    assert report["delta"]["adjacent_repeat_rate"] == -1.0
    assert (tmp_path / "chat_library_ab_benchmark.json").exists()
