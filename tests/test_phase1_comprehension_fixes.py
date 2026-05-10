from hpm_ai_v5.experiments.run_symbolic_matching_benchmark import get_benchmark_data
from hpm_ai_v5.planning.symbolic_matching import SymbolicPatternMatchingBenchmark
from hpm_ai_v5.experiments.run_atis_benchmark import ATISBenchmark
from hpm_ai_v5.adapter.atis import load_atis


def test_symbolic_matching_benchmark_runs():
    corpus, train, test, distractors = get_benchmark_data()
    benchmark = SymbolicPatternMatchingBenchmark(tool_corpus=corpus)

    benchmark.train(train, epochs=2)
    result = benchmark.run_test(test, distractors)

    assert result.status == "complete"
    assert 0.0 <= result.tool_accuracy <= 1.0
    assert 0.0 <= result.parameter_f1 <= 1.0
    assert 0.0 <= result.distractor_rejection <= 1.0


def test_atis_checkpoint_bundle_preserves_small_slice_accuracy(tmp_path):
    train, test = load_atis()
    train_small = train[:80]
    test_small = test[:20]

    checkpoint_path = tmp_path / "atis_bundle.pkl"

    fresh = ATISBenchmark()
    fresh._CHECKPOINT = str(checkpoint_path)
    acc_fresh = fresh.run_b1(train_small, test_small)

    loaded = ATISBenchmark()
    loaded._CHECKPOINT = str(checkpoint_path)
    acc_loaded = loaded.run_b1(train_small, test_small)

    assert acc_fresh > 0.0
    assert acc_loaded == acc_fresh
