from hpm_ai_v6.evaluators.reasoning_evaluator import (
    ReasoningBenchmarkCase,
    ReasoningEvaluator,
)
from hpm_ai_v6.experiments.reasoning_capability_eval import (
    build_synthetic_benchmark_cases,
    build_synthetic_reasoning_agent,
    corpus_pattern_cache_dir,
    extract_story_sentences,
    verify_expected_edges,
)


def test_reasoning_evaluator_scores_synthetic_suite():
    agent = build_synthetic_reasoning_agent()
    evaluator = ReasoningEvaluator(agent)

    report = evaluator.evaluate_cases(build_synthetic_benchmark_cases())

    assert report.total_cases == 5
    assert report.passed_cases == 5
    assert report.accuracy == 1.0
    assert report.by_method()["backward"]["count"] == 3.0


def test_reasoning_evaluator_reports_failures_when_expectations_do_not_match():
    agent = build_synthetic_reasoning_agent()
    evaluator = ReasoningEvaluator(agent)
    case = ReasoningBenchmarkCase(
        name="bad_expectation",
        question="How does alpha connect to delta?",
        method="beam",
        expected_relations=["lexical_transition"],
    )

    result = evaluator.evaluate_case(case)

    assert result.success is False
    assert result.failures
    assert "expected=['lexical_transition']" in result.failures[0]


def test_reasoning_benchmark_report_renders_summary_text():
    agent = build_synthetic_reasoning_agent()
    evaluator = ReasoningEvaluator(agent)

    report = evaluator.evaluate_cases(build_synthetic_benchmark_cases())
    text = report.render_text()

    assert "accuracy=1.000" in text
    assert "beam:" in text
    assert "backward:" in text
    assert "PASS beam_connection" in text


def test_reasoning_evaluator_can_require_path_and_forbid_fallback_language():
    agent = build_synthetic_reasoning_agent()
    evaluator = ReasoningEvaluator(agent)
    case = ReasoningBenchmarkCase(
        name="strict_failure",
        question="How does alpha connect to missinggoal?",
        method="beam",
        require_chosen_path=True,
        forbidden_answer_contains=["strongest learned transitions"],
    )

    result = evaluator.evaluate_case(case)

    assert result.success is False
    assert "chosen_path missing" in result.failures
    assert any("forbidden answer snippet" in failure for failure in result.failures)


def test_richer_story_extractor_keeps_focus_sentence_context(tmp_path):
    corpus_path = tmp_path / "alice.txt"
    corpus_path.write_text(
        "Intro sentence about the world. Alice meets the rabbit. "
        "The rabbit runs toward the hole. Later, the rabbit disappears.",
        encoding="utf-8",
    )

    sentences = extract_story_sentences(str(corpus_path), limit=4)

    assert any("alice meets the rabbit" in sentence.lower() for sentence in sentences)
    assert any("rabbit runs toward the hole" in sentence.lower() for sentence in sentences)


def test_expected_edge_verification_uses_reasoning_agent_resolution():
    agent = build_synthetic_reasoning_agent()
    report = verify_expected_edges(agent, [("alpha", "beta"), ("rabbit", "hole"), ("alpha", "missing")])

    assert ("alpha", "beta") in report["present"]
    assert ("rabbit", "hole") in report["present"]
    assert ("alpha", "missing") in report["missing"]


def test_corpus_pattern_cache_dir_points_at_existing_archive():
    path = corpus_pattern_cache_dir("data/corpus/alice_mini.txt")

    assert path.endswith("data/corpus/.hpm_pattern_cache/alice_mini")
