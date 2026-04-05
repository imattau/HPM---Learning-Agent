from hpm_fractal_node.experiments.experiment_competing_explanations import (
    ExperimentConfig,
    run_experiment,
)


def test_early_phase_prefers_simple_explanation():
    summary = run_experiment()
    assert summary.early_cheap_rate >= 0.60


def test_late_phase_shifts_to_correct_structure():
    summary = run_experiment()
    assert summary.late_complex_rate >= 0.60


def test_transition_point_is_detected():
    summary = run_experiment()
    assert summary.transition_step is not None
    assert summary.transition_step > 0
    assert summary.transition_step < summary.samples


def test_complex_weight_overtakes():
    summary = run_experiment()
    assert summary.complex_weight_trace[-1] > summary.cheap_weight_trace[-1]


def test_explicit_stagnation_mode_is_detected():
    summary = run_experiment(
        ExperimentConfig(
            samples=160,
            lambda_complexity=0.18,
            weight_gain=0.30,
            initial_cheap_weight=0.98,
            initial_complex_weight=0.02,
            transition_threshold=0.9,
            evidence_scale=0.0,
        )
    )
    assert summary.transition_step is None
    assert summary.stagnation
