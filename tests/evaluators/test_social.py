import pytest
import numpy as np
from hpm.evaluators.social import SocialEvaluator


def test_zero_frequency_gives_zero_signal():
    ev = SocialEvaluator(rho=1.0)
    assert ev.evaluate(freq=0.0) == pytest.approx(0.0)


def test_signal_scales_linearly_with_freq():
    ev = SocialEvaluator(rho=2.0)
    assert ev.evaluate(freq=0.5) == pytest.approx(1.0)
    assert ev.evaluate(freq=1.0) == pytest.approx(2.0)


def test_rho_zero_gives_zero_regardless_of_freq():
    ev = SocialEvaluator(rho=0.0)
    assert ev.evaluate(freq=0.9) == pytest.approx(0.0)


def test_evaluate_all_returns_one_value_per_pattern():
    ev = SocialEvaluator(rho=1.0)
    freqs = [0.0, 0.3, 0.7]
    result = ev.evaluate_all(freqs)
    assert len(result) == 3
    assert result[0] == pytest.approx(0.0)
    assert result[1] == pytest.approx(0.3)
    assert result[2] == pytest.approx(0.7)


def test_evaluate_all_empty_list_returns_empty():
    ev = SocialEvaluator(rho=1.0)
    result = ev.evaluate_all([])
    assert len(result) == 0


def test_evaluate_all_returns_numpy_array():
    ev = SocialEvaluator(rho=2.0)
    result = ev.evaluate_all([0.5, 1.0])
    assert isinstance(result, np.ndarray)
    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(2.0)
