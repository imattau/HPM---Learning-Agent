import numpy as np
import pytest
from hpm.patterns.gaussian import GaussianPattern


@pytest.fixture
def dim():
    return 4


@pytest.fixture
def simple_pattern(dim):
    return GaussianPattern(mu=np.zeros(dim), sigma=np.eye(dim))


@pytest.fixture
def rng():
    return np.random.default_rng(42)
