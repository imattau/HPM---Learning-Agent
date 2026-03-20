import numpy as np
from hpm.patterns.gaussian import GaussianPattern


def test_default_level_is_1():
    p = GaussianPattern(mu=np.zeros(2), sigma=np.eye(2))
    assert p.level == 1


def test_level_settable_via_constructor():
    p = GaussianPattern(mu=np.zeros(2), sigma=np.eye(2), level=3)
    assert p.level == 3


def test_level_settable_on_instance():
    p = GaussianPattern(mu=np.zeros(2), sigma=np.eye(2))
    p.level = 4
    assert p.level == 4


def test_level_preserved_through_update():
    p = GaussianPattern(mu=np.zeros(2), sigma=np.eye(2))
    p.level = 3
    p2 = p.update(np.ones(2))
    assert p2.level == 3


def test_update_also_preserves_id():
    p = GaussianPattern(mu=np.zeros(2), sigma=np.eye(2))
    p.level = 3
    p2 = p.update(np.ones(2))
    assert p2.id == p.id


def test_level_round_trips_through_to_dict_from_dict():
    p = GaussianPattern(mu=np.zeros(2), sigma=np.eye(2), level=4)
    d = p.to_dict()
    assert d['level'] == 4
    p2 = GaussianPattern.from_dict(d)
    assert p2.level == 4


def test_from_dict_defaults_level_to_1_when_key_absent():
    p = GaussianPattern(mu=np.zeros(2), sigma=np.eye(2))
    d = p.to_dict()
    del d['level']
    p2 = GaussianPattern.from_dict(d)
    assert p2.level == 1
