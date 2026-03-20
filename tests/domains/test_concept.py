import numpy as np
import pytest
from hpm.domains.concept import ConceptLearningDomain, Concept


@pytest.fixture
def domain():
    concepts = [
        Concept(
            deep_features=np.array([1.0, 0.0]),
            surface_templates=[np.array([0.5, 0.5]), np.array([0.8, 0.2])],
            label=0,
        ),
        Concept(
            deep_features=np.array([0.0, 1.0]),
            surface_templates=[np.array([0.2, 0.8]), np.array([0.3, 0.7])],
            label=1,
        ),
    ]
    return ConceptLearningDomain(concepts, noise=0.01, seed=42)


def test_observe_returns_correct_dim(domain):
    x = domain.observe()
    assert x.shape == (domain.feature_dim(),)


def test_feature_dim_is_surface_plus_deep(domain):
    assert domain.feature_dim() == 4   # 2 surface + 2 deep


def test_deep_perturb_changes_deep_features(domain):
    perturbed = domain.deep_perturb()
    # Deep features should differ
    orig_deep = domain.concepts[0].deep_features
    pert_deep = perturbed.concepts[0].deep_features
    assert not np.allclose(orig_deep, pert_deep)


def test_surface_perturb_preserves_deep_features(domain):
    perturbed = domain.surface_perturb()
    orig_deep = domain.concepts[0].deep_features
    pert_deep = perturbed.concepts[0].deep_features
    assert np.allclose(orig_deep, pert_deep)


def test_transfer_probe_returns_labelled_pairs(domain):
    probes = domain.transfer_probe(near=True)
    assert len(probes) > 0
    x, label = probes[0]
    assert isinstance(x, np.ndarray)
    assert isinstance(label, int)
    assert x.shape == (domain.feature_dim(),)


def test_far_transfer_novel_surface(domain):
    near = domain.transfer_probe(near=True)
    far = domain.transfer_probe(near=False)
    # Far transfer uses novel surface — surface features differ more
    near_surfaces = np.stack([x[:2] for x, _ in near])
    far_surfaces = np.stack([x[:2] for x, _ in far])
    assert far_surfaces.std() > near_surfaces.std()
